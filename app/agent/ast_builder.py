"""AST 选词主线 —— 程序化构建 SQL（MVP / L0）。

对照旧版 `generate_sql`（LLM 自由拼 SQL 字符串）：
    旧：query + table_infos/metric_infos → LLM → SQL 文本（可能编造表/列/聚合）
    新：候选装配 → LLM 选词(结构化 JSON) → sqlglot 程序化构建 AST → SQL

本模块只做「纯函数」的 AST 构建，不碰 LLM、不碰数据库连接，
便于离线单测（拿 meta_config.yaml 的真实表结构即可验证）。

设计对照：
    候选装配   = table_infos(过滤后) + metric_infos + 列示例值 + 当前时间
    选词       = LLM 只输出 JSON（stores / select / metrics / group_by / where / order_by / having / limit）
    主干推导   = 程序按星型关系求 join 链（外键列名 == 维度表主键列名），不让 LLM 编
    构建       = sqlglot `exp.Select()` 逐段拼装，语法天然合法
    出口锁     = 所有产出统一过 qualify 列校验，编造列直接抛

MVP 范围（L0）：
- 唯一主干推导：星型模型（唯一事实表 + N 个维度外键），由选中的 store 表自动求 join 链。
- 指标展开：LLM 选「指标概念 + 聚合方式」，程序从 `metric_info.relevant_columns`
  展开成真实聚合列，杜绝编造列。
- 多事实表（派生指标相减 / 子查询拼接）、L1 扩圈、L2 全量兜底：本轮不做。
"""
from __future__ import annotations

import re
from typing import Any

from sqlglot import exp
from sqlglot.optimizer.qualify import qualify

# SQL 方言：数仓固定 MySQL（调用方可传入 db_info.dialect 覆盖）
DIALECT = "mysql"

# 聚合白名单：LLM 只能从这里选聚合方式，白名单外一律拒绝（不静默降级）
AGG_WHITELIST = {"SUM", "AVG", "COUNT", "MAX", "MIN", "COUNT_DISTINCT", "DIFF"}

# 出口锁：统一过 qualify 列校验（编造列/不存在的表直接抛）。
# 设为 False 可跳过校验，但那样就失去了「死亡编造兜底」，仅建议在 schema 不全时临时关闭。
ENABLE_QUALIFY = True

# 列角色（来自 meta 库 column_info.role）
ROLE_PRIMARY_KEY = "primary_key"
ROLE_FOREIGN_KEY = "foreign_key"


# 失败归因（供上层区分「模型可修正」与「需扩圈」）
REASON_SELECT = "select"        # 选词非法：指标名不在候选 / JSON 结构错 / 聚合白名单外
REASON_STRUCTURE = "structure"  # 结构非法：qualify 拒绝 / where 非法 / AST 构建异常
REASON_BACKBONE = "backbone"    # 主干不足：多事实表 / 引用未 join 的表 ⇒ **模型改不动，需扩圈**


class ASTBuildError(ValueError):
    """AST 构建失败：选词结构非法 / 主干推导不出 / qualify 拒绝。

    ★ 带**失败归因** `reason_class`（`select` / `structure` / `backbone`）：
    上层（`build_and_execute` 工具）据此决定「喂什么反馈给模型重试」还是
    「直接终止并标记 `needs_next="L1"`」（`backbone` 属后者 —— 候选集合不足，
    重选词救不回来）。

    兼容性：仍是 `ValueError` 子类，既有 `except ASTBuildError` 不受影响；
    `reason_class` 默认为 `structure`（最保守的「模型可尝试修正」类）。
    """

    def __init__(self, message: str, reason_class: str = REASON_STRUCTURE) -> None:
        super().__init__(message)
        self.reason_class = reason_class


# ---------------------------------------------------------------- schema / 指标字典

def build_schema(table_infos: list[dict]) -> dict[str, dict[str, str]]:
    """表信息 → qualify 用的 schema：{表名: {列名: 类型}}。

    schema 来源 = 本次候选的表信息（meta 库反查得来），非硬编码。
    """
    schema: dict[str, dict[str, str]] = {}
    for t in table_infos or []:
        cols = {
            c["name"]: (c.get("type") or "VARCHAR")
            for c in (t.get("columns") or [])
            if c.get("name")
        }
        if cols:
            schema[t["name"]] = cols
    return schema


def build_metric_map(metric_infos: list[dict]) -> dict[str, dict]:
    """指标信息 → 指标字典：{指标名: {columns: [...], description, alias}}。

    说明：metric_info 表**没有 agg_type 字段**，所以聚合方式由 LLM 在选词时给出，
    程序只负责校验白名单 + 从 `relevant_columns` 展开真实列 —— 「聚合方式可选，列不可编」。
    """
    out: dict[str, dict] = {}
    for m in metric_infos or []:
        name = (m.get("name") or "").strip()
        if not name:
            continue
        cols = m.get("relevant_columns") or []
        if isinstance(cols, str):
            cols = [cols]
        cols = [c for c in cols if c]
        if not cols:
            continue
        out[name] = {
            "name": name,
            "columns": cols,
            "description": m.get("description") or "",
            "alias": m.get("alias") or [],
        }
    return out


# ---------------------------------------------------------------- 主干推导（星心 + 端点 → join 链）

def derive_join_chain(store_table: str, table_infos: list[dict]) -> list[tuple[str, str, str]]:
    """L0 唯一主干：星型模型下 store(事实表) 连各维度辐条。

    返回 [(star_table, join_column, dim_table)]。

    推导规则（程序化，不让 LLM 编 join）：
        事实表侧取外键列（role=foreign_key），维度表侧取主键列（role=primary_key），
        列名相同即视为一条 join 边。这是星型数仓的标准命名约定
        （fact_order.customer_id → dim_customer.customer_id 等）。

    兜底：外键列被上游 filter_table 裁掉时，退化为「用事实表全部列名去匹配维表主键列名」
    （主键列名在各维表内唯一，误配概率低）。仍推不出则返回空，由调用方明确报错。
    """
    by_name = {t.get("name"): t for t in (table_infos or [])}
    store = by_name.get(store_table)
    if not store:
        return []

    fk_cols = [
        c["name"] for c in (store.get("columns") or [])
        if c.get("role") == ROLE_FOREIGN_KEY and c.get("name")
    ]
    store_keys = fk_cols or [c["name"] for c in (store.get("columns") or []) if c.get("name")]

    chain: list[tuple[str, str, str]] = []
    for t in table_infos or []:
        dim = t.get("name")
        if not dim or dim == store_table:
            continue
        dim_pks = [
            c["name"] for c in (t.get("columns") or [])
            if c.get("role") == ROLE_PRIMARY_KEY and c.get("name")
        ]
        for key in store_keys:
            if key in dim_pks:
                chain.append((store_table, key, dim))
                break  # 星型下「事实表 → 单张维表」只连一条边
    return chain


# ---------------------------------------------------------------- 指标展开

def expand_metric(term: Any, metric_map: dict[str, dict] | None) -> str:
    """指标概念 → 真实聚合表达式。

    支持两种选词形态：
        {"name": "GMV", "agg": "SUM"}   标准形态（推荐，聚合方式由 LLM 按业务语义选）
        "GMV"                            纯概念名，聚合方式缺省 SUM

    列一律从 `metric_map[name]["columns"]`（即 meta 库 relevant_columns）展开，
    LLM 不能直接塞列名；若显式给了 column，也必须落在 relevant_columns 之内。
    """
    if isinstance(term, dict):
        name = str(term.get("name") or "").strip()
        agg = str(term.get("agg") or "").strip().upper()
        raw_col = str(term.get("column") or "").strip()
    else:
        name, agg, raw_col = str(term).strip(), "", ""

    if not name:
        raise ASTBuildError(f"metrics 项缺少指标名：{term!r}", REASON_SELECT)

    if metric_map and name in metric_map:
        me = metric_map[name]
        cols = list(me["columns"])
        if raw_col:
            if raw_col not in cols:
                raise ASTBuildError(
                    f"指标 {name} 指定的列 {raw_col} 不在其 relevant_columns={cols} 内",
                    REASON_SELECT,
                )
            cols = [raw_col]
        agg = agg or str(me.get("agg") or "").strip().upper() or "SUM"
        if agg not in AGG_WHITELIST:
            raise ASTBuildError(
                f"指标 {name} 的聚合方式 {agg} 不在白名单 {sorted(AGG_WHITELIST)}",
                REASON_SELECT,
            )
        if agg == "DIFF":
            # 派生指标：各列各自 SUM 后按 relevant_columns 顺序相减（净销售额语义）
            parts = [f"SUM({c})" for c in cols]
            return " - ".join(parts) if len(parts) > 1 else parts[0]
        if agg == "COUNT_DISTINCT":
            return f"COUNT(DISTINCT {cols[0]})"
        return f"{agg}({cols[0]})"

    # 非指标概念：只放行合法聚合表达式（如 SUM(fact_order.order_amount)），否则拒绝
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*\(.+\)", name):
        return name
    raise ASTBuildError(f"未知指标概念 {name}，且它不是合法聚合表达式", REASON_SELECT)


# ---------------------------------------------------------------- 字面量 / 小工具

def _sql_literal(value: Any) -> str:
    """where / having 的值 → SQL 字面量：数字裸写，字符串加单引号并转义。"""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value).strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return s
    return "'" + s.replace("'", "''") + "'"


def _bare(col: str) -> str:
    """取列名末段（`dim_region.region_name` → `region_name`），用于重复列判重。"""
    return str(col).split(".")[-1]


def _table_of(col: str) -> str:
    """取列的表前缀（`dim_region.region_name` → `dim_region`）；无前缀返回空串。"""
    s = str(col)
    return s.split(".", 1)[0] if "." in s else ""


def _stores_of(choice: dict) -> list[str]:
    """兼容 stores（列表）与 store（单值）两种选词形态。"""
    raw = choice.get("stores") or ([choice["store"]] if choice.get("store") else [])
    return [str(s).strip() for s in raw if str(s).strip()]


def _hav_avg_col(agg_exp: str) -> str | None:
    """从聚合表达式抽基础列：SUM(fact_order.order_amount) → fact_order.order_amount。

    COUNT(*) / COUNT(DISTINCT ...) 无单一列可言 → 返回 None（不生成非法标量子查询）。
    """
    m = re.match(r"[A-Za-z_]+\s*\((.+)\)$", agg_exp.strip())
    inner = m.group(1).strip() if m else agg_exp.strip()
    if not inner or inner == "*" or inner.upper().startswith("DISTINCT"):
        return None
    return inner


def _strip_agg(s: str) -> str:
    """剥掉聚合函数外壳：`SUM(fact_order.order_quantity)` → `fact_order.order_quantity`。

    ★ 不剥会把 `SUM(fact_order` 当成**表名**（见 `_referenced_tables` 的说明）。
    """
    m = re.fullmatch(r"(?is)(SUM|AVG|COUNT|MAX|MIN)\s*\(\s*(.+?)\s*\)", s.strip())
    return m.group(2).strip() if m else s.strip()


def _referenced_tables(choice: dict, table_infos: list[dict]) -> set[str]:
    """选词真正引用到的表集合。

    - 带表前缀的列（dim_region.region_name）→ 直接取前缀；
    - 裸列名（region_name）→ 按候选表反查归属，**唯一命中**才采纳
      （多表同名则留给 qualify 报错，不猜）。

    ★ 两个必须处理的形态（否则会**误报 backbone**）：
    1. **聚合表达式**：`order_by.column` 常被模型写成 `SUM(fact_order.order_quantity)`
       ⇒ 必须先剥函数外壳，否则 `_table_of` 会把 `SUM(fact_order` 当表名；
    2. **只认已知表**：前缀不在候选表集合内时**忽略**（那是函数名/笔误，
       不是表引用；真的引用了不存在的表会由 `qualify` 出口锁拦下）。
    """
    known = {t.get("name") for t in (table_infos or [])}

    by_col: dict[str, set[str]] = {}
    for t in table_infos or []:
        for c in t.get("columns") or []:
            if c.get("name"):
                by_col.setdefault(c["name"], set()).add(t.get("name"))

    items: list[Any] = list(choice.get("select") or []) + list(choice.get("group_by") or [])
    for w in choice.get("where") or []:
        if isinstance(w, dict):
            items.append(w.get("column"))
    ob = choice.get("order_by") or {}
    if isinstance(ob, dict):
        items.append(ob.get("column"))

    refs: set[str] = set()
    for it in items:
        s = _strip_agg(str(it or ""))
        if not s:
            continue
        tab = _table_of(s)
        if tab and tab in known:          # ★ 只认已知表
            refs.add(tab)
            continue
        owners = by_col.get(s) or set()
        if len(owners) == 1:
            refs.add(next(iter(owners)))
    return refs


# ---------------------------------------------------------------- AST 构建（单主干）

def build_ast_single(
    store_table: str,
    choice: dict,
    schema: dict[str, dict[str, str]],
    join_chain: list[tuple[str, str, str]],
    metric_map: dict[str, dict] | None = None,
) -> exp.Select:
    """L0 单事实表：由选词 JSON + join 链构建 SQLGlot AST（语法天然合法）。

    列序规范：分组维度列在前、聚合在后。
    原因：① 维度列必须进 select（LLM 只给 group_by 时补齐）；
         ② 必须排在聚合前面，否则列序反转，与人工参考 SQL 不一致。
    """
    from_table = store_table or (join_chain[0][0] if join_chain else "")
    if not from_table:
        raise ASTBuildError("未指定事实表，无法构建 AST", REASON_SELECT)

    sel = exp.Select().from_(from_table)

    # joins：星心连各维度（join 条件由程序按主外键同名约定生成）
    for star, key, dim in join_chain:
        sel = sel.join(dim, on=f"{star}.{key} = {dim}.{key}")

    # ① 分组维度列：进 select 且排在聚合之前
    group_keys = list(choice.get("group_by") or [])
    group_bare = {_bare(g) for g in group_keys}
    for g in group_keys:
        sel = sel.select(g)
        sel = sel.group_by(g)

    # ② 明细列（跳过与分组维度重复的项，防重复列）
    for col in choice.get("select") or []:
        if _bare(col) in group_bare:
            continue
        sel = sel.select(str(col))

    # ③ 聚合列（指标概念在此展开成真实聚合表达式；别名用指标概念名，结果表头更友好）
    for m in choice.get("metrics") or []:
        expanded = expand_metric(m, metric_map)
        if _bare(expanded) in group_bare:
            continue
        alias = str(m.get("name") or "").strip() if isinstance(m, dict) else ""
        if alias and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", alias):
            sel = sel.select(f"{expanded} AS {alias}")
        else:
            sel = sel.select(expanded)

    # ④ where 过滤
    for w in choice.get("where") or []:
        if not isinstance(w, dict) or not w.get("column"):
            raise ASTBuildError(f"where 项缺少 column：{w!r}", REASON_STRUCTURE)
        op = str(w.get("op") or "=").strip()
        if op not in ("=", "!=", ">", "<", ">=", "<=", "like", "LIKE", "in", "IN"):
            raise ASTBuildError(f"where 运算符 {op!r} 不在允许集合内", REASON_STRUCTURE)
        # ★ `IN` 的右值必须是**列表**（`region_name IN ('华东','华南')`）。
        #   历史缺陷：`_sql_literal` 对列表会退化成 `str(list)` ⇒ 产出 `'[''华东'', ''华南'']'` 这种垃圾，
        #   于是「同一列多个取值」**根本表达不出来**，模型只能硬凑 `a='x' AND a='y'`（永假）。
        if op.upper() == "IN":
            raw_val = w.get("value", "")
            values = list(raw_val) if isinstance(raw_val, (list, tuple, set)) else [raw_val]
            values = [v for v in values if str(v).strip() != ""]
            if not values:
                raise ASTBuildError(f"where 用 IN 但取值列表为空：{w!r}", REASON_STRUCTURE)
            rhs = "(" + ", ".join(_sql_literal(v) for v in values) + ")"
        else:
            rhs = _sql_literal(w.get("value", ""))
        sel = sel.where(f"{w['column']} {op} {rhs}")

    # ⑤ having：分组后做组间比较（如「高于/低于全站平均」）→ compare=avg 拼标量子查询
    hv = choice.get("having") or {}
    if isinstance(hv, dict) and hv.get("agg"):
        agg_exp = expand_metric(str(hv.get("agg")), metric_map)
        op = str(hv.get("op") or ">")
        compare = str(hv.get("compare") or "")
        if compare in ("avg", "AVG", "全站平均", "平均", "均值"):
            base = _hav_avg_col(agg_exp)
            right = f"(SELECT AVG({base}) FROM {from_table})" if base else compare
        else:
            right = _sql_literal(compare) if compare else "'?'"
        sel = sel.having(f"{agg_exp} {op} {right}")

    # ⑥ order_by / limit
    ob = choice.get("order_by") or {}
    if isinstance(ob, dict) and ob.get("column"):
        col = exp.to_column(expand_metric(ob["column"], metric_map))
        col = col.asc() if bool(ob.get("asc", True)) else col.desc()
        sel = sel.order_by(col)

    lim = choice.get("limit")
    if isinstance(lim, int) and lim > 0:
        sel = sel.limit(lim)

    # 出口锁：编造列 / 不存在的表在此被拦下。
    # ⚠️ sqlglot 的 qualify 是 **in-place** 修改（会给聚合补内部别名 `_col_N`、给标识符加引号），
    #    这里只把它当「校验器」用，故传 copy —— 保证对外产出的 SQL 保持干净。
    if ENABLE_QUALIFY:
        try:
            qualify(sel.copy(), schema=schema, dialect=DIALECT, validate_qualify_columns=True)
        except Exception as e:
            raise ASTBuildError(
                f"qualify 校验失败（疑似编造列/表或结构非法）：{type(e).__name__}: {e}",
                REASON_STRUCTURE,
            ) from e
    return sel


# ---------------------------------------------------------------- 对外主入口

def build_sql(
    choice: dict,
    table_infos: list[dict],
    metric_infos: list[dict],
    dialect: str = DIALECT,
) -> str:
    """选词 JSON + 表/指标信息 → SQL 字符串（AST 渲染）。

    Raises:
        ASTBuildError: 选词非法、主干推导不出、引用了未 join 的表、qualify 拒绝。
    """
    if not isinstance(choice, dict):
        raise ASTBuildError(f"选词结果不是 JSON 对象：{type(choice).__name__}", REASON_SELECT)

    schema = build_schema(table_infos)
    metric_map = build_metric_map(metric_infos)

    stores = _stores_of(choice)
    if not stores:
        raise ASTBuildError("选词未给出事实表(stores/store)，拒绝静默兜底", REASON_SELECT)
    if len(stores) > 1:
        raise ASTBuildError(
            f"MVP 仅支持单事实表主干，收到 stores={stores}（多主干子查询拼接后延）",
            REASON_BACKBONE,
        )
    store = stores[0]

    # 主干推导：允许空链 —— 只用事实表本身（如「平均订单金额」不需要 join 任何维表）。
    # 只保留「选词真正引用到」的维表：避免无关维表被 INNER JOIN 进来
    # （干净数仓下虽不改变结果，但维表缺记录时会丢行）。
    refs = _referenced_tables(choice, table_infos)
    chain = [edge for edge in derive_join_chain(store, table_infos) if edge[2] in refs]

    # 覆盖性校验：选词引用到的表必须在 {事实表} ∪ {已 join 维表} 内（不静默丢列）
    covered = {store} | {dim for _, _, dim in chain}
    missing = refs - covered
    if missing:
        raise ASTBuildError(
            f"选词引用了未 join 的表 {sorted(missing)}（当前主干覆盖：{sorted(covered)}）；"
            f"若这些表确实在候选里，通常是其主键列缺失导致 join 链推导失败",
            REASON_BACKBONE,
        )

    ast = build_ast_single(store, choice, schema, chain, metric_map)
    return ast.sql(dialect=dialect)
