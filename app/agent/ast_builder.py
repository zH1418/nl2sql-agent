from __future__ import annotations

import re
from typing import Any

from sqlglot import exp
from sqlglot.optimizer.qualify import qualify

DIALECT = "mysql"

AGG_WHITELIST = {"SUM", "AVG", "COUNT", "MAX", "MIN", "COUNT_DISTINCT", "DIFF"}

ENABLE_QUALIFY = True

ROLE_PRIMARY_KEY = "primary_key"
ROLE_FOREIGN_KEY = "foreign_key"


REASON_SELECT = "select"        # 选词非法：指标名不在候选 / JSON 结构错 / 聚合白名单外
REASON_STRUCTURE = "structure"  # 结构非法：qualify 拒绝 / where 非法 / AST 构建异常
REASON_BACKBONE = "backbone"    # 主干不足：多事实表 / 引用未 join 的表 ⇒ **模型改不动，需扩圈**


class ASTBuildError(ValueError):

    def __init__(self, message: str, reason_class: str = REASON_STRUCTURE) -> None:
        super().__init__(message)
        self.reason_class = reason_class



def build_schema(table_infos: list[dict]) -> dict[str, dict[str, str]]:
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



def derive_join_chain(store_table: str, table_infos: list[dict]) -> list[tuple[str, str, str]]:
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



def expand_metric(term: Any, metric_map: dict[str, dict] | None) -> str:
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
            parts = [f"SUM({c})" for c in cols]
            return " - ".join(parts) if len(parts) > 1 else parts[0]
        if agg == "COUNT_DISTINCT":
            return f"COUNT(DISTINCT {cols[0]})"
        return f"{agg}({cols[0]})"

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*\(.+\)", name):
        return name
    raise ASTBuildError(f"未知指标概念 {name}，且它不是合法聚合表达式", REASON_SELECT)



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
    m = re.match(r"[A-Za-z_]+\s*\((.+)\)$", agg_exp.strip())
    inner = m.group(1).strip() if m else agg_exp.strip()
    if not inner or inner == "*" or inner.upper().startswith("DISTINCT"):
        return None
    return inner


def _strip_agg(s: str) -> str:
    m = re.fullmatch(r"(?is)(SUM|AVG|COUNT|MAX|MIN)\s*\(\s*(.+?)\s*\)", s.strip())
    return m.group(2).strip() if m else s.strip()


def _referenced_tables(choice: dict, table_infos: list[dict]) -> set[str]:
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



def build_ast_single(
    store_table: str,
    choice: dict,
    schema: dict[str, dict[str, str]],
    join_chain: list[tuple[str, str, str]],
    metric_map: dict[str, dict] | None = None,
) -> exp.Select:
    from_table = store_table or (join_chain[0][0] if join_chain else "")
    if not from_table:
        raise ASTBuildError("未指定事实表，无法构建 AST", REASON_SELECT)

    sel = exp.Select().from_(from_table)

    for star, key, dim in join_chain:
        sel = sel.join(dim, on=f"{star}.{key} = {dim}.{key}")

    group_keys = list(choice.get("group_by") or [])
    group_bare = {_bare(g) for g in group_keys}
    for g in group_keys:
        sel = sel.select(g)
        sel = sel.group_by(g)

    for col in choice.get("select") or []:
        if _bare(col) in group_bare:
            continue
        sel = sel.select(str(col))

    for m in choice.get("metrics") or []:
        expanded = expand_metric(m, metric_map)
        if _bare(expanded) in group_bare:
            continue
        alias = str(m.get("name") or "").strip() if isinstance(m, dict) else ""
        if alias and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", alias):
            sel = sel.select(f"{expanded} AS {alias}")
        else:
            sel = sel.select(expanded)

    for w in choice.get("where") or []:
        if not isinstance(w, dict) or not w.get("column"):
            raise ASTBuildError(f"where 项缺少 column：{w!r}", REASON_STRUCTURE)
        op = str(w.get("op") or "=").strip()
        if op not in ("=", "!=", ">", "<", ">=", "<=", "like", "LIKE", "in", "IN"):
            raise ASTBuildError(f"where 运算符 {op!r} 不在允许集合内", REASON_STRUCTURE)
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

    ob = choice.get("order_by") or {}
    if isinstance(ob, dict) and ob.get("column"):
        col = exp.to_column(expand_metric(ob["column"], metric_map))
        col = col.asc() if bool(ob.get("asc", True)) else col.desc()
        sel = sel.order_by(col)

    lim = choice.get("limit")
    if isinstance(lim, int) and lim > 0:
        sel = sel.limit(lim)

    if ENABLE_QUALIFY:
        try:
            qualify(sel.copy(), schema=schema, dialect=DIALECT, validate_qualify_columns=True)
        except Exception as e:
            raise ASTBuildError(
                f"qualify 校验失败（疑似编造列/表或结构非法）：{type(e).__name__}: {e}",
                REASON_STRUCTURE,
            ) from e
    return sel



def build_sql(
    choice: dict,
    table_infos: list[dict],
    metric_infos: list[dict],
    dialect: str = DIALECT,
) -> str:
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

    refs = _referenced_tables(choice, table_infos)
    chain = [edge for edge in derive_join_chain(store, table_infos) if edge[2] in refs]

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
