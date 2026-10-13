"""SQL 执行链路支撑：join 键/指标列补全 + L2 执行探针。

由 v1 的 LLM 自由生成 SQL 节点演化而来，语义保持不变：
- `ensure_join_keys`      —— 上游裁列可能裁掉纯 join 键，回查 meta 库补齐（**只补主外键**）
- `ensure_metric_columns` —— 只补「**本次选中的指标**」真正引用的列（不破坏语义过滤）
- `execute_probe`         —— ★ L2 探针：以 `LIMIT cap+1` 执行，**返回行数以判定命中/截断**

★ L2 探针为什么取代 `EXPLAIN`：
AST 构建已保证语法合法、`qualify` 已拦住编造列，`EXPLAIN` 与之**完全重叠**，
却拦不住真正要拦的「0 行 / 逻辑答非所问」。
"""

from __future__ import annotations

import sqlglot
from langgraph.runtime import Runtime

from app.agent.state import ColumnInfoState
from app.core.log import logger

# 探针上限 ——单次查询返回的行数上限，兼作截断检测阈值
DEFAULT_CAP: int = 1000


# ---------------------------------------------------------------- 限流改写

def with_limit(sql: str, limit: int, dialect: str = "mysql") -> str:
    """把 SQL 的顶层 LIMIT 设为 `limit`（用 sqlglot 改写，避免拼接出 `LIMIT a LIMIT b`）。"""
    try:
        ast = sqlglot.parse_one(sql, dialect=dialect)
    except Exception as e:  # 解析失败则退化为字符串拼接（保守兜底）
        logger.warning("probe 改写 LIMIT 解析失败，退化为拼接：{}", e)
        return f"{sql} LIMIT {limit}"
    if isinstance(ast, sqlglot.exp.Select):
        ast.set("limit", sqlglot.exp.Limit(expression=sqlglot.exp.Literal.number(limit)))
        return ast.sql(dialect=dialect)
    return f"{sql} LIMIT {limit}"


def _model_limit(sql: str, dialect: str = "mysql") -> int | None:
    """取 SQL 自带的顶层 LIMIT（若有）。"""
    try:
        ast = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return None
    if isinstance(ast, sqlglot.exp.Select) and ast.args.get("limit"):
        try:
            return int(ast.args["limit"].expression.this)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------- join 键补全

async def ensure_join_keys(table_infos: list[dict], meta_repo) -> list[dict]:
    """把每张表的主键/外键列补回 `table_infos`（join 必需）。

    为什么需要：上游 `filter_table` 按语义裁列时，可能把 `fact_order.region_id` /
    `dim_region.region_id` 这类「纯 join 键」裁掉，导致主干推导失败。
    此处按表名回查 meta 库补齐 `role in (primary_key, foreign_key)` 的列，
    **不补 measure/dimension** —— 不改变 `filter_table` 的语义过滤效果。
    """
    if not meta_repo or not table_infos:
        return table_infos
    out: list[dict] = []
    for t in table_infos:
        try:
            keys = await meta_repo.get_column_key_by_id(t.get("name"))
        except Exception as e:  # 环境异常不应让整条链路炸掉
            logger.warning("join 键补全失败，跳过 {}：{}", t.get("name"), e)
            keys = []
        if not keys:
            logger.warning(
                f"join 键补全：表 {t.get('name')} 未查到主键/外键"
                f"（若表名与 table_info.id 不一致，请改用 table_id 回查）"
            )
        have = {c.get("name") for c in (t.get("columns") or [])}
        extra = [
            ColumnInfoState(
                name=k.name, type=k.type, role=k.role,
                examples=k.examples or [], description=k.description or "",
                alias=k.alias or [],
            )
            for k in (keys or [])
            if getattr(k, "name", None) and k.name not in have
        ]
        out.append({**t, "columns": list(t.get("columns") or []) + extra} if extra else t)
    return out


# ---------------------------------------------------------------- 指标列补全

async def ensure_metric_columns(
    choice: dict, table_infos: list[dict], metric_infos: list[dict], meta_repo,
) -> list[dict]:
    """确保「被选中指标」引用的列存在于候选表里（选词之后、构建之前）。

    与 `ensure_join_keys` 同源，区别：只补**本次选中的指标**真正引用的列
    （不是全量候选指标），不破坏语义过滤。
    """
    if not meta_repo or not choice:
        return table_infos

    mmap: dict[str, list[str]] = {}
    for m in metric_infos or []:
        name = (m.get("name") or "").strip()
        if not name:
            continue
        cols = m.get("relevant_columns") or []
        if isinstance(cols, str):
            cols = [cols]
        mmap[name] = [str(c) for c in cols if c]

    need: set[str] = set()
    for m in choice.get("metrics") or []:
        name = str(m.get("name") if isinstance(m, dict) else m).strip()
        for c in mmap.get(name, []):
            need.add(c)
    if not need:
        return table_infos

    out: list[dict] = []
    for t in table_infos or []:
        nt = dict(t)
        nt["columns"] = list(t.get("columns") or [])
        out.append(nt)
    by_name = {t.get("name"): t for t in out}

    for col_id in need:
        if "." not in col_id:
            continue
        table_name, col_name = col_id.split(".", 1)
        table = by_name.get(table_name)
        if table is None:
            continue
        if col_name in {c.get("name") for c in table["columns"]}:
            continue
        try:
            info = await meta_repo.get_column_info_by_id(col_id)
        except Exception as e:
            logger.warning("指标列补全失败，跳过 {}：{}", col_id, e)
            info = None
        if info is None:
            logger.warning(f"指标列补全：meta 库查不到 {col_id}")
            continue
        table["columns"].append(ColumnInfoState(
            name=getattr(info, "name", col_name),
            type=getattr(info, "type", "") or "",
            role=getattr(info, "role", "") or "",
            examples=getattr(info, "examples", None) or [],
            description=getattr(info, "description", "") or "",
            alias=getattr(info, "alias", None) or [],
        ))
        logger.info(f"指标列补全：补回 {col_id}")
    return out


# ---------------------------------------------------------------- L2 执行探针

class ProbeResult:
    """L2 探针结果。"""

    __slots__ = ("hit", "rows", "truncated", "sql", "failure_class", "error")

    def __init__(self, *, hit: bool, rows: list[dict] | None = None,
                 truncated: bool = False, sql: str = "",
                 failure_class: str | None = None, error: str | None = None) -> None:
        self.hit = hit
        self.rows = rows or []
        self.truncated = truncated
        self.sql = sql
        self.failure_class = failure_class
        self.error = error

    @property
    def row_count(self) -> int:
        return len(self.rows)


async def execute_probe(
    sql: str, dw_mysql_repository, cap: int = DEFAULT_CAP, dialect: str = "mysql",
) -> ProbeResult:
    """★ L2 探针：执行并返回行数，判定「命中 / 空 / 报错 / 截断」。

    策略：
    - 若 SQL 自带 `LIMIT n` 且 `n <= cap+1` ⇒ 用 `n` 探测（模型自己限了行，不算截断）
    - 否则用 `cap+1` 探测 ⇒ 恰好返回 `cap+1` 行即说明**被截断**，再取全量
    - `rows == 0` ⇒ `empty`（0 行，模型可核对 where 值）
    - DB 报错 ⇒ `dberror`

    ⚠️ `rows == cap+1` 时本函数会**再执行一次无上限查询**取全量，
    保证「命中即拿到完整结果」，模型侧完全不需要感知「探针」概念。
    """
    model_limit = _model_limit(sql, dialect)
    probe_limit = min(model_limit, cap + 1) if model_limit else cap + 1

    try:
        rows = await dw_mysql_repository.execute_sql(with_limit(sql, probe_limit, dialect))
    except Exception as e:
        logger.warning("L2 探针执行报错：{}: {}", type(e).__name__, e)
        return ProbeResult(hit=False, sql=sql, failure_class="dberror",
                           error=f"{type(e).__name__}: {e}")

    if not rows:
        return ProbeResult(hit=False, sql=sql, failure_class="empty", error="查询通过但 0 行返回")

    # 只有「探针上限就是 cap+1」且刚好取满，才是真被截断（模型自带 limit 不算）
    truncated = (probe_limit == cap + 1) and (len(rows) == cap + 1)
    if truncated:
        try:
            full = await dw_mysql_repository.execute_sql(
                with_limit(sql, model_limit, dialect) if model_limit else sql
            )
            rows = full
            logger.info(f"探针截断，已取全量：{len(rows)} 行")
        except Exception as e:  # 取全量失败不该丢掉已有结果
            logger.warning("截断后取全量失败，回退探针结果：{}", e)

    return ProbeResult(hit=True, rows=rows, truncated=truncated, sql=sql)
