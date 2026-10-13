"""召回链路（普通函数版）—— 从原 6 个 LangGraph 节点抽取，供 `create_agent` 工具调用。

★ 与原节点的**三处关键差异**：

1. **不再依赖 `Runtime` / `DataAgentState`** —— 改为显式传参，便于单元测试与复用。
2. ★ **不再调用 LLM 扩展关键词** —— 关键词由**模型**通过工具参数提供
   （原 `recall_utils.expand_keywords` 的那次 LLM 调用被**省掉**，
   因为「纠错 / 扩写 / 改写」已移到模型填参数时完成）。
3. **stage 事件走 `emit` 回调** —— 由工具传入 `runtime.stream_writer`，
   本模块自身不依赖 langgraph 运行时（保持可离线单测）。

⚠️ **分数口径差异（重要）**：
- Qdrant（列 / 指标）相似度 = **COSINE，取值 0~1** ⇒ 可用 0.85 阈值判定「命中够不够强」。
- ES（取值）分数 = **BM25，取值无上界** ⇒ **不可**与 0.85 直接比较，
  故取值一路只用「**有没有命中**」参与判定。
  ⇒ `judge_verdict` 的阈值**只作用于向量路**，这点在函数文档里写明。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Awaitable, Callable, Iterable, Sequence

import jieba.analyse

from app.agent.state import (
    ColumnInfoState,
    DateInfoState,
    DBInfoState,
    MetricInfoState,
    TableInfoState,
)
from app.agent.utils.llm_utils import invoke_llm_chain
from app.core.log import logger
from app.models.mysql.column_info_mysql import ColumnInfoMySQL

# ============================================================ 常量

# L1 探针「命中低」阈值（仅对**向量路**有效，见模块 docstring）
HIT_THRESHOLD: float = 0.85

# jieba 允许的词性（与原 extract_keywords 完全一致）
ALLOW_POS: tuple[str, ...] = (
    "n", "nr", "ns", "nt", "nz", "v", "vn", "a", "an", "eng", "i", "l",
)

Emit = Callable[[dict], None] | None


def _noop_emit(_: dict) -> None:
    """默认 emit：什么都不做（便于无 UI 场景调用）。"""


# ============================================================ ① 关键词（jieba）

def extract_keywords(query: str) -> list[str]:
    """从用户提问提取关键词（jieba TF-IDF + 词性过滤 + 原句兜底）。

    保留原 `extract_keywords` 节点的行为：
    - `jieba.analyse.extract_tags(query, allowPOS=...)`
    - **把原句也加入**（避免分词后关键信息丢失），并去重。

    ⚠️ 注意：本函数**只做分词**，不做 LLM 扩展 —— 扩展由模型通过工具参数完成。
    """
    keywords: list[str] = jieba.analyse.extract_tags(query, allowPOS=ALLOW_POS)
    return list(set(keywords + [query]))


# ============================================================ ② 三路召回

async def _search_all(
    keywords: Sequence[str],
    search_one: Callable[[str], Awaitable[list[tuple[dict, float]]]],
) -> tuple[list[dict], list[float]]:
    """逐词检索 + 按 `id` 去重合并，**保留每个 payload 的最好分数**。

    返回 `(payloads, scores)`，两者按下标对齐。
    """
    best: dict[str, tuple[dict, float]] = {}
    for kw in keywords:
        try:
            hits = await search_one(kw)
        except Exception as e:  # 单个关键词失败不应中断整路召回
            logger.warning("召回单关键词失败，跳过 {!r}：{}", kw, e)
            continue
        for payload, score in hits:
            pid = payload.get("id")
            if pid is None:
                continue
            cur = best.get(pid)
            if cur is None or score > cur[1]:
                best[pid] = (payload, float(score))

    payloads = [v[0] for v in best.values()]
    scores = [v[1] for v in best.values()]
    return payloads, scores


async def recall_columns(keywords: Sequence[str], *, embedding_client, repository):
    """召回字段候选（Qdrant 语义检索）。返回 `(payloads, scores)`。"""

    async def _one(kw: str):
        emb = await embedding_client.aembed_query(kw)
        return await repository.search_scored(emb)

    return await _search_all(keywords, _one)


async def recall_metrics(keywords: Sequence[str], *, embedding_client, repository):
    """召回指标候选（Qdrant 语义检索）。返回 `(payloads, scores)`。"""

    async def _one(kw: str):
        emb = await embedding_client.aembed_query(kw)
        return await repository.search_scored(emb)

    return await _search_all(keywords, _one)


async def recall_values(keywords: Sequence[str], *, repository):
    """召回取值候选（ES 全文检索）。返回 `(payloads, scores)`。

    ⚠️ 分数为 BM25（无界），不参与 0.85 阈值判定。
    """
    return await _search_all(keywords, repository.search_scored)


# ============================================================ ③ 合并与补全

def _convert_mysql_column_to_qdrant(column_info: ColumnInfoMySQL) -> dict:
    """MySQL 字段模型 → 召回侧字段结构（字段一一对应，显式映射便于维护）。"""
    return {
        "id": column_info.id,
        "name": column_info.name,
        "table_id": column_info.table_id,
        "role": column_info.role,
        "examples": column_info.examples,
        "description": column_info.description,
        "alias": column_info.alias,
        "type": column_info.type,
    }


async def _get_or_fetch_column(column_id: str, columns_map: dict, meta_repo) -> dict:
    """按 `表名.列名` 取字段；缺失则回查 meta 库并填回映射（原地修改 `columns_map`）。"""
    if column_id not in columns_map:
        mysql_col = await meta_repo.get_column_info_by_id(column_id)
        columns_map[column_id] = _convert_mysql_column_to_qdrant(mysql_col)
    return columns_map[column_id]


async def merge_retrieved(
    *,
    retrieved_columns: list[dict],
    retrieved_metrics: list[dict],
    retrieved_values: list[dict],
    meta_mysql_repository,
) -> tuple[list[TableInfoState], list[MetricInfoState]]:
    """把三路召回结果合并为下游所需的 `table_infos` + `metric_infos`。

    完整保留原 `merge_retrieved_info` 节点的 6 个子步骤：
    1. 以 `id`（表名.列名）为 key 建字段映射
    2. 从**指标**的 `relevant_columns` 补全缺失字段
    3. 从**取值**补全字段，并把具体取值追加到该字段的 `examples`（去重）
    4. **强制补齐每张表的主键/外键**（join 必需）
    5. 按表聚合为 `TableInfoState`
    6. 指标直转 `MetricInfoState`
    """
    columns_map: dict[str, Any] = {c["id"]: c for c in retrieved_columns}

    # 2. 指标关联字段补全
    for metric in retrieved_metrics:
        for col_id in metric.get("relevant_columns") or []:
            await _get_or_fetch_column(col_id, columns_map, meta_mysql_repository)

    # 3. 取值补全 + 示例值回填
    for value_item in retrieved_values:
        col_id = value_item.get("column_id")
        if not col_id:
            continue
        col_obj = await _get_or_fetch_column(col_id, columns_map, meta_mysql_repository)
        v = value_item.get("value")
        if v is not None and v not in (col_obj.get("examples") or []):
            col_obj.setdefault("examples", []).append(v)

    # 4. 主外键兜底（join 必需）
    table_ids = {c["table_id"] for c in columns_map.values()}
    for table_id in table_ids:
        try:
            key_columns = await meta_mysql_repository.get_column_key_by_id(table_id)
        except Exception as e:
            logger.warning("主外键补全失败，跳过 {}：{}", table_id, e)
            continue
        for key_col in key_columns or []:
            await _get_or_fetch_column(key_col.id, columns_map, meta_mysql_repository)

    # 5. 按表聚合
    table_to_columns: dict[str, list] = defaultdict(list)
    for col in columns_map.values():
        table_to_columns[col["table_id"]].append(col)

    table_infos: list[TableInfoState] = []
    for table_id, columns in table_to_columns.items():
        table_info = await meta_mysql_repository.get_table_info_by_id(table_id)
        table_infos.append(TableInfoState(
            name=table_info.name,
            role=table_info.role,
            description=table_info.description,
            columns=[
                ColumnInfoState(
                    name=c["name"], type=c["type"], role=c["role"],
                    examples=c.get("examples") or [], description=c.get("description") or "",
                    alias=c.get("alias") or [],
                )
                for c in columns
            ],
        ))

    # 6. 指标直转
    metric_infos: list[MetricInfoState] = [
        MetricInfoState(
            name=m.get("name", ""), description=m.get("description", ""),
            relevant_columns=m.get("relevant_columns") or [], alias=m.get("alias") or [],
        )
        for m in retrieved_metrics
    ]
    return table_infos, metric_infos


# ============================================================ ④ 过滤（LLM）

async def filter_tables(query: str, table_infos: list[TableInfoState]) -> list[TableInfoState]:
    """LLM 过滤无关表/列（保留原 `filter_table` 行为）。"""
    import yaml

    kept: dict[str, list[str]] = await invoke_llm_chain(
        "filter_table_info", query=query,
        table_infos=yaml.dump(table_infos, allow_unicode=True, sort_keys=False),
    )
    return [
        {**t, "columns": [c for c in t["columns"] if c["name"] in kept[t["name"]]]}
        for t in table_infos if t["name"] in kept
    ]


async def filter_metrics(query: str, metric_infos: list[MetricInfoState]) -> list[MetricInfoState]:
    """LLM 过滤无关指标（保留原 `filter_metric` 行为）。"""
    import yaml

    kept_names: list[str] = await invoke_llm_chain(
        "filter_metric_info", query=query,
        metric_infos=yaml.dump(metric_infos, allow_unicode=True, sort_keys=False),
    )
    return [m for m in metric_infos if m["name"] in kept_names]


# ============================================================ ⑤ 额外上下文

def get_date_info(now: datetime | None = None) -> DateInfoState:
    """当前日期 / 星期 / 季度（供相对时间换算，如「今年」「本季度」）。"""
    today = now or datetime.today()
    return DateInfoState(
        date=today.strftime("%Y-%m-%d"),
        weekday=today.strftime("%A"),
        quarter=f"Q{(today.month - 1) // 3 + 1}",
    )


async def load_extra_context(dw_mysql_repository) -> tuple[DateInfoState, DBInfoState]:
    """加载日期与数据库方言信息。"""
    return get_date_info(), await dw_mysql_repository.get_db_info()


# ============================================================ ⑥ L1 判定（探针）

def judge_verdict(
    vector_scores: Iterable[float],
    metrics: Sequence[dict],
    values: Sequence[dict],
    metric_scores: Sequence[float] | None = None,
    threshold: float = HIT_THRESHOLD,
) -> str:
    """L1 探针判定 —— 给模型的**启发式提示**（最终由模型决定下一步）。

    返回值：
    - `"low"`        —— 无任何命中，或向量路最高分 < 阈值
                        ⇒ 疑似**拼写错 / 表述偏**，模型可考虑纠错或扩展关键词
    - `"ambiguous"`  —— ★ 高分段出现**多个不同指标** ⇒ 疑似**多实体 / 多度量歧义**
    - `"unique"`     —— 命中明确

    ⚠️ **三点口径说明**（第一版实现曾在此处出错，据真实数据修正）：

    1. **阈值只作用于向量路**（Qdrant COSINE 0~1）；ES 的 BM25 分数无界，**不参与**比较。
    2. ★ **取值路不参与歧义判定** —— ES 单关键词最多返回 `limit=5` 条，
       「召回多个值」是**常态而非歧义**（实测：`华北`/`华东` 同属地区名，都会被召回）。
       早期版本用「≥2 个取值 ⇒ ambiguous」，在真实数据上**误报**，已移除。
    3. **字段列不参与歧义判定** —— 一次查询需要多列是正常的，多列 ≠ 歧义。
       歧义只对**指标**（选哪个度量）有意义，因为指标是**互斥**的聚合口径。

    ⇒ 真正的歧义判断依赖模型看到候选清单后自行推断；
    本函数的 `ambiguous` 只是一个**保守提示**，宁可漏报也不误报。
    """
    scores = [float(s) for s in vector_scores]
    best = max(scores) if scores else 0.0

    # 是否「完全没命中」
    if not scores and not metrics and not values:
        return "low"
    if best < threshold and not values:
        return "low"

    # 歧义：高相似度**指标** ≥2 个不同名 ⇒ 选哪个度量不确定
    # （指标互斥；字段列/取值不参与，理由见 docstring）
    if metric_scores:
        hi_metrics = {
            str(m.get("name"))
            for m, s in zip(metrics, metric_scores)
            if float(s) >= threshold and m.get("name")
        }
        if len(hi_metrics) >= 2:
            return "ambiguous"

    return "unique"


# ============================================================ ⑦ 候选渲染

def render_candidates(
    table_infos: Sequence[dict],
    metric_infos: Sequence[dict],
    date_info: dict | None,
    verdict: str = "",
    counts: dict | None = None,
) -> str:
    """候选信息 → 给模型的文本（表/列/示例值/别名/指标/当前时间）。

    与旧版把整张 yaml 甩给模型的区别：这里只给「**本次召回过、过滤后**」的候选，
    且显式标注 role / 示例值，让模型的「选」有据可依。

    开头带上 **verdict 与召回统计** —— 这既是模型的决策依据（分支由模型判），
    也是「不断召回调整关键词」时**下一轮能看到上一轮命中情况**的信息来源。
    """
    lines: list[str] = []
    c = counts or {}
    head = f"召回完成（verdict={verdict or 'n/a'}）"
    if c:
        head += f"：字段 {c.get('n_columns', 0)} / 指标 {c.get('n_metrics', 0)} / 取值 {c.get('n_values', 0)}"
    lines.append(head)

    lines.append("")
    lines.append("【候选表与字段】（只能从这里选表/列）")
    if not table_infos:
        lines.append("  (无)")
    for t in table_infos:
        lines.append(f"表 {t.get('name')}（{t.get('role')}）：{t.get('description') or ''}")
        for col in t.get("columns") or []:
            examples = col.get("examples") or []
            ex = "、".join(str(e) for e in examples[:8])
            alias = "、".join(str(a) for a in (col.get("alias") or []))
            seg = (f"    {t.get('name')}.{col.get('name')} "
                   f"({col.get('role')}, {col.get('type')}): {col.get('description') or ''}")
            if alias:
                seg += f"  同义={alias}"
            if ex:
                seg += f"  示例值=[{ex}]"
            lines.append(seg)

    lines.append("")
    lines.append("【候选指标】（选概念名 + 聚合方式，真实列由程序展开）")
    if not metric_infos:
        lines.append("  (无)")
    for m in metric_infos:
        rel = "、".join(str(x) for x in (m.get("relevant_columns") or []))
        alias = "、".join(str(a) for a in (m.get("alias") or []))
        seg = f"    指标 {m.get('name')}: {m.get('description') or ''}  相关列=[{rel}]"
        if alias:
            seg += f"  同义={alias}"
        lines.append(seg)

    lines.append("")
    d = date_info or {}
    lines.append(
        f"【当前时间】{d.get('date', '')}（{d.get('weekday', '')}，{d.get('quarter', '')}）"
        "—— 相对时间（今年/本季度/本月）请对照此换算"
    )
    return "\n".join(lines)


# ============================================================ ⑧ 高层入口

async def collect_candidates(
    *,
    query: str,
    columns: Sequence[str],
    metrics: Sequence[str],
    values: Sequence[str],
    ctx: dict,
    emit: Emit = None,
    do_filter: bool = True,
    threshold: float = HIT_THRESHOLD,
) -> dict:
    """`recall_candidates` 工具的主链路：三路召回 → 合并 → 过滤 → 候选就绪。

    Args:
        query:   原始提问（供 LLM 过滤使用；本链路**不做**关键词扩展）
        columns: ★ **模型填**的字段召回词表
        metrics: ★ **模型填**的指标召回词表
        values:  ★ **模型填**的取值召回词表
        ctx:     数据层依赖（`DataAgentContext` 或等价 dict）
        emit:    stage 事件回调（通常传 `runtime.stream_writer`）
        do_filter: 是否做 LLM 过滤（默认 True，保留原行为）

    Returns:
        `{table_infos, metric_infos, date_info, db_info, verdict, n_*}`

    Raises:
        RuntimeError: 三路全部无命中且合并后无表（无法继续）
        Exception:    合并/过滤过程中的异常（上抛，由调用方归因）
    """
    emit = emit or _noop_emit

    emit({"stage": "召回字段"})
    col_payloads, col_scores = await recall_columns(
        list(columns), embedding_client=ctx["embedding_client"],
        repository=ctx["column_qdrant_repository"],
    )

    emit({"stage": "召回指标"})
    metric_payloads, metric_scores = await recall_metrics(
        list(metrics), embedding_client=ctx["embedding_client"],
        repository=ctx["metric_qdrant_repository"],
    )

    emit({"stage": "召回取值"})
    value_payloads, _ = await recall_values(
        list(values), repository=ctx["value_es_repository"],
    )

    logger.info(
        f"三路召回：字段 {len(col_payloads)} / 指标 {len(metric_payloads)} / 取值 {len(value_payloads)}"
    )

    emit({"stage": "合并召回信息"})
    table_infos, metric_infos = await merge_retrieved(
        retrieved_columns=col_payloads, retrieved_metrics=metric_payloads,
        retrieved_values=value_payloads, meta_mysql_repository=ctx["meta_mysql_repository"],
    )

    if do_filter and table_infos:
        # ★ fail-safe：过滤是「锦上添花」（减少无关表和列的 token），
        # 一旦它失败（如 LLM 限流 429）**绝不能拖垮整条召回** ——
        # 退化为「未过滤候选」继续，主链路仍可完成取数。
        emit({"stage": "过滤表信息"})
        try:
            table_infos = await filter_tables(query, table_infos)
        except Exception as e:
            logger.warning("候选过滤失败，回退为未过滤候选：{}", e)

        if metric_infos:
            emit({"stage": "过滤指标"})
            try:
                metric_infos = await filter_metrics(query, metric_infos)
            except Exception as e:
                logger.warning("指标过滤失败，回退为未过滤指标：{}", e)

    emit({"stage": "添加额外上下文"})
    date_info, db_info = await load_extra_context(ctx["dw_mysql_repository"])

    verdict = judge_verdict(
        vector_scores=list(col_scores) + list(metric_scores),
        metrics=metric_payloads, metric_scores=metric_scores,
        values=value_payloads, threshold=threshold,
    )

    return {
        "table_infos": table_infos,
        "metric_infos": metric_infos,
        "date_info": date_info,
        "db_info": db_info,
        "verdict": verdict,
        "n_columns": len(col_payloads),
        "n_metrics": len(metric_payloads),
        "n_values": len(value_payloads),
    }
