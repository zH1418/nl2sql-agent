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


HIT_THRESHOLD: float = 0.85

ALLOW_POS: tuple[str, ...] = (
    "n", "nr", "ns", "nt", "nz", "v", "vn", "a", "an", "eng", "i", "l",
)

Emit = Callable[[dict], None] | None


def _noop_emit(_: dict) -> None:
    """默认 emit：什么都不做（便于无 UI 场景调用）。"""



def extract_keywords(query: str) -> list[str]:
    keywords: list[str] = jieba.analyse.extract_tags(query, allowPOS=ALLOW_POS)
    return list(set(keywords + [query]))



async def _search_all(
    keywords: Sequence[str],
    search_one: Callable[[str], Awaitable[list[tuple[dict, float]]]],
) -> tuple[list[dict], list[float]]:
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
    return await _search_all(keywords, repository.search_scored)



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
    columns_map: dict[str, Any] = {c["id"]: c for c in retrieved_columns}

    for metric in retrieved_metrics:
        for col_id in metric.get("relevant_columns") or []:
            await _get_or_fetch_column(col_id, columns_map, meta_mysql_repository)

    for value_item in retrieved_values:
        col_id = value_item.get("column_id")
        if not col_id:
            continue
        col_obj = await _get_or_fetch_column(col_id, columns_map, meta_mysql_repository)
        v = value_item.get("value")
        if v is not None and v not in (col_obj.get("examples") or []):
            col_obj.setdefault("examples", []).append(v)

    table_ids = {c["table_id"] for c in columns_map.values()}
    for table_id in table_ids:
        try:
            key_columns = await meta_mysql_repository.get_column_key_by_id(table_id)
        except Exception as e:
            logger.warning("主外键补全失败，跳过 {}：{}", table_id, e)
            continue
        for key_col in key_columns or []:
            await _get_or_fetch_column(key_col.id, columns_map, meta_mysql_repository)

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

    metric_infos: list[MetricInfoState] = [
        MetricInfoState(
            name=m.get("name", ""), description=m.get("description", ""),
            relevant_columns=m.get("relevant_columns") or [], alias=m.get("alias") or [],
        )
        for m in retrieved_metrics
    ]
    return table_infos, metric_infos



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



def judge_verdict(
    vector_scores: Iterable[float],
    metrics: Sequence[dict],
    values: Sequence[dict],
    metric_scores: Sequence[float] | None = None,
    threshold: float = HIT_THRESHOLD,
) -> str:
    scores = [float(s) for s in vector_scores]
    best = max(scores) if scores else 0.0

    if not scores and not metrics and not values:
        return "low"
    if best < threshold and not values:
        return "low"

    if metric_scores:
        hi_metrics = {
            str(m.get("name"))
            for m, s in zip(metrics, metric_scores)
            if float(s) >= threshold and m.get("name")
        }
        if len(hi_metrics) >= 2:
            return "ambiguous"

    return "unique"



def render_candidates(
    table_infos: Sequence[dict],
    metric_infos: Sequence[dict],
    date_info: dict | None,
    verdict: str = "",
    counts: dict | None = None,
) -> str:
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
