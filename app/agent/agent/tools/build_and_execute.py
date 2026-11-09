from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.types import Command

from app.agent.agent.sqlflow import (
    DEFAULT_CAP,
    ensure_join_keys,
    ensure_metric_columns,
    execute_probe,
)
from app.agent.agent.state import DataAgentState
from app.agent.ast_builder import ASTBuildError, DIALECT, build_sql
from app.core.log import logger

_FAIL_HINT: dict[str, str] = {
    "select": "选词不合法：只输出合法 JSON；指标名必须在**候选指标**内；聚合方式只能是 SUM/AVG/COUNT/MAX/MIN。",
    "structure": "结构不合法：列必须来自**候选**；where.column 用**真实列名**（不要写 year(dim_date) 这类把表名塞进函数）。",
    "empty": "结果为空：核对 where 的值是否**取自候选示例值**；相对时间（今年/本季度）是否按【当前时间】换算正确。",
    "dberror": "SQL 执行报错：请检查选词是否与候选一致。",
    "backbone": "主干不足：涉及的 join 链推不出来 —— 这属于召回候选不足，重试无用。",
}


@tool
async def build_and_execute(
    choice: Annotated[dict, "选词 JSON（结构见下方说明），所有表名/列名必须来自召回候选。"],
    runtime: ToolRuntime[Any, DataAgentState],
) -> Command:
    ctx = runtime.context
    state = runtime.state
    emit = runtime.stream_writer
    attempts = int(state.get("sql_attempts") or 0) + 1

    table_infos = list(state.get("table_infos") or [])
    metric_infos = list(state.get("metric_infos") or [])
    db_info = state.get("db_info") or {}

    if not table_infos:
        return _fail(runtime, "message_only",
                     "还没有候选信息：请先调用 `recall_candidates` 获取候选表与字段，再调用本工具。",
                     attempts=attempts)

    meta_repo = ctx.get("meta_mysql_repository")
    dw_repo = ctx.get("dw_mysql_repository")

    try:
        table_infos = await ensure_join_keys(table_infos, meta_repo)
    except Exception as e:
        logger.warning("join 键补全异常（继续）：{}", e)

    try:
        table_infos = await ensure_metric_columns(choice, table_infos, metric_infos, meta_repo)
    except Exception as e:
        logger.warning("指标列补全异常（继续）：{}", e)

    emit({"stage": "生成sql语句"})
    dialect = (db_info or {}).get("dialect") or DIALECT
    try:
        sql = build_sql(choice, table_infos, metric_infos, dialect=dialect)
    except ASTBuildError as e:
        cls = e.reason_class
        logger.info("AST 构建失败（{}）：{}", cls, e)
        hint = _FAIL_HINT.get(cls, "")
        if cls == "backbone":
            return _fail(runtime, cls, f"选词导致主干推导失败：{e}", attempts=attempts,
                         needs_next="L1", hint=hint)
        return _fail(runtime, cls, f"SQL 构建失败：{e}", attempts=attempts, hint=hint)
    except Exception as e:
        logger.opt(exception=True).error("AST 构建未预期异常：{}", e)
        return _fail(runtime, "structure", f"SQL 构建异常：{type(e).__name__}: {e}", attempts=attempts)

    emit({"stage": f"取数口径：{_summarize(choice, metric_infos)}"})

    emit({"stage": "执行sql"})
    probe = await execute_probe(sql, dw_repo, cap=DEFAULT_CAP, dialect=dialect)

    if not probe.hit:
        cls = probe.failure_class or "dberror"
        return _fail(runtime, cls, probe.error or "执行失败", attempts=attempts,
                     sql=probe.sql, hint=_FAIL_HINT.get(cls, ""))

    emit({"result": probe.rows})
    summary = {
        "status": "hit",
        "row_count": probe.row_count,
        "truncated": probe.truncated,
        "sql": probe.sql,
    }
    logger.info(f"build_and_execute 命中：{probe.row_count} 行")
    return Command(update={
        "sql_attempts": attempts,
        "reason_class": "",
        "needs_next": None,
        "messages": [ToolMessage(
            content=("查询成功，结果已展示给用户。\n" + json.dumps(summary, ensure_ascii=False)
                     + "\n若结果已满足用户问题，请直接给出简要结论并结束；无需再调用工具。"),
            tool_call_id=runtime.tool_call_id,
        )],
    })


def _summarize(choice: dict, metric_infos: list[dict]) -> str:
    """选词摘要（外显「系统理解成了什么」）—— 指标 / 维度 / 过滤。"""
    parts: list[str] = []
    mets = choice.get("metrics") or []
    if mets:
        names = [m.get("name") if isinstance(m, dict) else str(m) for m in mets]
        parts.append("指标=" + "、".join(str(n) for n in names if n))
    dims = choice.get("group_by") or []
    if dims:
        parts.append("维度=" + "、".join(str(d).split(".")[-1] for d in dims))
    whs = choice.get("where") or []
    if whs:
        conds = [f"{str(w.get('column','')).split('.')[-1]}{w.get('op','=')}{w.get('value','')}"
                 for w in whs if isinstance(w, dict)]
        parts.append("过滤=" + "、".join(conds))
    return " / ".join(parts) or "无聚合条件"


def _fail(
    runtime: ToolRuntime, failure_class: str, error: str, *,
    attempts: int, sql: str = "", hint: str = "", needs_next: str | None = None,
) -> Command:
    """失败统一出口：回状态摘要给模型（**不含**数据），并写归因供审计/预算锁。"""
    payload = {
        "status": "fail",
        "failure_class": failure_class,
        "error": str(error)[:300],
        "sql": sql,
    }
    if hint:
        payload["hint"] = hint
    if needs_next:
        payload["needs_next"] = needs_next
        payload["note"] = "该失败属于候选不足，重试无用：请直接结束并说明未找到相关数据。"

    logger.info("build_and_execute 失败（{}）：{}", failure_class, str(error)[:150])
    return Command(update={
        "sql_attempts": attempts,
        "reason_class": failure_class,
        "needs_next": needs_next,
        "messages": [ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=runtime.tool_call_id,
        )],
    })
