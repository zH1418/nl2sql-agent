from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.types import Command

from app.agent.agent.state import CANDIDATE_FILTER, DataAgentState
from app.agent.recall import collect_candidates, render_candidates
from app.core.log import logger


@tool
async def recall_candidates(
    columns: Annotated[list[str], "该问题必需的数据字段（业务语义名）。只输出字段概念，禁止表名/取值/指标/SQL。"],
    metrics: Annotated[list[str], "该问题的度量概念（如 销售额、GMV、转化率）。禁止字段名/表名/数值/实体取值。"],
    values: Annotated[list[str], "问题里出现的真实取值候选（如 华北、实习生、最近三个月）。禁止字段名/表名/SQL。"],
    runtime: ToolRuntime[Any, DataAgentState],
) -> Command:
    ctx = runtime.context
    emit = runtime.stream_writer

    try:
        result = await collect_candidates(
            query=runtime.state.get("query") or _last_user_text(runtime.state),
            columns=columns, metrics=metrics, values=values,
            ctx=ctx, emit=emit,
            do_filter=CANDIDATE_FILTER,
        )
    except Exception as e:
        logger.opt(exception=True).error(
            "recall_candidates 失败：{}: {}", type(e).__name__, e)
        return Command(update={
            "reason_class": "low_recall",
            "messages": [ToolMessage(
                content=f"召回失败（环境或数据层异常）：{type(e).__name__}: {e}",
                tool_call_id=runtime.tool_call_id,
            )],
        })

    text = render_candidates(
        result["table_infos"], result["metric_infos"], result["date_info"],
        verdict=result["verdict"],
        counts={k: result[k] for k in ("n_columns", "n_metrics", "n_values")},
    )

    hit_note = ""
    if result["verdict"] == "low":
        hit_note = "\n\n⚠️ verdict=low：命中偏弱，请判断是**拼写错**（→纠正）还是**表述偏**（→换同义/上下位词），必要时调整关键词后再次调用本工具。"
    elif result["verdict"] == "ambiguous":
        hit_note = "\n\n⚠️ verdict=ambiguous：存在**多个同样合理的候选**，通常应调用 `ask_user` 请用户确认。"

    return Command(update={
        "table_infos": result["table_infos"],
        "metric_infos": result["metric_infos"],
        "date_info": result["date_info"],
        "db_info": result["db_info"],
        "probe_verdict": result["verdict"],
        "reason_class": "low_recall" if result["verdict"] == "low" else "",
        "needs_next": None,
        "messages": [ToolMessage(
            content=text + hit_note,
            tool_call_id=runtime.tool_call_id,
        )],
    })


def _last_user_text(state: dict) -> str:
    """从 messages 里取最近一条用户消息文本（`query` 未显式写入 state 时的兜底）。"""
    for m in reversed(state.get("messages") or []):
        if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage":
            return getattr(m, "content", "") or ""
    return ""
