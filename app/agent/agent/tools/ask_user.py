from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.types import Command

from app.agent.agent.state import DataAgentState
from app.core.log import logger

ASK_SENTINEL = "AWAITING_USER"

_ALLOWED_TYPES = ("radio", "checkbox", "text")


@tool
async def ask_user(
    questions: Annotated[
        list[dict],
        "待用户确认的问题列表。每项：{id, text, type: radio|checkbox|text, options}",
    ],
    runtime: ToolRuntime[Any, DataAgentState],
) -> Command:
    normalized = _normalize(questions)
    if not normalized:
        return Command(update={
            "reason_class": "select",
            "messages": [ToolMessage(
                content="ask_user 调用无效：questions 为空。请给出至少一个问题，或改为直接取数。",
                tool_call_id=runtime.tool_call_id,
            )],
        })

    message = "请补充以下信息："
    if len(normalized) == 1:
        message = normalized[0]["text"] or message

    runtime.stream_writer({
        "type": "clarification",
        "message": message,
        "questions": normalized,
    })
    logger.info(f"ask_user 已推送澄清事件：{len(normalized)} 个问题")

    return Command(update={
        "pending_ask": True,          # ★ 中间件据此 jump_to="end"
        "reason_class": "",
        "messages": [ToolMessage(
            content=(f"{ASK_SENTINEL}: 已向用户提出 {len(normalized)} 个问题，")
                    + "本轮到此结束，等待用户回答后继续。",
            tool_call_id=runtime.tool_call_id,
        )],
    })


def _normalize(questions: list[dict] | None) -> list[dict]:
    raw: list[dict] = []
    for q in questions or []:
        if not isinstance(q, dict):
            continue
        text = str(q.get("text") or q.get("question") or "").strip()
        if not text:
            continue
        qtype = str(q.get("type") or "").strip().lower()
        options = [str(o) for o in (q.get("options") or []) if str(o).strip()]

        if qtype not in _ALLOWED_TYPES:
            qtype = "radio" if options else "text"
        if qtype in ("radio", "checkbox") and not options:
            qtype = "text"       # 单选/多选必须有选项，否则退化为自由输入
        if qtype == "text":
            options = []

        item: dict[str, Any] = {"text": text, "type": qtype}
        if options:
            item["options"] = options
        explicit = str(q.get("id") or "").strip()
        if explicit:
            item["id"] = explicit
        raw.append(item)

    taken = {it["id"] for it in raw if "id" in it}
    seq = 0
    for it in raw:
        if "id" in it:
            continue
        seq += 1
        while f"q{seq}" in taken:
            seq += 1
        it["id"] = f"q{seq}"
        taken.add(it["id"])
    return raw
