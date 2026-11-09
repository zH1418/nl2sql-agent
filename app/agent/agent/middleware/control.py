from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app.agent.agent.state import INITIAL_BUDGET
from app.core.log import logger

REASON_BUDGET_EXHAUSTED = "budget_exhausted"

_TERMINAL_TALK: dict[str, str] = {
    "low_recall": "没有在数据字典里找到能回答这个问题的字段或指标，可能是问题超出了当前数据的范围。可以换个说法，或补充更具体的业务词再试。",
    "budget_exhausted": "我尝试了几次仍未取到有效结果，已停下以免继续无效消耗。可以补充条件（比如明确指标口径、时间范围）后重试。",
    "backbone": "这个问题涉及的表关联超出了当前支持范围（暂时只支持单事实表查询）。可以拆成更简单的问法试试。",
    "select": "我没能在候选范围内选出合适的字段/指标，已停止。可以换个更贴近业务的问法。",
    "structure": "生成的查询结构不合法，重试后仍未修正，已停止。可以换个问法。",
    "empty": "查询没有命中任何数据，可能是筛选条件不在数据范围内。可以确认下时间/地区等条件。",
    "dberror": "查询执行出错，已停止。请稍后重试或换一种问法。",
}
_TERMINAL_FALLBACK = "我没能完成这次查询，已停止。可以换个问法或补充条件再试。"


def _talk_for(reason: str | None, needs_next: str | None) -> str:
    if needs_next:
        return _TERMINAL_TALK.get("backbone", _TERMINAL_FALLBACK)
    return _TERMINAL_TALK.get(reason or "", _TERMINAL_FALLBACK)


def _update_of(result: Any) -> dict:
    """取工具返回里的 state 更新（`Command.update` 或 `ToolMessage.content` 解析）。"""
    if isinstance(result, Command):
        return dict(getattr(result, "update", None) or {})
    content = getattr(result, "content", None)
    if isinstance(content, str) and content.strip().startswith("{"):
        import json

        try:
            obj = json.loads(content)
            if isinstance(obj, dict):
                return obj
        except Exception:  # 解析失败不影响主流程
            return {}
    return {}


def _failed(result: Any) -> bool:
    """工具是否失败 —— 以 `reason_class` 是否非空为准（工具们的统一约定）。"""
    upd = _update_of(result)
    return bool(upd.get("reason_class"))


class ControlMiddleware(AgentMiddleware):
    """动态轮次预算 + 循环终止。"""


    def _before(self, state: dict, runtime: Any) -> dict | None:
        budget = state.get("sql_budget")
        if budget is None:
            budget = INITIAL_BUDGET          # 首轮初始化（service 未显式给时兜底）

        if state.get("pending_ask"):
            logger.info("ControlMiddleware：pending_ask ⇒ 终止循环，等待用户回答")
            return {"jump_to": "end"}

        if state.get("needs_next"):
            reason = state.get("reason_class")
            logger.info("ControlMiddleware：needs_next={} ⇒ 终止循环", state.get("needs_next"))
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=_talk_for(reason, state.get("needs_next")))],
            }

        if budget <= 0:
            logger.warning("ControlMiddleware：预算耗尽（{}）⇒ 快速失败", budget)
            return {
                "jump_to": "end",
                "reason_class": REASON_BUDGET_EXHAUSTED,
                "needs_next": "L1",
                "messages": [AIMessage(content=_talk_for(REASON_BUDGET_EXHAUSTED, None))],
            }

        return {"sql_budget": budget - 1}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict, runtime: Any) -> dict | None:
        return self._before(state, runtime)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: dict, runtime: Any) -> dict | None:
        return self._before(state, runtime)


    def _after_tool(self, result: Any) -> bool:
        """返回 True 表示成功（+1 返还），False 表示失败（额外 −1）。"""
        return not _failed(result)

    def _settle(self, request: Any, result: Any) -> Any:
        delta = 1 if self._after_tool(result) else -1
        cur = (getattr(request, "state", None) or {}).get("sql_budget")
        cur = INITIAL_BUDGET if cur is None else int(cur)
        new = cur + delta
        logger.info(
            f"ControlMiddleware：工具结算 delta={delta:+d}（{cur} → {new}）"
            f"{'［失败：惩罚，不返还］' if delta < 0 else '［成功：返还思考消耗］'}"
        )
        upd = dict(getattr(result, "update", None) or {}) if isinstance(result, Command) else {}
        if not upd and isinstance(result, ToolMessage):
            upd = {"messages": [result]}
        return Command(update={**upd, "sql_budget": new})

    def wrap_tool_call(self, request: Any, handler: Any):
        return self._settle(request, handler(request))

    async def awrap_tool_call(self, request: Any, handler: Any):
        return self._settle(request, await handler(request))
