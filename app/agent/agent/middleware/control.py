"""中间件 1：`ControlMiddleware` —— 动态轮次 + 循环终止。

★ 为什么「预算」与「终止」放在**同一个**中间件：
`hook_config(can_jump_to=...)` 的能力由框架按**单个**中间件解析
（`langchain/agents/factory.py` 里取 `middleware_w_before_model[...]` 的一个来读
`__can_jump_to__`）⇒ 若把「预算耗尽终止」与「澄清终止」拆到两个中间件，
可能只有一个生效。**合并成一个**最稳。

## 三层记账

| 动作 | 预算 |
|---|---|
| ① 每轮**思考**（`before_model`） | **−1** |
| ② 工具**正常返回**（`wrap_tool_call`） | **＋1**（返还）⇒ ①② 净 **0** |
| ③ 工具**异常** | **额外 −1**（不返还）⇒ ①③ 净 **−2** |

⇒ 正常路径**净消耗 0**（不出错就不会被预算终止）；
初始 `budget=5` ⇒ 异常路径 **3 次机会**（5 → 3 → 1 → −1 出局）。

## 终止条件（`before_model` 依序判断，命中即 `jump_to="end"`）

1. `pending_ask` —— `ask_user` 已触发，**必须停下来等用户**（不扣预算）
2. `needs_next` —— 上一步判定了「需扩圈」（如 `backbone`），重试无用
3. `sql_budget <= 0` —— 预算耗尽（快速失败）

★ 判成败的单一信号 = `reason_class` **是否非空**（工具们已统一约定：
成功写 `""`，失败写归因字符串）。无需解析 ToolMessage 文本。
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app.agent.agent.state import INITIAL_BUDGET
from app.core.log import logger

# 预算耗尽的归因标记（审计/前端话术用）
REASON_BUDGET_EXHAUSTED = "budget_exhausted"

# ★ 终止时按归因生成的**话术**：是没找到相关字段，还是超出系统能力，两种道歉不一样
# 否则循环被中间件终止后，用户拿不到任何解释（模型没机会再说话）。
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

    # ---------------------------------------------------------- 终止判断 + 思考扣费

    def _before(self, state: dict, runtime: Any) -> dict | None:
        budget = state.get("sql_budget")
        if budget is None:
            budget = INITIAL_BUDGET          # 首轮初始化（service 未显式给时兜底）

        # ① 澄清/消歧已触发 ⇒ 停下来等用户（★ 不扣预算：工具成功返还，净 0）
        if state.get("pending_ask"):
            logger.info("ControlMiddleware：pending_ask ⇒ 终止循环，等待用户回答")
            return {"jump_to": "end"}

        # ② 已判定需扩圈（如 backbone）⇒ 重试无用，直接结束
        if state.get("needs_next"):
            reason = state.get("reason_class")
            logger.info("ControlMiddleware：needs_next={} ⇒ 终止循环", state.get("needs_next"))
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=_talk_for(reason, state.get("needs_next")))],
            }

        # ③ 预算耗尽 ⇒ 快速失败
        if budget <= 0:
            logger.warning("ControlMiddleware：预算耗尽（{}）⇒ 快速失败", budget)
            return {
                "jump_to": "end",
                "reason_class": REASON_BUDGET_EXHAUSTED,
                "needs_next": "L1",
                "messages": [AIMessage(content=_talk_for(REASON_BUDGET_EXHAUSTED, None))],
            }

        # ④ 思考消耗 −1
        return {"sql_budget": budget - 1}

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict, runtime: Any) -> dict | None:
        return self._before(state, runtime)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: dict, runtime: Any) -> dict | None:
        return self._before(state, runtime)

    # ---------------------------------------------------------- 工具成败结算

    def _after_tool(self, result: Any) -> bool:
        """返回 True 表示成功（+1 返还），False 表示失败（额外 −1）。"""
        return not _failed(result)

    def _settle(self, request: Any, result: Any) -> Any:
        """按工具成败把预算增减**写成绝对值**（`ToolCallRequest.state` 可读当前值）。

        写成绝对值而不是增量：state 里没有 `_delta` 这类字段，
        写未知键不会生效（且可能报错）⇒ 必须直接给 `sql_budget` 新值。
        """
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
