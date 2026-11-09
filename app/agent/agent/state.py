from __future__ import annotations

from typing import Any, NotRequired

from langchain.agents import AgentState


class DataAgentState(AgentState):
    """继承 `AgentState`（提供 `messages` / `jump_to` / `structured_response`）。"""

    query: NotRequired[str]

    table_infos: NotRequired[list[dict]]
    metric_infos: NotRequired[list[dict]]
    date_info: NotRequired[dict]
    db_info: NotRequired[dict]
    probe_verdict: NotRequired[str]        # L1 判定（unique / ambiguous / low）

    sql_budget: NotRequired[int]           # ★ 剩余预算（初始 5）
    pending_ask: NotRequired[bool]         # ★ ask_user 已触发 ⇒ before_model 据此 jump_to="end"

    needs_next: NotRequired[str | None]    # 分层标记（"L1" | None）
    reason_class: NotRequired[str]         # 最近一次失败归因
    sql_attempts: NotRequired[int]         # 已尝试次数（审计用；控制权在 sql_budget）


INITIAL_BUDGET: int = 5

MAX_ROWS: int = 1000

CANDIDATE_FILTER: bool = False


def initial_state_fields() -> dict[str, Any]:
    """供 service 构造初始 state（避免各处硬编码初值）。"""
    return {"sql_budget": INITIAL_BUDGET, "sql_attempts": 0}
