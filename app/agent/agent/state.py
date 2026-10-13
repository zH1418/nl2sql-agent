"""`create_agent` 的 state 定义（ReAct 消息流 + 程序侧控制字段）。

★ 两条设计要点：

1. **候选工作集必须驻留 state** —— `recall_candidates` 产出的
   `table_infos` / `metric_infos` 要**跨工具**传给 `build_and_execute`，
   而两者是**独立的工具调用**，没有别的载体。
   ⇒ 由 `recall_candidates` 返回 `Command(update=...)` 写入，`build_and_execute` 读取。
   （实测确认：工具返回 `Command` 可更新 state，且下一个工具能读到。）

2. **模型侧上下文由 `messages` 天然承载** —— ReAct 会把每轮的
   「模型填的参数 + 工具返回」自动记入 `messages`，
   故**不需要** `last_choice` / `last_sql` / `probe_rows` 之类的搬运字段。

字段命名约定如下。
"""

from __future__ import annotations

from typing import Any, NotRequired

from langchain.agents import AgentState


class DataAgentState(AgentState):
    """继承 `AgentState`（提供 `messages` / `jump_to` / `structured_response`）。"""

    # —— 用户问题（service 写入；工具直接读，避免从 messages 反查）——
    query: NotRequired[str]

    # —— 候选工作集（recall_candidates 写 → build_and_execute 读）——
    table_infos: NotRequired[list[dict]]
    metric_infos: NotRequired[list[dict]]
    date_info: NotRequired[dict]
    db_info: NotRequired[dict]
    probe_verdict: NotRequired[str]        # L1 判定（unique / ambiguous / low）

    # —— 程序侧控制（BudgetMiddleware / 澄清中间件读写）——
    sql_budget: NotRequired[int]           # ★ 剩余预算（初始 5）
    pending_ask: NotRequired[bool]         # ★ ask_user 已触发 ⇒ before_model 据此 jump_to="end"

    # —— 审计 / 末次校验 ——
    needs_next: NotRequired[str | None]    # 分层标记（"L1" | None）
    reason_class: NotRequired[str]         # 最近一次失败归因
    sql_attempts: NotRequired[int]         # 已尝试次数（审计用；控制权在 sql_budget）


# 预算初值（异常路径净 −2 ⇒ 3 次机会）
INITIAL_BUDGET: int = 5

# 结果推前端时，探针的行数上限（实测最大结果 115 行 ⇒ 8.7 倍余量）
MAX_ROWS: int = 1000

# ★ 候选过滤开关（**关闭**）
#   过滤（filter_tables / filter_metrics）要花 **2 次 LLM 调用（约 4~8s）**，
#   而实测它省的只是候选文本的几百字符（候选文本本身仅 1.7k~2.2k 字符）。
#   ⇒ 当前数据规模下收益为负，故关闭。
#   ⇒ 何时改回 True：**候选文本远大于 8k 字符**时（schema 显著变大，如 50 表/500 列）。
#   ⇒ 关闭只是「跳过调用」，两个 prompt 与函数**均保留**（Retire 不 Delete）。
CANDIDATE_FILTER: bool = False


def initial_state_fields() -> dict[str, Any]:
    """供 service 构造初始 state（避免各处硬编码初值）。"""
    return {"sql_budget": INITIAL_BUDGET, "sql_attempts": 0}
