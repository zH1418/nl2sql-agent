"""装配 `create_agent`（对新链路唯一对外入口）。

```python
create_agent(
    model          = llm,                    # 复用现状 ChatOpenAI（火山方舟）
    tools          = [recall_candidates, build_and_execute, ask_user],
    system_prompt  = SYSTEM_PROMPT,
    state_schema   = DataAgentState,         # 候选工作集 + 预算/pending_ask
    context_schema = DataAgentContext,       # ★ 复用现有 TypedDict（实测可行 → dict）
    middleware     = [ControlMiddleware()],  # 动态轮次 + 循环终止
)
```

★ 只保留 LangGraph 的**外壳**（`CompiledStateGraph`）：
service 层继续用 `astream_events` 驱动、前端 5 类 SSE 事件契约不变 ⇒ 改动面最小。

★ 初始 state 由 service 提供（`initial_state_fields()`），包含 `sql_budget` 等控制字段。
"""

from __future__ import annotations

from langchain.agents import create_agent

from app.agent.agent.middleware import ALL_MIDDLEWARE
from app.agent.agent.prompt import SYSTEM_PROMPT
from app.agent.agent.state import DataAgentState
from app.agent.agent.tools import ALL_TOOLS
from app.agent.context import DataAgentContext
from app.agent.llm import llm


def build_agent():
    """构建（未编译缓存的）agent —— 便于测试用不同 model/tools 装配。"""
    return create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        state_schema=DataAgentState,
        context_schema=DataAgentContext,
        middleware=ALL_MIDDLEWARE,
    )


# 模块级编译一次（与 v1 的 `compiled_graph` 对齐，service 直接引用）
compiled_agent = build_agent()
