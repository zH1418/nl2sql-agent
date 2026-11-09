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


compiled_agent = build_agent()
