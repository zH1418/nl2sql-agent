"""新链路（v2）：`create_agent` 探针驱动循环。

由 v1（13 节点固定扁平图）演进而来：编排收敛为单 `create_agent` 节点，分支决策从图拓扑移交模型。

模块划分：
- `state.py`    —— `DataAgentState`（ReAct 消息流 + 程序侧控制字段）
- `sqlflow.py`  —— join 键/指标列补全 + L2 执行探针
- `tools/`      —— 3 个工具（recall_candidates / build_and_execute / ask_user）
- `middleware/` —— 动态轮次预算 + 澄清终止循环
- `prompt.py`   —— system prompt（含从 3 套 extend prompt 迁移的约束）
- `builder.py`  —— 装配 `compiled_agent`（对外唯一入口）

★ 对外入口 `compiled_agent` 用惰性导入（PEP 562）：模块级导入 builder
需要模型/中间件齐备，惰性化可让本包在**部分模块尚未就绪**时仍可被引用
（也避免 `app.agent.agent.tools` 的使用者被迫加载整个 agent）。
"""

from __future__ import annotations

from typing import Any

from .state import DataAgentState

__all__ = ["DataAgentState", "compiled_agent", "build_agent"]


def __getattr__(name: str) -> Any:  # PEP 562：惰性代理
    if name in ("compiled_agent", "build_agent"):
        from . import builder

        return getattr(builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
