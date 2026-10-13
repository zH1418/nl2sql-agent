"""`create_agent` 的 3 个工具。

工具边界 = **模型的决策点**：
1. `recall_candidates` —— 用**什么词**去召回（纠错 / 扩写 / 改写在此完成）
2. `build_and_execute` —— **选什么**（指标 / 维度 / 过滤条件），并立即执行
3. `ask_user`          —— **要不要问用户**（歧义无法自行判定时）
"""

from .ask_user import ASK_SENTINEL, ask_user
from .build_and_execute import build_and_execute
from .recall_candidates import recall_candidates

__all__ = ["recall_candidates", "build_and_execute", "ask_user", "ASK_SENTINEL", "ALL_TOOLS"]

ALL_TOOLS = [recall_candidates, build_and_execute, ask_user]
