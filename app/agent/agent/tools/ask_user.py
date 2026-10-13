"""工具 3：`ask_user` —— 澄清 / 消歧（停下来等用户）。

★ 形态：**模型自制表单** ——
`questions` 由模型生成（问题文本 + 控件类型 + 选项），**不是**固定开放输入框。
前端已支持 `radio` / `checkbox` / `text` 三类控件 ⇒ **前端零改动**（仅新增开放性输入框）。

★ 触发后如何**终止循环**（已实测）：
本工具写 `pending_ask=True` ⇒ `ClarifyMiddleware.before_model` 检测到后
返回 `{"jump_to": "end"}` ⇒ ReAct 循环立即结束（模型**不会**被再次调用）。
⇒ **不需要 checkpointer、不需要 `interrupt()`**（与「最小改动」一致）。

⚠️ **字段名必须是 `text`**：实测前端绑定的是 `q.text`（`App.vue:81`）。
本工具对模型可能给出的 `question` 做**归一化兜底**，避免弹窗标题空白。
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.types import Command

from app.agent.agent.state import DataAgentState
from app.core.log import logger

# 回给模型的哨兵文本（便于 messages 里一眼看出「已停下来等用户」）
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
    """向用户提问以**澄清**或**消歧**。**仅在确实无法自行判定时**使用。

    两种典型场景（同一工具覆盖）：

    - **消歧**（召回出现多个同样合理的候选）：
      例：指标同时命中「成交总额(GMV)」与「平均订单金额(AOV)」⇒
      `{"id":"q1","text":"你说的\\"销售额\\"指哪个？","type":"radio","options":["成交总额(GMV)","平均订单金额(AOV)"]}`
      → 用 `radio`（单选）或 `checkbox`（可多选）。
    - **澄清**（提问缺关键信息）：
      例：`{"id":"q1","text":"统计口径是含税还是不含税？","type":"radio","options":["含税","不含税"]}`
      `{"id":"q2","text":"时间范围是？","type":"text"}`（无候选选项时用 text）

    规则：
    - `type` 只能是 `radio` / `checkbox` / `text`；
    - `radio` / `checkbox` **必须**给 `options`，且选项要**业务可读**（用户能一眼选择）；
    - `text` 不需要 `options`（前端会渲染为自由输入框）；
    - 调用后**流程会立即停止**等待用户回答，本轮结束。不要在同一轮里既提问又继续取数。
    - 前端会在选项下方**附带开放性输入框**（用户可自由说明），故你**不必**为「选项不全」担心。

    Args:
        runtime: 由框架注入（`ToolRuntime`），**无需**在参数里提供。
    """
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

    # 复用现状前端契约（5 类 SSE 事件之一）
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
    """归一化模型给的 questions（字段名与控件类型兜底）。

    实测前端绑定 `q.text` / `q.id` / `q.type` / `q.options`，
    而模型可能写成 `question` 或漏 `id` ⇒ 统一纠正，避免弹窗空白或渲染异常。

    ★ `id` 的两条规则：
    1. 模型**给了**就保留（它可能有意用有意义的 id）；
    2. 模型**没给**则按**输出序号**补 `q1` / `q2`…（**不是原始下标** —— 否则
       前面有被丢弃的无效项时会留下空洞，如只剩 `q3`），且与显式 id **去重**。
    """
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
            # 有选项 ⇒ 默认单选；无选项 ⇒ 自由文本
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

    # ---- 补齐缺失的 id（输出序号 + 与显式 id 去重）----
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
