"""工具 1：`recall_candidates` —— L1 探针（三路召回 → 合并 → 过滤 → 候选装配）。

★ 本工具**不做关键词扩展**（不调 LLM 扩词）：
「纠错 / 扩写 / 改写」由**模型在填 `columns`/`metrics`/`values` 参数时**完成 ——
省下一轮 LLM 往返，且「根据上一轮召回结果调整关键词」由 `messages` 天然承载。

工具描述（docstring）里的约束**承接自原 3 套 `extend_keywords_for_*_recall.prompt`**
（约束迁往 tool description）。
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.types import Command

from app.agent.agent.state import CANDIDATE_FILTER, DataAgentState
from app.agent.recall import collect_candidates, render_candidates
from app.core.log import logger


@tool
async def recall_candidates(
    columns: Annotated[list[str], "该问题必需的数据字段（业务语义名）。只输出字段概念，禁止表名/取值/指标/SQL。"],
    metrics: Annotated[list[str], "该问题的度量概念（如 销售额、GMV、转化率）。禁止字段名/表名/数值/实体取值。"],
    values: Annotated[list[str], "问题里出现的真实取值候选（如 华北、实习生、最近三个月）。禁止字段名/表名/SQL。"],
    runtime: ToolRuntime[Any, DataAgentState],
) -> Command:
    """按「字段 / 指标 / 取值」三类关键词做三路召回，返回候选表与字段清单。

    三类关键词的语义**不同**，请按语义分别填写（混填会显著降低召回质量）：

    - `columns`：回答该问题**必需存在的数据字段**，用稳定的业务字段概念。
      例：「最近三个月在职实习生的转正情况」→ ["员工身份类型","员工在职状态","转正日期","入职日期"]
      规则：只输出字段名；**禁止**输出字段取值、表名、指标口径、SQL。
      若涉及时间/人群/状态/趋势，必须包含对应字段（时间字段、分类状态字段等）。
    - `metrics`：问题的**度量目标**（指标概念短语，不是字段）。
      例：["转正人数","转正率"] / ["销售额","GMV","成交总额"]
      规则：允许同义词与常见缩写（语义等价为前提），**禁止**字段名、表名、SQL、具体数值、实体取值。
    - `values`：问题里出现的**取值候选**（枚举值 / 业务实体 / 时间语义词）。
      例：["在职","实习生","最近三个月"] / ["华北","华东"]
      规则：只输出字段值层面的内容；**禁止**字段名、表名、指标、SQL；
      明显臆造或过度泛化的取值不要生成。

    ⚠️ 何时需要**再次调用**本工具：
    - 返回的 `verdict=low`（**召回命中太低**）⇒ 说明你的关键词可能**拼写有误**或**表述过于口语**，
      请判断属于哪种：
      · 疑似**错别字**（如「万用标」应为「万用表」）⇒ **纠正写法**
      · 表述**过于口语或过窄**（如「那个烫金的东西」）⇒ **换同义 / 上位 / 下位词**
      · 二者兼有 ⇒ 先纠错再扩写，可再调一次本工具
    - 返回的 `verdict=ambiguous`（**多个同样合理的候选**）⇒ 通常应改用 `ask_user` 向用户**消歧**。

    返回：候选表与字段清单（含示例值与别名）+ 当前时间 + 召回判定（verdict）。

    Args:
        runtime: 由框架注入（`ToolRuntime`），**无需**在参数里提供。
    """
    ctx = runtime.context
    emit = runtime.stream_writer

    try:
        result = await collect_candidates(
            query=runtime.state.get("query") or _last_user_text(runtime.state),
            columns=columns, metrics=metrics, values=values,
            ctx=ctx, emit=emit,
            # ★ 候选过滤关闭：2 次 LLM 调用（~4~8s）换几百 token，当前规模下不划算
            do_filter=CANDIDATE_FILTER,
        )
    except Exception as e:
        logger.opt(exception=True).error(
            "recall_candidates 失败：{}: {}", type(e).__name__, e)
        # 失败也要回消息给模型（否则 ReAct 拿不到反馈）；预算由 BudgetMiddleware 扣
        return Command(update={
            "reason_class": "low_recall",
            "messages": [ToolMessage(
                content=f"召回失败（环境或数据层异常）：{type(e).__name__}: {e}",
                tool_call_id=runtime.tool_call_id,
            )],
        })

    text = render_candidates(
        result["table_infos"], result["metric_infos"], result["date_info"],
        verdict=result["verdict"],
        counts={k: result[k] for k in ("n_columns", "n_metrics", "n_values")},
    )

    hit_note = ""
    if result["verdict"] == "low":
        hit_note = "\n\n⚠️ verdict=low：命中偏弱，请判断是**拼写错**（→纠正）还是**表述偏**（→换同义/上下位词），必要时调整关键词后再次调用本工具。"
    elif result["verdict"] == "ambiguous":
        hit_note = "\n\n⚠️ verdict=ambiguous：存在**多个同样合理的候选**，通常应调用 `ask_user` 请用户确认。"

    return Command(update={
        # ★ 候选工作集写入 state —— build_and_execute 要读它（跨工具流转的唯一载体）
        "table_infos": result["table_infos"],
        "metric_infos": result["metric_infos"],
        "date_info": result["date_info"],
        "db_info": result["db_info"],
        "probe_verdict": result["verdict"],
        "reason_class": "low_recall" if result["verdict"] == "low" else "",
        "needs_next": None,
        "messages": [ToolMessage(
            content=text + hit_note,
            tool_call_id=runtime.tool_call_id,
        )],
    })


def _last_user_text(state: dict) -> str:
    """从 messages 里取最近一条用户消息文本（`query` 未显式写入 state 时的兜底）。"""
    for m in reversed(state.get("messages") or []):
        if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage":
            return getattr(m, "content", "") or ""
    return ""
