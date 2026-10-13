"""查询服务：驱动 `create_agent` 探针闭环并以 SSE 推送前端。

相对 v1（13 节点固定流水线）的核心变化：

| 维度 | v1（旧） | v2（本文件） |
|---|---|---|
| 入口 | `compiled_graph.astream(stream_mode="custom")` | **`compiled_agent.astream_events(version="v3")`** |
| 图 | 13 节点固定流水线 | 1 个 `create_agent` 节点（ReAct 循环） |
| 事件 | 单 `custom` 流 | **`.custom`（转发 SSE）+ `.values`（取最终 state）并发消费** |
| 结论 | 无（只出表格） | ★ 推最终 AI 消息（**归一化后**）作为 `{"type":"text"}` |

★ **前端契约不变**：工具内 `stream_writer` 推的仍是 `{"stage":…}` / `{"result":…}` /
`{"type":"clarification",…}` ⇒ 原样转发即可（`type:"text"` 是**新增**事件，旧前端忽略亦无害）。

⚠️ 三个实测坑：
1. **`custom` 投影默认不注册** ⇒ 必须传 `transformers=[lambda scope: CustomTransformer(scope)]`
   （**必须传工厂，不能传实例**）；
2. **v3 投影共享同一 pump** ⇒ 必须**并发消费**，顺序消费会让后一个投影拿到 0 事件；
3. **`content` 可能是内容块列表**（`ark-code-latest`）⇒ 推前端前必须 `content_to_text()` 归一化。
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from app.agent.context import DataAgentContext
from app.core.log import logger
from app.repository.es.value_es_repository import ValueEsRepository
from app.repository.mysql.dw_mysql_repository import DWMysqlRepository
from app.repository.mysql.meta_mysql_repository import MetaMysqlRepository
from app.repository.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repository.qdrant.metric_qdrant_repository import MetricQdrantRepository


class ConversationStore:
    """内存会话管理器（后续可替换为 Redis）。"""

    def __init__(self, ttl_minutes: int = 30):
        self._store: dict[str, dict] = {}
        self._ttl = timedelta(minutes=ttl_minutes)

    def create(self, original_query: str) -> str:
        """创建新会话，返回 conversation_id。"""
        cid = str(uuid.uuid4())
        self._store[cid] = {
            "original_query": original_query,
            "round": 0,
            "created_at": datetime.now(),
        }
        self._cleanup_expired()
        return cid

    def get_and_bump(self, conversation_id: str) -> Optional[dict]:
        """读取会话状态并递增轮次。不存在或已过期返回 None。"""
        self._cleanup_expired()
        session = self._store.get(conversation_id)
        if session is None:
            return None
        if datetime.now() - session["created_at"] > self._ttl:
            del self._store[conversation_id]
            return None
        session["round"] += 1
        return session

    def remove(self, conversation_id: str):
        """完成处理后清理会话。"""
        self._store.pop(conversation_id, None)

    def _cleanup_expired(self):
        """清理过期会话。"""
        now = datetime.now()
        expired = [cid for cid, s in self._store.items() if now - s["created_at"] > self._ttl]
        for cid in expired:
            del self._store[cid]


def _assemble_clarified_query(original_query: str, answers: dict) -> str:
    """将用户补充的信息拼装到原始问题中。

    ★ 跳过空值（避免 `q1=。q1__other=` 这类污染），且**不再产生 `补充信息：。` 的残留标点**
    （v1 用 `"。".join([query, "补充信息："])` 会多出一个句号）。
    """
    items: list[str] = []
    for key, value in answers.items():
        if value is None:
            continue
        if isinstance(value, list):
            value = [str(v) for v in value if str(v).strip()]
            if not value:
                continue
            items.append(f"{key}={', '.join(value)}")
        else:
            text = str(value).strip()
            if not text:
                continue
            items.append(f"{key}={text}")

    base = original_query.strip()
    if not items:
        return base
    return f"{base}。补充信息：{'，'.join(items)}"


def _sse(data: dict) -> str:
    """打包为 SSE 行（与 v1 完全一致，前端契约不变）。"""
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _extract_final_text(final_state: dict) -> str:
    """取最后一条有内容的 AI 消息（★ 归一化内容块列表）。"""
    from app.agent.utils.llm_utils import content_to_text

    for m in reversed(final_state.get("messages") or []):
        if getattr(m, "type", "") != "ai":
            continue
        text = content_to_text(getattr(m, "content", "")).strip()
        if text:
            return text
    return ""


class QueryService:
    def __init__(self,
                 embedding_client: HuggingFaceEndpointEmbeddings,
                 column_qdrant_repository: ColumnQdrantRepository,
                 value_es_repository: ValueEsRepository,
                 metric_qdrant_repository: MetricQdrantRepository,
                 meta_mysql_repository: MetaMysqlRepository,
                 dw_mysql_repository: DWMysqlRepository):

        self.embedding_client = embedding_client
        self.column_qdrant_repository = column_qdrant_repository
        self.value_es_repository = value_es_repository
        self.metric_qdrant_repository = metric_qdrant_repository
        self.meta_mysql_repository = meta_mysql_repository
        self.dw_mysql_repository = dw_mysql_repository
        self.conversations = ConversationStore()

    async def query(self, query: str, conversation_id: str = None,
                    answers: dict = None) -> AsyncIterator[str]:
        """
        用户查询的业务处理（SSE 流）。

        :param query: 用户自然语言提问
        :param conversation_id: 第二轮澄清时携带的会话 ID（可选）
        :param answers: 用户填写的澄清表单答案（可选）
        """
        effective_query = query

        # 第二轮：携带 conversation_id 和 answers ⇒ 拼装成完整问题
        if conversation_id and answers:
            session = self.conversations.get_and_bump(conversation_id)
            if session:
                effective_query = _assemble_clarified_query(session["original_query"], answers)
                logger.info("第二轮澄清，拼装 query: {}", effective_query[:120])
            else:
                logger.warning("会话 {} 不存在或已过期，作为新会话处理", conversation_id[:8])

        context = DataAgentContext(
            embedding_client=self.embedding_client,
            column_qdrant_repository=self.column_qdrant_repository,
            value_es_repository=self.value_es_repository,
            metric_qdrant_repository=self.metric_qdrant_repository,
            meta_mysql_repository=self.meta_mysql_repository,
            dw_mysql_repository=self.dw_mysql_repository
        )

        try:
            async for chunk in self._run_agent(effective_query, context, query, conversation_id):
                yield chunk
        except Exception as e:
            # ⚠️ 不要把异常内联进 f-string：loguru 会对模板调 .format()，
            #    异常文本含 `{'x':...}` 时抛 KeyError，反而掩盖真错
            logger.opt(exception=True).error("查询失败：{}: {}", type(e).__name__, e)
            yield _sse({"error": str(e)})

    async def _run_agent(self, effective_query: str, context: DataAgentContext,
                         original_query: str, conversation_id: str | None) -> AsyncIterator[str]:
        """驱动 v2 新链路（`create_agent`），把事件转成 SSE。"""
        from langgraph.stream.transformers import CustomTransformer

        from app.agent.agent import compiled_agent
        from app.agent.agent.state import initial_state_fields

        # ★ 初始 state：messages（ReAct 消息流）+ query（工具直读）+ 控制字段
        state = {
            "messages": [("user", effective_query)],
            "query": effective_query,
            **initial_state_fields(),
        }

        run = await compiled_agent.astream_events(
            state,
            version="v3",
            context=context,
            # ★ 坑 1：custom 投影默认不注册；必须传**工厂**（不能传实例）
            transformers=[lambda scope: CustomTransformer(scope)],
        )

        final_state: dict = {}

        async def _drain_values():
            """并发消费 `.values` —— 用于取最终 state（末条 AI 消息）。"""
            async for v in run.values:
                final_state.clear()
                final_state.update(v or {})

        # ★ 坑 2：投影共享同一 pump ⇒ 必须并发消费（`.custom` 在主循环、`.values` 在 task）
        values_task = asyncio.create_task(_drain_values())
        interrupted = False
        try:
            async for ev in run.custom:
                if not isinstance(ev, dict):
                    continue

                # 澄清 / 消歧：注入 conversation_id 后结束本轮，等用户下一轮输入
                if ev.get("type") == "clarification":
                    ev["conversation_id"] = self.conversations.create(original_query)
                    yield _sse(ev)
                    interrupted = True
                    return

                # stage / result / error ⇒ 原样转发（前端契约不变）
                yield _sse(ev)

            # custom 消费完 ⇒ 等 values 收尾
            await values_task
        finally:
            if not values_task.done():
                values_task.cancel()

        # ★ 预算锁：按最终 state 记一次成败（失败率记账 + 阈值告警，见 budget_lock.py）
        #    澄清中断路径在上方 `return`，不计入（那不是失败）
        from app.agent.agent.budget_lock import record_outcome

        record_outcome(final_state)

        # ★ 最终结论：归一化后作为 `type:"text"` 事件推前端
        text = _extract_final_text(final_state)
        if text:
            yield _sse({"type": "text", "content": text})

        # 正常结束 ⇒ 清理会话（澄清中断时保留，等下一轮）
        if conversation_id and not interrupted:
            self.conversations.remove(conversation_id)
