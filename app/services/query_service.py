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
        effective_query = query

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
            logger.opt(exception=True).error("查询失败：{}: {}", type(e).__name__, e)
            yield _sse({"error": str(e)})

    async def _run_agent(self, effective_query: str, context: DataAgentContext,
                         original_query: str, conversation_id: str | None) -> AsyncIterator[str]:
        """驱动 v2 新链路（`create_agent`），把事件转成 SSE。"""
        from langgraph.stream.transformers import CustomTransformer

        from app.agent.agent import compiled_agent
        from app.agent.agent.state import initial_state_fields

        state = {
            "messages": [("user", effective_query)],
            "query": effective_query,
            **initial_state_fields(),
        }

        run = await compiled_agent.astream_events(
            state,
            version="v3",
            context=context,
            transformers=[lambda scope: CustomTransformer(scope)],
        )

        final_state: dict = {}

        async def _drain_values():
            """并发消费 `.values` —— 用于取最终 state（末条 AI 消息）。"""
            async for v in run.values:
                final_state.clear()
                final_state.update(v or {})

        values_task = asyncio.create_task(_drain_values())
        interrupted = False
        try:
            async for ev in run.custom:
                if not isinstance(ev, dict):
                    continue

                if ev.get("type") == "clarification":
                    ev["conversation_id"] = self.conversations.create(original_query)
                    yield _sse(ev)
                    interrupted = True
                    return

                yield _sse(ev)

            await values_task
        finally:
            if not values_task.done():
                values_task.cancel()

        from app.agent.agent.budget_lock import record_outcome

        record_outcome(final_state)

        text = _extract_final_text(final_state)
        if text:
            yield _sse({"type": "text", "content": text})

        if conversation_id and not interrupted:
            self.conversations.remove(conversation_id)
