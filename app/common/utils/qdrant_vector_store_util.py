import uuid
from typing import List, Callable, TypeVar, Any
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# 泛型定义：T=源业务实体，P=Qdrant Payload 实体
T = TypeVar("T")
P = TypeVar("P")


class QdrantVectorStoreUtil:
    """
    Qdrant 向量存储通用工具类
    封装「实体拆分多向量点 + 批量向量化 + 批量写入」通用流程，与具体业务实体完全解耦
    所有业务线的向量入库逻辑均可复用该工具
    """

    @staticmethod
    async def batch_upsert(
        embedding_client: HuggingFaceEndpointEmbeddings,
        qdrant_repo: Any,
        entities: List[T],
        text_extractor: Callable[[T], List[str]],
        payload_converter: Callable[[T], P],
        batch_size: int = 10,
    ) -> None:
        """
        批量将业务实体转换为向量点并写入 Qdrant
        :param embedding_client: 向量模型客户端实例
        :param qdrant_repo: Qdrant 仓库实例，需实现 ensure_collection() 和 upsert_embeddings() 方法
        :param entities: 待处理的业务实体列表
        :param text_extractor: 文本提取函数：输入单个实体，返回所有需要向量化的文本列表
        :param payload_converter: Payload转换函数：输入源实体，返回Qdrant存储的Payload对象
        :param batch_size: 向量化批次大小，默认10
        """
        # 1. 前置校验：确保集合存在
        await qdrant_repo.ensure_collection()

        # 2. 构建向量点：单个实体对应多个向量文本，共用同一份 payload
        points = []
        for entity in entities:
            payload = payload_converter(entity)
            embedding_texts = text_extractor(entity)
            for text in embedding_texts:
                points.append({
                    "id": str(uuid.uuid4()),
                    "embedding_text": text,
                    "payload": payload
                })

        # 空数据直接返回，避免无效调用
        if not points:
            return

        # 3. 批量向量化：按批次调用接口，兼容原逻辑的分批策略
        all_texts = [point["embedding_text"] for point in points]
        embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_embeddings = await embedding_client.aembed_documents(batch_texts)
            embeddings.extend(batch_embeddings)

        # 4. 批量写入 Qdrant
        ids = [point["id"] for point in points]
        payloads = [point["payload"] for point in points]
        await qdrant_repo.upsert_embeddings(ids, embeddings, payloads)