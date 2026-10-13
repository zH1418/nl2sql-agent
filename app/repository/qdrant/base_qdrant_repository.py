from typing import Generic, TypeVar, List
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from app.conf.app_config import app_config

# 声明一个泛型类型变量，代表不同的 Payload 实体
PayloadType = TypeVar('PayloadType')


class BaseQdrantRepository(Generic[PayloadType]):
    # 声明为类属性，由子类重写
    collection_name: str = None

    def __init__(self, client: AsyncQdrantClient):
        self.client = client

    async def ensure_collection(self):
        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=app_config.qdrant.embedding_size,
                    distance=Distance.COSINE
                )
            )

    async def upsert_embeddings(
            self,
            ids: List[str],
            embeddings: List[List[float]],
            payloads: List[PayloadType],
            batch_size: int = 64
    ):
        zipped = list(zip(ids, embeddings, payloads))

        # 分批处理逻辑
        for i in range(0, len(zipped), batch_size):
            batch = zipped[i:i + batch_size]
            points = [
                PointStruct(id=id, vector=embedding, payload=payload)
                for id, embedding, payload in batch
            ]
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    async def search(
            self,
            embedding: List[float],
            score_threshold: float = 0.6,
            limit: int = 5
    ) -> List[PayloadType]:
        """
        根据向量检索相似记录，返回对应的 Payload 列表。
        """
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            score_threshold=score_threshold,
            limit=limit
        )
        return [point.payload for point in results.points]

    async def search_scored(
            self,
            embedding: List[float],
            score_threshold: float = 0.6,
            limit: int = 5
    ) -> List[tuple[PayloadType, float]]:
        """同 `search`，但**保留相似度分数**。

        为什么需要：L1 探针要用 **0.85 阈值**判定「命中是否够强」，
        而 `search` 只返回 payload、把 score 丢了（`point.score` 未被取出）。
        本方法是**纯新增**，不改变 `search` 的既有行为。
        """
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            score_threshold=score_threshold,
            limit=limit
        )
        return [(point.payload, float(point.score)) for point in results.points]