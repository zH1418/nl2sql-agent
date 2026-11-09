from typing import Generic, TypeVar, List
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from app.conf.app_config import app_config

PayloadType = TypeVar('PayloadType')


class BaseQdrantRepository(Generic[PayloadType]):
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
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            score_threshold=score_threshold,
            limit=limit
        )
        return [(point.payload, float(point.score)) for point in results.points]