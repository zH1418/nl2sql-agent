import asyncio
import random
import uuid
from typing import Optional

from qdrant_client import AsyncQdrantClient, models

from app.conf.app_config import app_config,QdrantConfig


class QdrantClientManager:
    def __init__(self, qdrant_config: QdrantConfig):
        # 初始化Qdrant客户端管理器
        self.qdrant_config = qdrant_config
        self.client: Optional[AsyncQdrantClient] = None

    def _get_url(self):
        return (f"http://{self.qdrant_config.host}"
                f":{self.qdrant_config.port}")

    def init(self):
        self.client = AsyncQdrantClient(self._get_url())

    async def close(self):
        await self.client.close()

qdrant_client_manager = QdrantClientManager(app_config.qdrant)

if __name__ == "__main__":
    qdrant_client_manager.init()
    client = qdrant_client_manager.client

    async def test():
        # 不存在collection_name的集合就创建
        if not await client.collection_exists(collection_name="new_collection"):
            # 集合创建
            await client.create_collection(
                collection_name="new_collection",
                vectors_config=models.VectorParams(
                    size=app_config.qdrant.embedding_size,
                    distance=models.Distance.COSINE
                )
            )

        await client.upsert(
            collection_name="new_collection",

            points=[
                models.PointStruct(
                    id = uuid.uuid4(),
                    payload={
                        "vector": f"vector{i}"
                    },
                    vector=[random.random() for _ in range(app_config.qdrant.embedding_size)],
                )
                for i in range(10)
            ]
        )

        result = await client.query_points(
            collection_name="new_collection",
            query=[random.random() for _ in range(app_config.qdrant.embedding_size)],
            limit=10,
            score_threshold=0.7
        )

        for point in result.points:
            print(point)

        await client.close()

    asyncio.run(test())