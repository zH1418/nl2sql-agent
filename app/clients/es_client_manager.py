import asyncio
from typing import Optional

from elasticsearch import AsyncElasticsearch
from app.conf.app_config import ESConfig, app_config


class ESClientManager:
    def __init__(self, es_config: ESConfig):
        self.es_config = es_config
        self.es_client: Optional[AsyncElasticsearch] = None

    def _get_url(self):
        return (f"http://{self.es_config.host}"
                f":{self.es_config.port}")

    def init(self):
        self.es_client = AsyncElasticsearch([self._get_url()])

    async def close(self):
        await self.es_client.close()


es_client_manager = ESClientManager(app_config.es)

if __name__ == "__main__":
    es_client_manager.init()

    async def test():
        client = es_client_manager.es_client


        # 创建索引
        if not await client.indices.exists(index="my-test-books"):
            await client.indices.create(
                index="my-test-books",
                mappings={
                    "dynamic": False,
                    "properties": {
                        "name": {
                            "type": "text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart"
                        },
                        "author": {"type": "text"},
                        "release_date": {
                            "type": "date",
                            "format": "yyyy-MM-dd"
                        },
                        "page_count": {"type": "long"}
                    }
                }
            )

        # 添加文档批量
        await client.bulk(
            operations=[
                {
                    "index": {
                        "_index": "my-test-books"
                    }
                },
                {
                    "name": "Revelation Space",
                    "author": "Alastair Reynolds",
                    "release_date": "2000-03-15",
                    "page_count": 585
                },
                {
                    "index": {
                        "_index": "my-test-books"
                    }
                },
                {
                    "name": "1984",
                    "author": "George Orwell",
                    "release_date": "1985-06-01",
                    "page_count": 328
                },
                {
                    "index": {
                        "_index": "my-test-books"
                    }
                },
                {
                    "name": "Fahrenheit 451",
                    "author": "Ray Bradbury",
                    "release_date": "1953-10-15",
                    "page_count": 227
                },
                {
                    "index": {
                        "_index": "my-test-books"
                    }
                },
                {
                    "name": "Brave New World",
                    "author": "Aldous Huxley",
                    "release_date": "1932-06-01",
                    "page_count": 268
                },
                {
                    "index": {
                        "_index": "my-test-books"
                    }
                },
                {
                    "name": "The Handmaids Tale",
                    "author": "Margaret Atwood",
                    "release_date": "1985-06-01",
                    "page_count": 311
                }
            ],
        )

        await client.indices.refresh(index="my-test-books")  # 强制刷新，使数据立即可搜索

        # 检索文档
        resp = await client.search(
            index="my-test-books",
            query={
                "match": {
                    "name": "Revelation Space"
                }
            }
        )

        if resp['hits']['hits']:
            for hit in resp['hits']['hits']:
                print(hit['_source'])

        await client.close()

    asyncio.run(test())
