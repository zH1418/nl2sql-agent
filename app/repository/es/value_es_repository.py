from elasticsearch import AsyncElasticsearch

from app.models.es.value_info_es import ValueInfoEs


class ValueEsRepository:
    es_index_name = "data_agent_values"
    es_index_mappings = {
        "dynamic": False,
        "properties": {
            "id": {"type": "keyword"},
            "value": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_max_word"},
            "type": {"type": "keyword"},
            "column_id": {"type": "keyword"},
            "column_name": {"type": "keyword"},
            "table_id": {"type": "keyword"},
            "table_name": {"type": "keyword"},
        }
    }

    def __init__(self, client: AsyncElasticsearch):
        self.client = client

    async def ensure_index(self):
        """
        确保保存值数据的索引存在
        :return:
        """
        # 判断值存储的索引是否存在
        if not await self.client.indices.exists(index=self.es_index_name):
            # 索引不存在，创建索引
            await self.client.indices.create(
                index=self.es_index_name,
                mappings=self.es_index_mappings
            )

    async def upsert_values(self, value_infos: list[ValueInfoEs], batch_size: int = 20):
        """
        为值数据添加全文索引
        :param value_infos:
        :return:
          {
            "index": {
                "_index": "books"
            }
        },
        {
            "name": "Revelation Space",
            "author": "Alastair Reynolds",
            "release_date": "2000-03-15",
            "page_count": 585
        },
        """
        # 循环遍历批量获取处理
        for i in range(0, len(value_infos), batch_size):
            # 定义集合，组成operations
            operations = []
            # 获取批次value数据
            batch_value_infos = value_infos[i:i + batch_size]
            # 构建operations
            for batch_value_info in batch_value_infos:
                # 追加索引指定
                operations.append({"index": {"_index": self.es_index_name}})
                # 追加文档值
                operations.append(batch_value_info)
            await self.client.bulk(
                operations=operations
            )

    async def search(self, keyword: str, threshold: float = 0.6, limit: int = 5) -> list[ValueInfoEs]:
        """
        根据关键字keyword 在es中全文检索字段值数据
        :param keyword:
        :param threshold:
        :param limit:
        :return: list[ValueInfoES]
        """
        # 1.请求es的查询接口进行match查询
        result = await self.client.search(
            index=self.es_index_name,
            query={
                "match": {
                    "value": keyword
                }
            },
            min_score=threshold,
            size=limit,
        )
        # 2.解析查询结果
        return [hit['_source'] for hit in result["hits"]["hits"]]

    async def search_scored(self, keyword: str, threshold: float = 0.6,
                            limit: int = 5) -> list[tuple[ValueInfoEs, float]]:
        """同 `search`，但**保留 ES 的相关性分数**（`_score`）。

        为什么需要：L1 探针的 0.85 阈值判定需要分数；
        `search` 只取 `_source`，把 `_score` 丢了。
        本方法是**纯新增**，不改变 `search` 的既有行为。
        """
        result = await self.client.search(
            index=self.es_index_name,
            query={"match": {"value": keyword}},
            min_score=threshold,
            size=limit,
        )
        return [(hit['_source'], float(hit.get('_score') or 0.0))
                for hit in result["hits"]["hits"]]

