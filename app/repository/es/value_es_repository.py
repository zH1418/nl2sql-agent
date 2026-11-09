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
        if not await self.client.indices.exists(index=self.es_index_name):
            await self.client.indices.create(
                index=self.es_index_name,
                mappings=self.es_index_mappings
            )

    async def upsert_values(self, value_infos: list[ValueInfoEs], batch_size: int = 20):
        for i in range(0, len(value_infos), batch_size):
            operations = []
            batch_value_infos = value_infos[i:i + batch_size]
            for batch_value_info in batch_value_infos:
                operations.append({"index": {"_index": self.es_index_name}})
                operations.append(batch_value_info)
            await self.client.bulk(
                operations=operations
            )

    async def search(self, keyword: str, threshold: float = 0.6, limit: int = 5) -> list[ValueInfoEs]:
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
        return [hit['_source'] for hit in result["hits"]["hits"]]

    async def search_scored(self, keyword: str, threshold: float = 0.6,
                            limit: int = 5) -> list[tuple[ValueInfoEs, float]]:
        result = await self.client.search(
            index=self.es_index_name,
            query={"match": {"value": keyword}},
            min_score=threshold,
            size=limit,
        )
        return [(hit['_source'], float(hit.get('_score') or 0.0))
                for hit in result["hits"]["hits"]]

