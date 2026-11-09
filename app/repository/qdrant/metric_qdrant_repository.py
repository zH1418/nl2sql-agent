from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant
from app.repository.qdrant.base_qdrant_repository import BaseQdrantRepository

class MetricQdrantRepository(BaseQdrantRepository[MetricInfoQdrant]):
    collection_name = "nl2sql-agent-metric"