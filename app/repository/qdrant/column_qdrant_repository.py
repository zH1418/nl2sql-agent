from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.repository.qdrant.base_qdrant_repository import BaseQdrantRepository


class ColumnQdrantRepository(BaseQdrantRepository[ColumnInfoQdrant]):
    collection_name = "nl2sql-agent-column"