from typing import TypedDict
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from app.repository.es.value_es_repository import ValueEsRepository
from app.repository.mysql.dw_mysql_repository import DWMysqlRepository
from app.repository.mysql.meta_mysql_repository import MetaMysqlRepository
from app.repository.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repository.qdrant.metric_qdrant_repository import MetricQdrantRepository


class DataAgentContext(TypedDict):
    embedding_client: HuggingFaceEndpointEmbeddings  # 向量服务器客户端
    column_qdrant_repository: ColumnQdrantRepository  # 字段向量库
    metric_qdrant_repository: MetricQdrantRepository  # 指标向量库
    value_es_repository: ValueEsRepository  # 字段值索引库
    meta_mysql_repository: MetaMysqlRepository  # 元数据MySQL仓库（表、列、指标等信息）
    dw_mysql_repository: DWMysqlRepository  # 数据仓库MySQL仓库（列类型、列值、数据库信息）
