from fastapi import Depends
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import meta_mysql_client_manager, dw_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.repository.es.value_es_repository import ValueEsRepository
from app.repository.mysql.dw_mysql_repository import DWMysqlRepository
from app.repository.mysql.meta_mysql_repository import MetaMysqlRepository
from app.repository.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repository.qdrant.metric_qdrant_repository import MetricQdrantRepository

from app.services.query_service import QueryService

async def get_embedding_client():
    return embedding_client_manager.client


async def get_column_qdrant_repository():

    return ColumnQdrantRepository(qdrant_client_manager.client)


async def get_metric_qdrant_repository():
    return MetricQdrantRepository(qdrant_client_manager.client)

async def get_value_es_repository():
    return ValueEsRepository(es_client_manager.es_client)

async def get_meta_session():
    async with meta_mysql_client_manager.session_factory() as session:
        yield  session


async def get_dw_session():
    async with dw_mysql_client_manager.session_factory() as session:
        yield  session

async def get_meta_mysql_repository(meta_session:AsyncSession=Depends(get_meta_session)):

    return MetaMysqlRepository(meta_session)


async def get_dw_mysql_repository(dw_session:AsyncSession=Depends(get_dw_session)):

    return DWMysqlRepository(dw_session)




async def get_query_service(
                 embedding_client: HuggingFaceEndpointEmbeddings= Depends(get_embedding_client),
                 column_qdrant_repository: ColumnQdrantRepository=Depends(get_column_qdrant_repository),
                 metric_qdrant_repository: MetricQdrantRepository= Depends(get_metric_qdrant_repository),
                 value_es_repository: ValueEsRepository=Depends(get_value_es_repository),
                 meta_mysql_repository: MetaMysqlRepository=Depends(get_meta_mysql_repository),
                 dw_mysql_repository: DWMysqlRepository=Depends(get_dw_mysql_repository)
):
    return QueryService(
        embedding_client=embedding_client,
        column_qdrant_repository=column_qdrant_repository,
        metric_qdrant_repository=metric_qdrant_repository,
        value_es_repository=value_es_repository,
        meta_mysql_repository=meta_mysql_repository,
        dw_mysql_repository=dw_mysql_repository

    )