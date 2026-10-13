from pathlib import Path
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from omegaconf import OmegaConf
from app.conf.meta_config import MetaConfig
from app.core.log import logger
from app.common.utils.qdrant_vector_store_util import QdrantVectorStoreUtil
from app.common.utils.transactional_batch_util import TransactionalBatchUtil
from app.common.utils.entity_mapper_util import EntityMapperUtil
from app.models.es.value_info_es import ValueInfoEs
from app.models.mysql.column_info_mysql import ColumnInfoMySQL
from app.models.mysql.column_metric_mysql import ColumnMetricMySQL
from app.models.mysql.metric_info_mysql import MetricInfoMySQL
from app.models.mysql.table_info_mysql import TableInfoMySQL
from app.models.qdrant.column_info_qdrant import ColumnInfoQdrant
from app.models.qdrant.metric_info_qdrant import MetricInfoQdrant
from app.repository.es.value_es_repository import ValueEsRepository
from app.repository.mysql.dw_mysql_repository import DWMysqlRepository
from app.repository.mysql.meta_mysql_repository import MetaMysqlRepository
from app.repository.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repository.qdrant.metric_qdrant_repository import MetricQdrantRepository


class MetaKnowledgeService:
    def __init__(self,
                 meta_mysql_repository: MetaMysqlRepository,
                 dw_mysql_repository: DWMysqlRepository,
                 column_qdrant_repository: ColumnQdrantRepository,
                 metric_qdrant_repository: MetricQdrantRepository,
                 embedding_client: HuggingFaceEndpointEmbeddings,
                 value_es_repository: ValueEsRepository):
        self.meta_mysql_repository = meta_mysql_repository
        self.dw_mysql_repository = dw_mysql_repository
        self.column_qdrant_repository = column_qdrant_repository
        self.metric_qdrant_repository = metric_qdrant_repository
        self.embedding_client = embedding_client
        self.value_es_repository = value_es_repository

    async def build(self, config_file: Path):
        # 1.加载配置文件，构建配置对象
        context = OmegaConf.load(config_file)
        schema = OmegaConf.structured(MetaConfig)
        meta_config: MetaConfig = OmegaConf.to_object(OmegaConf.merge(schema, context))
        logger.info("加载配置文件完成")

        if meta_config.tables:
            # 2.处理表信息
            column_infos: list[ColumnInfoMySQL] = await self._save_table_info_to_meta_db(meta_config)
            logger.info("保存表信息到meta数据库")
            await self._save_column_info_to_qdrant(column_infos)
            logger.info("为列信息添加向量索引")
            await self._save_value_info_to_es(column_infos, meta_config)
            logger.info("为值信息添加全文索引")

        if meta_config.metrics:
            # 3.处理指标信息
            metric_infos: list[MetricInfoMySQL] = await self._save_metric_info_to_meta_db(meta_config)
            logger.info("保存指标信息到meta数据库")
            await self._save_metric_info_to_qdrant(metric_infos)
            logger.info("为指标信息添加向量索引")

    async def _save_table_info_to_meta_db(self, meta_config: MetaConfig) -> list[ColumnInfoMySQL]:
        table_infos: list[TableInfoMySQL] = []
        column_infos: list[ColumnInfoMySQL] = []
        for table in meta_config.tables:
            column_types: dict[str, str] = await self.dw_mysql_repository.get_column_types(table.name)
            table_info = TableInfoMySQL(
                id=table.name,
                name=table.name,
                role=table.role,
                description=table.description
            )
            table_infos.append(table_info)

            for column in table.columns:
                column_values: list[str] = await self.dw_mysql_repository.get_column_values(table.name, column.name)
                column_info = ColumnInfoMySQL(
                    id=f"{table.name}.{column.name}",
                    name=column.name,
                    type=column_types[column.name],
                    role=column.role,
                    examples=column_values,
                    description=column.description,
                    alias=column.alias,
                    table_id=table.name
                )
                column_infos.append(column_info)

        # 复用事务工具类，统一事务边界
        await TransactionalBatchUtil.save_in_transaction(
            session=self.meta_mysql_repository.session,
            save_tasks=[
                self.meta_mysql_repository.save_table_infos(table_infos),
                self.meta_mysql_repository.save_column_infos(column_infos)
            ]
        )
        return column_infos

    async def _save_column_info_to_qdrant(self, column_infos: list[ColumnInfoMySQL]):
        """重构后：向量写入复用通用工具，实体转换复用映射工具，无手写样板代码"""
        def extract_embedding_texts(col: ColumnInfoMySQL) -> list[str]:
            return [col.name, col.description, *col.alias]

        await QdrantVectorStoreUtil.batch_upsert(
            embedding_client=self.embedding_client,
            qdrant_repo=self.column_qdrant_repository,
            entities=column_infos,
            text_extractor=extract_embedding_texts,
            payload_converter=lambda col: EntityMapperUtil.map_fields(col, ColumnInfoQdrant),
            batch_size=20
        )

    async def _save_value_info_to_es(self, column_infos: list[ColumnInfoMySQL], meta_config: MetaConfig):
        await self.value_es_repository.ensure_index()
        column2sync: dict[str, bool] = {}
        for table in meta_config.tables:
            for column in table.columns:
                column2sync[f"{table.name}.{column.name}"] = column.sync

        value_infos: list[ValueInfoEs] = []
        for column_info in column_infos:
            sync = column2sync[column_info.id]
            if sync:
                values: list[str] = await self.dw_mysql_repository.get_column_values(
                    column_info.table_id, column_info.name, 10000
                )
                current_column_values = [ValueInfoEs(
                    id=f"{column_info.id}.{value}",
                    value=value,
                    type=column_info.type,
                    column_id=column_info.id,
                    column_name=column_info.name,
                    table_id=column_info.table_id,
                    table_name=column_info.table_id
                ) for value in values]
                value_infos.extend(current_column_values)

        await self.value_es_repository.upsert_values(value_infos)

    async def _save_metric_info_to_meta_db(self, meta_config: MetaConfig):
        metric_infos: list[MetricInfoMySQL] = []
        column_metric_infos: list[ColumnMetricMySQL] = []
        for metric in meta_config.metrics:
            metric_info = MetricInfoMySQL(
                id=metric.name,
                name=metric.name,
                description=metric.description,
                relevant_columns=metric.relevant_columns,
                alias=metric.alias
            )
            metric_infos.append(metric_info)

            for relevant_column in metric.relevant_columns:
                relevant_column_info = ColumnMetricMySQL(
                    column_id=relevant_column,
                    metric_id=metric.name
                )
                column_metric_infos.append(relevant_column_info)

        # 复用事务工具类，统一事务边界
        await TransactionalBatchUtil.save_in_transaction(
            session=self.meta_mysql_repository.session,
            save_tasks=[
                self.meta_mysql_repository.save_metric_infos(metric_infos),
                self.meta_mysql_repository.save_column_metric_infos(column_metric_infos)
            ]
        )
        return metric_infos

    async def _save_metric_info_to_qdrant(self, metric_infos: list[MetricInfoMySQL]):
        """重构后：向量写入复用通用工具，实体转换复用映射工具，无手写样板代码"""
        def extract_embedding_texts(metric: MetricInfoMySQL) -> list[str]:
            return [metric.name, metric.description, *metric.alias]

        await QdrantVectorStoreUtil.batch_upsert(
            embedding_client=self.embedding_client,
            qdrant_repo=self.metric_qdrant_repository,
            entities=metric_infos,
            text_extractor=extract_embedding_texts,
            payload_converter=lambda metric: EntityMapperUtil.map_fields(metric, MetricInfoQdrant),
            batch_size=10
        )