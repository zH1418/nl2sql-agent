import argparse
import asyncio
from pathlib import Path

from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import meta_mysql_client_manager, dw_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.repository.es.value_es_repository import ValueEsRepository
from app.repository.mysql.dw_mysql_repository import DWMysqlRepository
from app.repository.mysql.meta_mysql_repository import MetaMysqlRepository
from app.repository.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repository.qdrant.metric_qdrant_repository import MetricQdrantRepository
from app.services.meta_knowledge_service import MetaKnowledgeService


async def build(file_path:Path):

    # 初始化客户端对象
    meta_mysql_client_manager.init()
    dw_mysql_client_manager.init()
    qdrant_client_manager.init()
    embedding_client_manager.init()
    es_client_manager.init()

    # 获取session
    async with meta_mysql_client_manager.session_factory() as meta_session,dw_mysql_client_manager.session_factory()as dw_session:

        # 创建repository
        meta_mysql_repository= MetaMysqlRepository(meta_session)
        dw_mysql_repository=DWMysqlRepository(dw_session)
        # 创建qdrant客户端对象
        column_qdrant_repository=ColumnQdrantRepository(qdrant_client_manager.client)
        metric_qdrant_repository=MetricQdrantRepository(qdrant_client_manager.client)
        # 创建embedding客户端对象
        embedding_client=embedding_client_manager.client
        # 创建ValueEsRepository对象
        value_es_repository=ValueEsRepository(es_client_manager.es_client)

        # 创建service层对象
        meta_knowledge_service= MetaKnowledgeService(
            meta_mysql_repository=meta_mysql_repository,
            dw_mysql_repository=dw_mysql_repository,
            column_qdrant_repository=column_qdrant_repository,
            metric_qdrant_repository=metric_qdrant_repository,
            embedding_client=embedding_client,
            value_es_repository=value_es_repository,
        )
        # 调用业务层
        await meta_knowledge_service.build(file_path)
    # 释放资源
    await meta_mysql_client_manager.close()
    await dw_mysql_client_manager.close()
    await qdrant_client_manager.close()
    await es_client_manager.close()






if __name__ == '__main__':
    # 创建argparse的实例对象
    parser = argparse.ArgumentParser()
    # 设置参数要求
    parser.add_argument('-c', '--conf')  # 接受一个值的选项
    # 解析参数
    args = parser.parse_args()
    # 获取路径参数，转换为path对象
    file_path=Path(args.conf)
    asyncio.run(build(file_path))

