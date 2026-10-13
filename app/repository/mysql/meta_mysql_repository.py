from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.mysql.column_info_mysql import ColumnInfoMySQL
from app.models.mysql.column_metric_mysql import ColumnMetricMySQL
from app.models.mysql.metric_info_mysql import MetricInfoMySQL
from app.models.mysql.table_info_mysql import TableInfoMySQL


class MetaMysqlRepository():
    def __init__(self,session:AsyncSession):
        self.session:AsyncSession = session

    async def save_table_infos(self, table_infos:list[TableInfoMySQL]):
        self.session.add_all(table_infos)

    async def save_column_infos(self, column_infos:list[ColumnInfoMySQL]):
        self.session.add_all(column_infos)

    async def save_metric_infos(self, metric_infos: list[MetricInfoMySQL]):
        self.session.add_all(metric_infos)

    async def save_column_metric_infos(self, column_metric_infos: list[ColumnMetricMySQL]):
        self.session.add_all(column_metric_infos)

    async def get_column_info_by_id(self, column_id: str):
        """
        根据ID查询字段信息
        :param column_id:
        :return:
        """
        return await self.session.get(ColumnInfoMySQL, column_id)

    async def get_column_key_by_id(self, table_id: str):
        # 定义sql语句
        sql = """ \
              select * \
              from column_info \
              where role in ('primary_key', 'foreign_key') \
                and table_id = :table_id \
             """

        # 设置返回结果
        query = select(ColumnInfoMySQL).from_statement(text(sql))

        # 执行sql
        result = await self.session.execute(query, {"table_id": table_id})
        # 获取结果[(ColumnInfoMysql),(ColumnInfoMysql),(ColumnInfoMysql)]
        return result.scalars().fetchall()

    async def get_table_info_by_id(self, table_id: str):
        return await self.session.get(TableInfoMySQL, table_id)