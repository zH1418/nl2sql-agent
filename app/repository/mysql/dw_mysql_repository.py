from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

class DWMysqlRepository:
    def __init__(self, session: AsyncSession):
        self.session = session


    async def get_column_types(self, table_name: str):
        """
        查询指定表字段的类型
        :param table_name:
        :return:
        """

        #定义sql语句
        sql = f"SHOW COLUMNS FROM {table_name}"
        # 执行sql
        result = await self.session.execute(text(sql))
        return {row.Field: row.Type for row in result.fetchall()}

    async def get_column_values(self, table_name:str, column_name:str,limit: int | None = 10):
        """
        查询指定字段的实例值
        :param table_name:
        :param column_name:
        :return:
        """
        # 定义sql
        sql =f"select distinct {column_name} from {table_name}"
        if limit is not None:
            sql += f" LIMIT {limit}"
        # 执行sql
        result = await self.session.execute(text(sql))
        return result.scalars().fetchall()

    async def get_db_info(self):
        """
        获取数据信息
        :return:
        """
        # 1.版本
        result = await self.session.execute(text("SELECT VERSION()"))
        version = result.scalar()
        # 2.方言
        dialect = self.session.get_bind().dialect.name
        return {"version": version, "dialect": dialect}

    async def validate_sql(self, sql: str):
        """
        explain校验sql
        :param sql:
        :return:
        """
        # 定义sql
        sql = f"explain {sql}"
        # 执行sql
        await self.session.execute(text(sql))

    async def execute_sql(self, sql: str):
        """
        执行sql
        :param sql:
        :return:
        """
        # 执行sql
        result = await self.session.execute(text(sql))
        # [(Row)(Row)] -->[{字段：值},{字段：值},{字段：值}]
        return [dict(row) for row in result.mappings().fetchall()]