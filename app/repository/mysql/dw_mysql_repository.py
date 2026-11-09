from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

class DWMysqlRepository:
    def __init__(self, session: AsyncSession):
        self.session = session


    async def get_column_types(self, table_name: str):

        sql = f"SHOW COLUMNS FROM {table_name}"
        result = await self.session.execute(text(sql))
        return {row.Field: row.Type for row in result.fetchall()}

    async def get_column_values(self, table_name:str, column_name:str,limit: int | None = 10):
        sql =f"select distinct {column_name} from {table_name}"
        if limit is not None:
            sql += f" LIMIT {limit}"
        result = await self.session.execute(text(sql))
        return result.scalars().fetchall()

    async def get_db_info(self):
        result = await self.session.execute(text("SELECT VERSION()"))
        version = result.scalar()
        dialect = self.session.get_bind().dialect.name
        return {"version": version, "dialect": dialect}

    async def validate_sql(self, sql: str):
        sql = f"explain {sql}"
        await self.session.execute(text(sql))

    async def execute_sql(self, sql: str):
        result = await self.session.execute(text(sql))
        return [dict(row) for row in result.mappings().fetchall()]