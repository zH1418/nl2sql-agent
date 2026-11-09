import asyncio
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, async_sessionmaker

from app.conf.app_config import DBConfig, app_config


class MysqlClientManager:
    def __init__(self,db_config:DBConfig):
        self.db_config = db_config
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None

    def _get_url(self):
        return (f"mysql+asyncmy://{self.db_config.user}"
                f":{self.db_config.password}@{self.db_config.host}"
                f":{self.db_config.port}/{self.db_config.database}"
                f"?charset=utf8mb4")

    def init(self):
        if self.engine is None:
            self.engine = create_async_engine(
                url=self._get_url(),
                pool_size=5,
                pool_pre_ping=True
            )

            self.session_factory = async_sessionmaker(
                bind=self.engine,
                autoflush=True,
                expire_on_commit=False
            )

    async def close(self):
        if self.engine:
            await self.engine.dispose()
            await asyncio.sleep(0.1)

dw_mysql_client_manager = MysqlClientManager(app_config.db_dw)
meta_mysql_client_manager = MysqlClientManager(app_config.db_meta)

if __name__ == '__main__':
    dw_mysql_client_manager.init()

    async def test():
        async with dw_mysql_client_manager.session_factory() as session:
            result = await session.execute(text("select * from fact_order limit 10"))
            rows = result.mappings().fetchall()
            print(type(rows[0]))
            print(rows[0])

    asyncio.run(test())
