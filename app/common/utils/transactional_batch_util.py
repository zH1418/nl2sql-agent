from typing import List, Coroutine, Any
from sqlalchemy.ext.asyncio import AsyncSession


class TransactionalBatchUtil:

    @staticmethod
    async def save_in_transaction(
        session: AsyncSession,
        save_tasks: List[Coroutine[Any, Any, None]]
    ) -> None:
        async with session.begin():
            for task in save_tasks:
                await task