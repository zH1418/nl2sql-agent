from typing import List, Coroutine, Any
from sqlalchemy.ext.asyncio import AsyncSession


class TransactionalBatchUtil:
    """
    数据库事务批量工具类
    封装事务上下文，统一管理多组数据在同一事务内的批量写入
    避免业务代码重复编写事务上下文，保证事务边界一致性
    """

    @staticmethod
    async def save_in_transaction(
        session: AsyncSession,
        save_tasks: List[Coroutine[Any, Any, None]]
    ) -> None:
        """
        在单个数据库事务中顺序执行多个批量保存操作
        :param session: 数据库会话实例
        :param save_tasks: 待执行的保存操作协程列表，按列表顺序在事务内执行
        """
        async with session.begin():
            for task in save_tasks:
                await task