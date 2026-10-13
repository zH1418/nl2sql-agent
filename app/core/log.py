import asyncio
import sys
from pathlib import Path

from Lib import uuid
from loguru import logger

from app.conf.app_config import app_config
from app.core.context import request_id_ctx_var

# 日志格式
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<magenta>request_id - {extra[request_id]}</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def inject_request_id(record):
    """
    为日志记录注入 request_id，用于请求链路追踪。
    若获取失败则生成 UUID 作为兜底。
    """
    try:
        request_id = request_id_ctx_var.get()
    except Exception:
        request_id = uuid.uuid4()
    record["extra"]["request_id"] = request_id


logger.remove()
logger = logger.patch(inject_request_id)

if app_config.logging.console.enable:
    logger.add(sink=sys.stdout, level=app_config.logging.console.level, format=log_format)

if app_config.logging.file.enable:
    path = Path(__file__).parents[2] / "logs"
    path.mkdir(parents=True, exist_ok=True)
    logger.add(
        sink=path / "api.log",
        level=app_config.logging.file.level,
        format=log_format,
        rotation=app_config.logging.file.rotation,
        retention=app_config.logging.file.retention,
        encoding="utf-8"
    )

if __name__ == '__main__':
    async def graph():
        logger.info("完成调用")

    async def test1():
        request_id_ctx_var.set("111111111")
        await asyncio.sleep(1)
        await graph()

    async def test2():
        request_id_ctx_var.set("2222222222")
        await asyncio.sleep(1)
        await graph()

    async def main():
        await asyncio.gather(test1(), test2())

    asyncio.run(main())