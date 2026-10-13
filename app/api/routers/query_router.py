import asyncio

from fastapi import APIRouter
from fastapi.params import Depends
from starlette.responses import StreamingResponse

from app.api.dependencies import get_query_service
from app.api.schemas.query_schema import QueryScheme
from app.services.query_service import QueryService

# 创建路由实例
query_router = APIRouter()


@query_router.post("/api/query")
async def query(query:QueryScheme,service:QueryService=Depends(get_query_service)):

    return StreamingResponse(
        service.query(query.query, conversation_id=query.conversation_id, answers=query.answers),
        media_type="text/event-stream"
    )