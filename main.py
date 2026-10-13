import uuid
from fastapi import FastAPI, Request

from app.api.routers.query_router import query_router
from app.core.context import request_id_ctx_var
from app.core.lifespan import lifespan

# 声明fastapi对象
app = FastAPI(lifespan=lifespan)

# 绑定查询router
app.include_router(query_router)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    # 调用路径函数之前
    request_id_ctx_var.set(uuid.uuid4())
    # 调用路径函数
    response = await call_next(request)
    # 调用路径函数之后
    return response