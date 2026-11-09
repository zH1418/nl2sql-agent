from typing import Optional

from pydantic import BaseModel


class QueryScheme(BaseModel):
    query: str
    conversation_id: Optional[str] = None   # 第二轮澄清时携带的会话 ID
    answers: Optional[dict] = None           # 用户填写的澄清表单答案