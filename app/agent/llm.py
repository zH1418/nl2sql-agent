from langchain_openai import ChatOpenAI
from app.conf.app_config import app_config

llm = ChatOpenAI(
    model=app_config.llm.model_name,
    api_key=app_config.llm.api_key,
    base_url=app_config.llm.base_url,
    temperature=0,
    timeout=30,
    extra_body={"enable_thinking": False}   # 作为直接参数
)

if __name__ == '__main__':
    for chunk in llm.stream("你好，介绍一下你自己"):
        print(chunk.text, end="")