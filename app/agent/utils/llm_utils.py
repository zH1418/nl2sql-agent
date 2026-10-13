from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser

from app.agent.llm import llm
from app.core.log import logger
from prompt.prompt_loader import loader_prompt


def content_to_text(content) -> str:
    """把 LLM 的 `content` 归一为**纯文本**。

    ★ 为什么需要：部分端点（实测 `ark-code-latest` @ `/api/coding/v3`）返回的是
    **内容块列表**（如 `[{"type":"text","text":"...","index":0}]`）而不是字符串。
    直接拿它去 `JsonOutputParser.parse()` 会炸，或在 `Generation.text` 校验时报
    `Input should be a valid string`。
    ⇒ 这里统一：列表 ⇒ 抽取其中 `type=="text"` 的 `text` 片段并拼接；字符串 ⇒ 原样。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                # 只取文本块；reasoning / tool_use 等非文本块忽略
                if blk.get("type") in (None, "text") and blk.get("text") is not None:
                    parts.append(str(blk["text"]))
            elif hasattr(blk, "text"):
                parts.append(str(getattr(blk, "text") or ""))
        return "".join(parts)
    return str(content)


async def invoke_llm_chain(prompt_name: str, output_parser: BaseOutputParser = None, **kwargs):
    """
    通用 LLM 调用，自动构建 Prompt 并执行链。

    ★ 不走 langchain `PromptTemplate`：
      其 template_format 默认是 **f-string**，会把提示词里的 **JSON 花括号**
      （如 `{ "status": "clear" }`）当成占位符，直接抛
      `Invalid format specifier in f-string template`。
      改为**手工替换 `{var}` + 直接 ainvoke**（与 `generate_sql` 选词 prompt 的内嵌口径一致）。

    :param prompt_name: 提示词模板名称（不含路径）
    :param output_parser: 输出解析器，默认 JsonOutputParser
    :param kwargs: 模板变量键值对
    :return: 解析后的结果
    """
    template = await loader_prompt(prompt_name)

    # 手工替换占位符（只替换传入的变量名，其余花括号原样保留）
    text = template
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", str(value))

    parser = output_parser if output_parser is not None else JsonOutputParser()
    resp = await llm.ainvoke([("human", text)])
    content = content_to_text(getattr(resp, "content", ""))
    return parser.parse(content)
