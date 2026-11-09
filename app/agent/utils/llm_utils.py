from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser

from app.agent.llm import llm
from app.core.log import logger
from prompt.prompt_loader import loader_prompt


def content_to_text(content) -> str:
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
                if blk.get("type") in (None, "text") and blk.get("text") is not None:
                    parts.append(str(blk["text"]))
            elif hasattr(blk, "text"):
                parts.append(str(getattr(blk, "text") or ""))
        return "".join(parts)
    return str(content)


async def invoke_llm_chain(prompt_name: str, output_parser: BaseOutputParser = None, **kwargs):
    template = await loader_prompt(prompt_name)

    text = template
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", str(value))

    parser = output_parser if output_parser is not None else JsonOutputParser()
    resp = await llm.ainvoke([("human", text)])
    content = content_to_text(getattr(resp, "content", ""))
    return parser.parse(content)
