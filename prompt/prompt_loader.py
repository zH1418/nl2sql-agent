import asyncio
from pathlib import Path


async def loader_prompt(name:str)->str:
    prompt_path=Path(__file__).parents[1]/'prompts'/f"{name}.prompt"
    return prompt_path.read_text(encoding="utf-8")


if __name__ == '__main__':
    async def test():

        prompt_text=await loader_prompt("extend_keywords_for_column_recall")
        print(prompt_text)

    asyncio.run(test())