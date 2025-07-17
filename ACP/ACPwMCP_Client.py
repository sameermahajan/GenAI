import asyncio
import nest_asyncio
from acp_sdk.client import Client
from colorama import Fore 

nest_asyncio.apply() 

async def run_research_workflow() -> None:
    async with Client(base_url="http://localhost:8000") as research:
        run1 = await research.run_sync(
            agent="research_agent", input="Suggest research papers on MCP"
        )
        print("run output = ", run1.output)
        # content = run1.output[0].parts[0].content
        # print(Fore.LIGHTMAGENTA_EX+ content + Fore.RESET)

asyncio.run(run_research_workflow())