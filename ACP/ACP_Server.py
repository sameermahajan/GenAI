from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server

from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool

import warnings
warnings.filterwarnings('ignore')

import nest_asyncio

nest_asyncio.apply()

server = Server()

llm=LLM(model="ollama/llama3", base_url="http://localhost:11434")

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3",
        }
    },
    "embedding_model": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text"
        }
    }
}

rag_tool = RagTool(config=config,  
                   chunk_size=1200,       
                   chunk_overlap=200,     
                  )
# rag_tool.add("./sameer_mahajan.pdf", data_type="pdf_file")
rag_tool.add("./gold-hospital-and-premium-extras.pdf", data_type="pdf_file")

@server.agent()
async def policy_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is an agent for questions around policy coverage, it uses a RAG pattern to find answers based on policy documentation. Use it to help answer questions on coverage and waiting periods."

    insurance_agent = Agent(
        role="Senior Insurance Coverage Assistant", 
        goal="Determine whether something is covered or not",
        backstory="You are an expert insurance agent designed to assist with coverage queries",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[rag_tool], 
        max_retry_limit=5
    )

    task1 = Task(
            description=input[0].parts[0].content,
            expected_output = "A comprehensive response as to the users question",
            agent=insurance_agent
    )

    crew = Crew(agents=[insurance_agent], tasks=[task1], verbose=True)
    task_output = await crew.kickoff_async()
    yield Message(parts=[MessagePart(content=str(task_output))])

if __name__ == "__main__":
    server.run(port=8001)
