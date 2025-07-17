from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from smolagents import LiteLLMModel, ToolCallingAgent, ToolCollection
from mcp import StdioServerParameters

server = Server()

model = LiteLLMModel(
    model_id="ollama/llama3",  
    max_tokens=2048
)

server_params = StdioServerParameters(
            command="python",  # Executable
            args=["../MCP/research_server_simple.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )

@server.agent()
async def research_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a Research Agent which helps users find research papers."
    with ToolCollection.from_mcp(server_params, trust_remote_code=True) as tool_collection:
        print("tool collection = ", tool_collection)
        agent = ToolCallingAgent(tools=[*tool_collection.tools], model=model)
        prompt = input[0].parts[0].content
        response = agent.run(prompt)

    yield Message(parts=[MessagePart(content=str(response))])

if __name__ == "__main__":
    server.run(port=8000)