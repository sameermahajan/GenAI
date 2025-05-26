from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio

nest_asyncio.apply()

load_dotenv()

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        self.client = OpenAI(
                            base_url = 'http://localhost:11434/v1',
                            api_key='ollama', # required, but unused
                    )
        self.available_tools: List[dict] = []

    async def process_query(self, query):
        messages = [{'role':'user', 'content':query}]
        response = self.client.chat.completions.create(max_tokens = 2024,
                                      model = 'qwen2.5-coder:32b', 
                                      tools = self.available_tools,
                                      messages = messages)
        process_query = True
        while process_query:
            print(response.choices[0].message.tool_calls)
            tool_calls = response.choices[0].message.tool_calls
            for tool_call in tool_calls:
                tool_args = eval(tool_call.function.arguments)
                tool_name = tool_call.function.name
                print(f"Calling tool {tool_name} with args {tool_args}")
                # Call a tool
                result = await self.session.call_tool(tool_name, arguments=tool_args)
                print("made tool call\n")
                print("tool result is ", result)
            process_query = False

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python",  # Executable
            args=["mcp_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()

                # List available tools
                response = await session.list_tools()
                
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])
                
                self.available_tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in response.tools]
    
                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()
  

if __name__ == "__main__":
    asyncio.run(main())

# Try following queries
#
# search_papers on topic MCP
# search_papers on topic MCP with max_results of 2

# extract_info for paper_id 2504.08999v1