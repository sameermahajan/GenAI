
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import json
import asyncio

load_dotenv()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = [] # new
        self.exit_stack = AsyncExitStack() # new
        self.client = OpenAI(
                            base_url = 'http://localhost:11434/v1',
                            api_key='ollama', # required, but unused
                    )
        self.available_tools: List[ToolDefinition] = [] # new
         # Prompts list for quick display 
        self.available_prompts = []
        # Sessions dict maps tool/prompt names or resource URIs to MCP client sessions
        self.sessions = {}

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            ) # new
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            ) # new
            await session.initialize()
            
            try:
                # List available tools
                response = await session.list_tools()
                tools = response.tools

                print("\nConnected to server ", server_name, " with tools:", [tool.name for tool in tools])

                for tool in tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
            
                # List available prompts
                prompts_response = await session.list_prompts()
                prompts = prompts_response.prompts

                print("\nConnected to server ", server_name, " with prompts:", [prompt.name for prompt in prompts])

                if prompts_response and prompts:
                    for prompt in prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        })
                # List available resources
                resources_response = await session.list_resources()
                resources = resources_response.resources

                print("\nConnected to server ", server_name, " with resources:", [resource.uri for resource in resources])

                if resources_response and resources:
                    for resource in resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
            
            except Exception as e:
                print(f"Error {e}")
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self): # new
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise
    
    async def process_query(self, query):
        messages = [{'role':'user', 'content':query}]
        response = self.client.chat.completions.create(max_tokens = 2024,
                                      model = 'qwen2.5-coder:32b', 
                                      tools = self.available_tools,
                                      messages = messages)
        process_query = True
        while process_query:
            assistant_content = []
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                content = response.choices[0].message.content
                assistant_content.append(content)
                messages.append({'role':'assistant', 'content':assistant_content})
                for tool_call in tool_calls:
                    tool_args = eval(tool_call.function.arguments)
                    tool_name = tool_call.function.name
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    # Call a tool                    
                    result = await self.sessions[tool_name].call_tool(tool_name, arguments=tool_args)
                    print("made tool call\n")
                    print("tool result is ", result)
                    messages.append({
                        "role": "user", 
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": result
                            }
                        ]
                    })
                process_query = False
            else:
                content = response.choices[0].message.content
                print(content)
                assistant_content.append(content)
                if(len(content) == 1):
                    process_query= False
    
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
    
    async def cleanup(self): # new
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        # the mcp clients and sessions are not initialized using "with"
        # like in the previous lesson
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers() # new! 
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup() #new! 


if __name__ == "__main__":
    asyncio.run(main())

# call tool list_directory with path .
# call tool write_file with path new_file content 'This is new file'
# call tool search_papers with topic MCP and then call tool write_file with path new_file and content as result of search_papers
