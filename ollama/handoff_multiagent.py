from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
import asyncio

set_tracing_disabled(True)

model = OpenAIChatCompletionsModel( 
    model="llama3.1",
    openai_client=AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="unused")
)

# hola
spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    model=model
)

# salut
french_agent = Agent(
    name="French agent",
    instructions="You only speak French.",
    model=model
)

# hi
english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model=model
)

# नमस्ते
hindi_agent = Agent(
    name="Hindi agent",
    instructions="You only speak Hindi",
    model=model
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, french_agent, english_agent, hindi_agent],
    model=model
)


async def main():
    while True:
        user_input = input(">").strip()
        # print ("user_input is '", user_input, "'")
        if "quit" in user_input:
            break
        result = await Runner.run(triage_agent, input = user_input)
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())