from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool

import warnings
warnings.filterwarnings('ignore')

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
        description='What is the waiting period for rehabilitation?',
        expected_output = "A comprehensive response as to the users question",
        agent=insurance_agent
)

crew = Crew(agents=[insurance_agent], tasks=[task1], verbose=True)
task_output = crew.kickoff()
print(task_output) 
