from langchain_community.llms import Ollama

llm = Ollama(model='deepseek-r1:7b')

response = llm.invoke("what is ollama?")

print (response)
