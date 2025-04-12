from langchain_ollama.llms import OllamaLLM

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="deepseek-r1:7b")

chain = prompt | model

response = chain.invoke({"question": "what is ollama"})

print (response)
