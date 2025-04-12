# Structured Output

from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

response = client.beta.chat.completions.parse(
 model="deepseek-r1:7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Make up a user."},
    ],
    response_format=User,
)
print(response.choices[0].message.parsed)
