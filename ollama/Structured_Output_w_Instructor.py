# Structured Output using instructor

from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

import instructor

instructor_client = instructor.from_openai(client)

from pydantic import BaseModel
from pydantic import Field
from datetime import date
from typing import List

# Person only has a name, but we can easily extend it with other fields
class Person(BaseModel):
    name: str

class CalendarEvent(BaseModel):
    name: str

    # not supported by OpenAI. We want a format like 2025-01-30
    date: date
    participants: List[Person]

    address_number: str
    street_name: str
      city_name: str

    # Inline regex patterns not supported by OpenAI restrict state code
    # to be two capital letters (OpenAI does not support pattern fields)
    state_code: str = Field(pattern=r'[A-Z]{2}')

    # Zip code must be five digits
    zip_code: str = Field(pattern=r'\d{5}')

event_description = """
Sameer and Shivali are going to a science fair on Friday, January 2024.
The science fair is hosted at the gymnasium of Hazeldale Elementary
School at 20080 SW Farmington Road in Beaverton Oregon.
"""

system_prompt = """
Make a calendar event. Respond in JSON with
the event name, date, list of participants,
and the address.
"""

user_prompt = 'Event description: ' + event_description

event = instructor_client.chat.completions.create(
  model="llama2",
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
  ],
  response_model=CalendarEvent,
  max_retries=3
  )

print (event)
