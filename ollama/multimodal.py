# OpenAI compatibility

from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

import base64
# opens the image file in "read binary" mode
with open("kashmir.jpg", "rb") as image_file:
    #reads the contents of the image as a bytes object
    binary_data = image_file.read() 
    #encodes the binary data using Base64 encoding
    base_64_encoded_data = base64.b64encode(binary_data) 
    #decodes base_64_encoded_data from bytes to a string
    base64_string = base_64_encoded_data.decode('utf-8')

    response = client.chat.completions.create(
        # model="deepseek-r1:7b",
        model="llava",
        messages=[
          {"role": "system", "content": "You are a helpful assistant to analyze given image"},
          {"role": "user", "content": [{
            "type": "image_url",
            # "image_url": "https://upload.wikimedia.org/wikipedia/commons/4/41/Beauty-of-Kashmir.jpg"
	    "image_url": f"data:image/jpeg;base64,{base64_string}",
            },
            {
            "type": "text",
            "text": """Describe what all you see in the image and also
		guess possible place from where is the image"""
            }]
          }
       ]
    )

    print(response.choices[0].message.content)
