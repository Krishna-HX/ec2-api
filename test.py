# def get_price(product : str) -> int:

#     prices = {"laptop" : 50}

#     if type(product) is not str:
#         raise ValueError("product must be string")

#     if product not in prices:
#         raise ValueError("product is not in the list")

#     return prices[product]


# print(get_price('laptop'))

from dotenv import load_dotenv
from groq import Groq
import os 

load_dotenv()

client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
    timeout=20.0,
    max_retries=0
)

MODEL = "openai/gpt-oss-20b"

messages = [{
    'role' : 'user',
    'content' : 'what is docker ? explain in just two lines'
}]

response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

print(response.choices[0].message.content)