
import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
    timeout=20.0,
    max_retries=0
)

MODEL = "openai/gpt-oss-20b"


# 1. Actual Python tools

def get_price(product: str) -> int:
    prices = {"laptop": 50000, "phone": 20000}

    if not isinstance(product, str):
        raise ValueError("Product must be a string")

    name = product.lower().strip()

    if name not in prices:
        raise ValueError("Product not found")

    return prices[name]


def apply_discount(price: float, percent: float) -> float:
    if type(price) not in (int, float):
        raise ValueError("Invalid price")

    if type(percent) not in (int, float):
        raise ValueError("Invalid percentage")

    if price < 0 or not 0 <= percent <= 100:
        raise ValueError("Invalid discount values")

    return round(price * (1 - percent / 100), 2)


# 2. Function registry

available_functions = {
    "get_price": get_price,
    "apply_discount": apply_discount
}


# 3. Schemas given to the LLM

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Get a product's price from the demo catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string"}
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Calculate the final price after a percentage discount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number"},
                    "percent": {"type": "number"}
                },
                "required": ["price", "percent"]
            }
        }
    }
]


# 4. Conversation

messages = [
    {
        "role": "system",
        "content": (
            "You are a shopping assistant. "
            "First fetch the product price using get_price. "
            "Then use apply_discount with the returned price. "
            "Never invent product prices."
        )
    },
    {
        "role": "user",
        "content": (
            "What is the final price of a laptop "
            "after a 15% discount? Use the tools."
        )
    }
]


# 5. Automatic tool-calling loop

MAX_ROUNDS = 5

for iteration in range(1, MAX_ROUNDS + 1):

    print(f"\n--- LLM CALL {iteration} ---")

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=False
    )

    message = response.choices[0].message

    # No tool call means final response
    if not message.tool_calls:
        print("Final Answer:", message.content)
        break

    # Preserve assistant tool requests
    messages.append(message)

    # Execute each requested tool
    for tool_call in message.tool_calls:

        name = tool_call.function.name
        print("Requested Tool:", name)

        try:
            arguments = json.loads(
                tool_call.function.arguments
            )

            function = available_functions.get(name)

            if function is None:
                raise ValueError("Unknown tool")

            result = function(**arguments)
            content = str(result)

            print("Arguments:", arguments)
            print("Tool Result:", result)

        except (ValueError, TypeError, KeyError) as error:
            content = json.dumps({"error": str(error)})
            print("Tool Error:", content)

        # Return tool result to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": name,
            "content": content
        })

        print(messages)

else:
    print("Stopped: maximum LLM rounds reached.")
