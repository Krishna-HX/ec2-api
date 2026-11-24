import json

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Step 1: Tool schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers accurately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

# Step 2: User message
messages = [
    {
        "role": "user",
        "content": "Use the multiply tool to calculate 125 times 48."
    }
]

# Step 3: Ask LLM to select a tool
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    tools=tools,
    tool_choice="required"
)

# Actual Python implementation
def multiply(a: int, b: int) -> int:
    return a * b

# Step 4: Inspect response
message = response.choices[0].message   

# Assistant ka tool-call message preserve karo
messages.append(message)
    
print("Content:", message.content)

for tool_call in message.tool_calls or []:
    function_name =  tool_call.function.name

    # JSON string -> Python dictionary
    arguments = json.loads(tool_call.function.arguments)

    if function_name != "multiply":
        raise ValueError("Unknown tool requested")

    # Schema ke alawa runtime validation bhi
    if (
        type(arguments.get("a")) is not int
        or type(arguments.get("b")) is not int
    ):
        raise ValueError("Invalid multiply arguments")

    result = multiply(**arguments)


    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": function_name,
        "content": str(result)
    })


print(messages)

# Second LLM call: result se final answer
final_response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages
)

print("Final Answer:", final_response.choices[0].message.content)