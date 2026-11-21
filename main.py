
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import APIError, Groq, RateLimitError
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI()


@app.get("/")
def home():
    return {
        "status": "live on AWS EC2!",
        "environment": os.getenv("ENV_NAME", "default")
    }


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
def ask_llm(request: AskRequest):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Groq API key is not configured"
        )

    try:
        client = Groq(
            api_key=api_key,
            timeout=10.0,
            max_retries=0
        )

        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": request.prompt}
            ],
            max_completion_tokens=512
        )

    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Groq rate limit reached"
        ) from None

    except APIError:
        raise HTTPException(
            status_code=502,
            detail="LLM provider request failed"
        ) from None

    answer = completion.choices[0].message.content

    if not answer:
        raise HTTPException(
            status_code=502,
            detail="Empty LLM response"
        )

    return {"answer": answer}
