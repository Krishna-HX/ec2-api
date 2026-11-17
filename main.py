import os

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "status": "live on AWS EC2!",
        "environment": os.getenv("ENV_NAME", "default")
    }