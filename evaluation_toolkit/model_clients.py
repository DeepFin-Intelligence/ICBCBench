import os
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1",
    max_retries=1
)