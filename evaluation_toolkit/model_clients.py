import os
from functools import lru_cache
from pathlib import Path
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

def resolve_openrouter_model(model):
    """Accept canonical provider IDs and the short names used in the README."""
    aliases = {
        "perplexity-deep-research": "perplexity/sonar-deep-research",
        "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
        "kimi-k2.5-thinking": "moonshotai/kimi-k2.5",
    }
    if model in aliases:
        return aliases[model]
    if "/" in model:
        return model
    for prefix, provider in (("gpt-", "openai"), ("o1", "openai"),
                             ("o3", "openai"), ("o4", "openai"),
                             ("gemini-", "google"), ("claude-", "anthropic")):
        if model.startswith(prefix):
            return f"{provider}/{model}"
    return model

@lru_cache(maxsize=1)
def get_openrouter_client():
    # Load only this checkout's configuration, never an ancestor checkout's .env.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Missing OPENROUTER_API_KEY. Set it in the environment or project .env.")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=1)


def load_project_env():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def require_complete_chat(response):
    """Do not checkpoint truncated, filtered, or unfinished tool-call output."""
    reason = getattr(response.choices[0], "finish_reason", None)
    if isinstance(reason, str) and reason != "stop":
        raise RuntimeError(f"Model response did not finish normally: {reason}")
    return response


class _LazyOpenRouter:
    """Delay credential validation until a request, so imports and --help work."""

    def __getattr__(self, name):
        return getattr(get_openrouter_client(), name)


OPENROUTER = _LazyOpenRouter()
