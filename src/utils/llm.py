# src/utils/llm.py
import os
from langchain_openai import ChatOpenAI


def create_llm(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", "deepseek-chat"),
        base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=api_key or os.getenv("LLM_API_KEY", "sk-placeholder"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
