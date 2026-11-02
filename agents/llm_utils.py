"""Shared LangChain helpers for LLM-backed agents."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


def build_chat_model(temperature: float = 0.2) -> Any:
    """Create the shared LLM client (Groq if configured, else OpenAI) used by LangChain agents."""
    groq_key = settings.GROQ_API_KEY
    if groq_key and not groq_key.startswith("PASTE_") and not groq_key.startswith("your_"):
        from langchain_groq import ChatGroq
        model_name = settings.GROQ_MODEL or "openai/gpt-oss-120b"
        return ChatGroq(
            model=model_name,
            api_key=groq_key,
            temperature=temperature,
        )

    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=temperature,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def invoke_with_retry(chain: Any, payload: dict) -> Any:
    """Invoke a LangChain runnable with retry for transient model/network failures."""
    return chain.invoke(payload)
