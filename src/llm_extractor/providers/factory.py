from __future__ import annotations

from ..config import settings
from .base import LLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or settings.default_provider).lower()
    if name == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key or None, default_model=settings.openai_model
        )
    if name == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key or None, default_model=settings.gemini_model
        )
    raise ValueError(f"Unknown provider: {name!r}")
