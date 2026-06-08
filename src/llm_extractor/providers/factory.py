from __future__ import annotations

from ..config import settings
from .base import LLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from ..errors import ProviderError


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


class FallbackProvider(LLMProvider):
    """Try each provider in order; on a ProviderError, fall back to the next one.
    Because every provider returns the same LLMResponse, callers never notice the switch."""

    provider_name = "fallback"

    def __init__(self, providers: list[LLMProvider]) -> None:
        self._providers = providers

    def complete(self, messages, **kwargs):
        last_error = None
        for p in self._providers:
            try:
                return p.complete(messages, **kwargs)
            except ProviderError as e:
                last_error = e  # log this in real life, then try the next
        raise last_error

    def parse(self, messages, response_model, **kwargs):
        last_error = None
        for p in self._providers:
            try:
                return p.parse(messages, response_model, **kwargs)
            except ProviderError as e:
                last_error = e
        raise last_error


def get_fallback_provider() -> FallbackProvider:
    # primary = OpenAI, backup = Gemini
    return FallbackProvider([get_provider("openai"), get_provider("gemini")])
