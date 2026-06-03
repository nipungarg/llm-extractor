from __future__ import annotations

import time

from openai import OpenAI

from ..models import LLMResponse, TokenUsage
from ..pricing import compute_cost
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(
        self, api_key: str | None = None, default_model: str = "gpt-4o-mini"
    ) -> None:
        self._client = OpenAI(api_key=api_key)  # None -> reads OPENAI_API_KEY
        self.default_model = default_model

    def complete(
        self, messages, *, model=None, temperature=0.7, max_tokens=1024
    ) -> LLMResponse:
        model = model or self.default_model
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                m.model_dump() for m in messages
            ],  # LLMMessage -> {"role","content"}
            temperature=temperature,
            max_completion_tokens=max_tokens,  # newer param name (was max_tokens)
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = TokenUsage(
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            provider=self.provider_name,
            model=model,
            usage=usage,
            cost_usd=compute_cost(model, usage),
            latency_ms=latency_ms,
            finish_reason=resp.choices[0].finish_reason,
            raw=resp.model_dump(),
        )
