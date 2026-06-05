from __future__ import annotations

import time

from google import genai
from google.genai import types

from ..models import LLMResponse, TokenUsage
from ..pricing import compute_cost
from .base import LLMProvider


class GeminiProvider(LLMProvider):
    provider_name = "gemini"

    def __init__(
        self, api_key: str | None = None, default_model: str = "gemini-2.5-flash"
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self.default_model = default_model

    def complete(
        self, messages, *, model=None, temperature=0.7, max_tokens=1024
    ) -> LLMResponse:
        model = model or self.default_model
        # Gemini differences our seam hides: system prompt is separate; 'assistant' -> 'model'.
        system = "\n".join(m.content for m in messages if m.role == "system") or None
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
        )
        start = time.perf_counter()
        resp = self._client.models.generate_content(
            model=model, contents=contents, config=config
        )
        latency_ms = (time.perf_counter() - start) * 1000
        um = resp.usage_metadata
        usage = TokenUsage(
            input_tokens=um.prompt_token_count or 0,
            output_tokens=um.candidates_token_count or 0,
        )
        return LLMResponse(
            text=resp.text or "",
            provider=self.provider_name,
            model=model,
            usage=usage,
            cost_usd=compute_cost(model, usage),
            latency_ms=latency_ms,
            raw=resp.model_dump(),
        )

    def parse(
        self, messages, response_model, *, model=None, temperature=0.0, max_tokens=1024
    ):
        import time
        from ..models import ParsedResponse, TokenUsage
        from ..pricing import compute_cost

        model = model or self.default_model
        system = "\n".join(m.content for m in messages if m.role == "system") or None
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=response_model,  # schema-constrained
        )
        start = time.perf_counter()
        resp = self._client.models.generate_content(
            model=model, contents=contents, config=config
        )
        latency_ms = (time.perf_counter() - start) * 1000
        um = resp.usage_metadata
        usage = TokenUsage(
            input_tokens=um.prompt_token_count or 0,
            output_tokens=um.candidates_token_count or 0,
        )
        return ParsedResponse(
            data=resp.parsed,
            provider=self.provider_name,
            model=model,
            usage=usage,
            cost_usd=compute_cost(model, usage),
            latency_ms=latency_ms,
        )
