from __future__ import annotations

import time

import httpx
from google import genai
from google.genai import types, errors as genai_errors

from ..models import LLMResponse, TokenUsage
from ..pricing import compute_cost
from .base import LLMProvider

from ..errors import ProviderError, ProviderRateLimit, ProviderTimeout, retry_transient


def _gemini_contents(messages, images):
    """Build Gemini contents, attaching image Parts to the first user turn."""
    contents, attached = [], False
    for m in messages:
        if m.role == "system":
            continue                                  # system goes in system_instruction, not contents
        parts = [types.Part(text=m.content)]
        if m.role == "user" and images and not attached:
            for img in images:
                mime = "image/jpeg" if img[:3] == b"\xff\xd8\xff" else "image/png"
                parts.append(types.Part.from_bytes(data=img, mime_type=mime))
            attached = True
        contents.append(types.Content(role="model" if m.role == "assistant" else "user", parts=parts))
    return contents

class GeminiProvider(LLMProvider):
    provider_name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gemini-2.5-flash",
        timeout: float = 30.0,
    ) -> None:
        # timeout: don't let a single call hang forever (HttpOptions wants milliseconds)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )
        self.default_model = default_model

    @retry_transient
    def complete(
        self, messages, *, model=None, temperature=0.7, max_tokens=1024, images=None
    ) -> LLMResponse:
        model = model or self.default_model
        # Gemini differences our seam hides: system prompt is separate; 'assistant' -> 'model'.
        system = "\n".join(m.content for m in messages if m.role == "system") or None
        contents = _gemini_contents(messages, images)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
        )
        start = time.perf_counter()
        try:
            resp = self._client.models.generate_content(
                model=model, contents=contents, config=config
            )
        # google-genai raises APIError subclasses; 429 -> rate limit, 5xx -> server (retryable)
        except httpx.TimeoutException as e:
            raise ProviderTimeout(str(e)) from e  # client-side timeout (retryable)
        except genai_errors.ClientError as e:
            raise (
                ProviderRateLimit(str(e))
                if getattr(e, "code", None) == 429
                else ProviderError(str(e))
            ) from e
        except genai_errors.ServerError as e:
            raise ProviderTimeout(
                str(e)
            ) from e  # treat transient server errors as retryable
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

    @retry_transient
    def parse(
        self, messages, response_model, *, model=None, temperature=0.0, max_tokens=1024, images=None
    ):
        import time
        from ..models import ParsedResponse, TokenUsage
        from ..pricing import compute_cost

        model = model or self.default_model
        system = "\n".join(m.content for m in messages if m.role == "system") or None
        contents = _gemini_contents(messages, images)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=response_model,  # schema-constrained
        )
        start = time.perf_counter()

        try:
            resp = self._client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except httpx.TimeoutException as e:
            raise ProviderTimeout(str(e)) from e  # client-side timeout (retryable)
        except genai_errors.ClientError as e:
            raise (
                ProviderRateLimit(str(e))
                if getattr(e, "code", None) == 429
                else ProviderError(str(e))
            ) from e
        except genai_errors.ServerError as e:
            raise ProviderTimeout(
                str(e)
            ) from e  # treat transient server errors as retryable
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
