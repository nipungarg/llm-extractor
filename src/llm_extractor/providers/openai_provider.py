from __future__ import annotations

import time

import base64

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from ..models import LLMResponse, TokenUsage
from ..pricing import compute_cost
from .base import LLMProvider

from ..errors import (
    ProviderError,
    ProviderRateLimit,
    ProviderRefusal,
    ProviderTimeout,
    retry_transient,
)


def _guess_mime(data: bytes) -> str:
    # detect format from the file's magic bytes (we only need jpeg vs png)
    return "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"


def _openai_messages(messages, images):
    """Turn our messages into OpenAI's format, attaching images to the first user turn."""
    out, attached = [], False
    for m in messages:
        if m.role == "user" and images and not attached:
            # multimodal user turn = a list of text + image parts
            content = [{"type": "text", "text": m.content}]
            for img in images:
                b64 = base64.b64encode(img).decode()
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{_guess_mime(img)};base64,{b64}"},
                    }
                )
            out.append({"role": "user", "content": content})
            attached = True
        else:
            out.append(m.model_dump())
    return out


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gpt-4o-mini",
        timeout: float = 30.0,
    ) -> None:
        # timeout: don't let a single call hang forever
        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self.default_model = default_model

    @retry_transient  # retry only on timeouts / rate limits
    def complete(
        self, messages, *, model=None, temperature=0.7, max_tokens=1024, images=None
    ) -> LLMResponse:
        model = model or self.default_model
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=_openai_messages(messages, images),
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        # Map the SDK's exceptions onto OUR typed errors:
        except APITimeoutError as e:
            raise ProviderTimeout(str(e)) from e
        except RateLimitError as e:
            raise ProviderRateLimit(str(e)) from e
        except APIError as e:
            raise ProviderError(str(e)) from e
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

    @retry_transient
    def parse(
        self,
        messages,
        response_model,
        *,
        model=None,
        temperature=0.0,
        max_tokens=1024,
        images=None,
    ):
        import time
        from ..models import ParsedResponse, TokenUsage
        from ..pricing import compute_cost

        model = model or self.default_model
        start = time.perf_counter()
        # .parse() forces the model's output to match the Pydantic schema (was .beta. in old SDKs)
        try:
            completion = self._client.chat.completions.parse(
                model=model,
                messages=_openai_messages(messages, images),
                temperature=temperature,
                max_completion_tokens=max_tokens,
                response_format=response_model,
            )
        except APITimeoutError as e:
            raise ProviderTimeout(str(e)) from e
        except RateLimitError as e:
            raise ProviderRateLimit(str(e)) from e
        except APIError as e:
            raise ProviderError(str(e)) from e
        latency_ms = (time.perf_counter() - start) * 1000
        msg = completion.choices[0].message
        if msg.refusal:  # a refusal is a provider failure, so it can trigger fallback
            raise ProviderRefusal(f"Model refused: {msg.refusal}")
        usage = TokenUsage(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )
        return ParsedResponse(
            data=msg.parsed,
            provider=self.provider_name,
            model=model,
            usage=usage,
            cost_usd=compute_cost(model, usage),
            latency_ms=latency_ms,
        )
