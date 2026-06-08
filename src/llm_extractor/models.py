from __future__ import annotations

from typing import Any, Literal, Generic, TypeVar

from pydantic import BaseModel, Field, computed_field

Role = Literal["system", "user", "assistant"]
T = TypeVar("T", bound=BaseModel)


class LLMMessage(BaseModel):
    role: Role
    content: str


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @computed_field  # exposes total_tokens as if it were a field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """The normalized result EVERY provider returns — so the rest of the app never
    has to care which provider was used."""

    text: str
    provider: str
    model: str
    usage: TokenUsage
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str | None = None
    raw: dict[str, Any] | None = Field(
        default=None, repr=False
    )  # full payload for debugging


class ParsedResponse(BaseModel, Generic[T]):
    data: T  # the validated, structured object
    provider: str
    model: str
    usage: TokenUsage
    cost_usd: float = 0.0
    latency_ms: float = 0.0
