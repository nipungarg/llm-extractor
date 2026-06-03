from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..models import LLMMessage, LLMResponse


class LLMProvider(ABC):
    """Every provider implements complete() and returns a normalized LLMResponse."""

    provider_name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...
