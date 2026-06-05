from __future__ import annotations

from pydantic import BaseModel, Field

from .models import ParsedResponse
from .prompts import EXTRACT_PROMPT
from .providers.factory import get_provider


class Entity(BaseModel):
    name: str
    type: str = Field(description="One of: PERSON, ORG, DATE, MONEY, LOCATION")


class ExtractionResult(BaseModel):
    summary: str = Field(description="One-sentence summary of the text")
    entities: list[Entity]
    monetary_amounts: list[str] = Field(default_factory=list)


def extract(
    text: str, provider_name: str | None = None
) -> ParsedResponse[ExtractionResult]:
    """Pull structured info out of free text, validated against ExtractionResult."""
    messages = EXTRACT_PROMPT.render(text=text)
    return get_provider(provider_name).parse(
        messages, ExtractionResult, temperature=0.0
    )
