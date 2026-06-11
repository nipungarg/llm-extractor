from __future__ import annotations

from pydantic import BaseModel, Field

from .models import LLMMessage, ParsedResponse
from .providers.factory import get_provider


class LineItem(BaseModel):
    name: str
    price: str | None = None


class DocumentExtraction(BaseModel):
    """The fields we want out of a document image (tuned for receipts/invoices)."""

    doc_type: str = Field(description="receipt, invoice, id_card, or unknown")
    merchant: str | None = None
    date: str | None = None
    total: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)


def extract_document(
    image_bytes: bytes, provider_name: str | None = None
) -> ParsedResponse[DocumentExtraction]:
    """Extract structured fields from a document image, via the provider seam."""
    messages = [
        LLMMessage(
            role="system",
            content="You read document images and extract fields. Use only what is visible; "
            "leave fields null if absent.",
        ),
        LLMMessage(
            role="user", content="Extract the key fields from this document image."
        ),
    ]
    # images=[...] is the multimodal part; the seam handles provider differences
    return get_provider(provider_name).parse(
        messages, DocumentExtraction, images=[image_bytes]
    )
