from __future__ import annotations

from .models import TokenUsage

# USD per 1,000,000 tokens: (input_rate, output_rate). Update with current prices.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}


def compute_cost(model: str, usage: TokenUsage) -> float:
    if model not in PRICING:
        return 0.0
    in_rate, out_rate = PRICING[model]
    return (
        usage.input_tokens / 1_000_000 * in_rate
        + usage.output_tokens / 1_000_000 * out_rate
    )
