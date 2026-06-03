from __future__ import annotations

import sys

from llm_extractor.models import LLMMessage
from llm_extractor.providers.factory import get_provider

prompt = " ".join(sys.argv[1:]) or "Explain RAG in two sentences."
messages = [
    LLMMessage(role="system", content="You are concise and precise."),
    LLMMessage(role="user", content=prompt),
]

# Run the SAME prompt through both providers and compare cost/latency/output:
for name in ("openai", "gemini"):
    r = get_provider(name).complete(messages, temperature=0.2, max_tokens=200)
    print(f"\n[{r.provider} · {r.model}]\n{r.text.strip()}")
    print(
        f"  tokens={r.usage.total_tokens} cost=${r.cost_usd:.6f} latency={r.latency_ms:.0f}ms"
    )
