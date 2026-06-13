from __future__ import annotations

import pytest

from llm_extractor.errors import ProviderError
from llm_extractor.models import LLMMessage, TokenUsage
from llm_extractor.providers.factory import FallbackProvider


class _FakeProvider:
    """A stand-in provider for tests — no network, no cost."""

    def __init__(self, name, fail=False):
        self.provider_name, self._fail = name, fail

    def complete(self, messages, **kw):
        if self._fail:
            raise ProviderError("simulated outage")
        from llm_extractor.models import LLMResponse

        return LLMResponse(
            text="ok",
            provider=self.provider_name,
            model="fake",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def parse(self, messages, response_model, **kw):
        raise NotImplementedError


def test_fallback_skips_failing_provider():
    # primary fails -> should fall back to the working backup
    fb = FallbackProvider(
        [_FakeProvider("primary", fail=True), _FakeProvider("backup")]
    )
    result = fb.complete([LLMMessage(role="user", content="hi")])
    assert result.provider == "backup"


def test_fallback_raises_when_all_fail():
    fb = FallbackProvider(
        [_FakeProvider("a", fail=True), _FakeProvider("b", fail=True)]
    )
    with pytest.raises(ProviderError):
        fb.complete([LLMMessage(role="user", content="hi")])


@pytest.mark.live  # only runs when you ask for it: `uv run pytest -m live`
def test_real_extraction_smoke():
    # a real call — costs a few cents; skipped by default
    from llm_extractor.extraction import extract

    r = extract("Globex paid $5 to Initech on 2024-01-02.")
    assert any(e.type == "MONEY" for e in r.data.entities) or r.data.monetary_amounts
