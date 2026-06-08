from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# --- Our own error types (so the app doesn't depend on each SDK's exception classes) ---
class ProviderError(Exception):
    """Base class for any provider failure."""


class ProviderTimeout(ProviderError):
    """The call took too long."""


class ProviderRateLimit(ProviderError):
    """We hit the provider's rate limit (HTTP 429)."""


class ProviderRefusal(ProviderError):
    """The model refused to answer (safety)."""


# Only these are worth retrying — a bad request won't succeed on retry.
TRANSIENT = (ProviderTimeout, ProviderRateLimit)

# Reusable decorator: try up to 3 times, waiting 1s, 2s, 4s... (capped at 10s) between tries.
retry_transient = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(TRANSIENT),
    reraise=True,  # if all retries fail, raise the real error (not tenacity's wrapper)
)
