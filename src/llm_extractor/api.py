from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from langfuse import Langfuse, observe
from pydantic import BaseModel

from .config import settings
from .extraction import ExtractionResult, extract

langfuse = Langfuse(  # initializes + registers the client
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    langfuse.flush()  # make sure traces are sent on shutdown


app = FastAPI(title="LLM Extractor v0", lifespan=lifespan)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    # simple shared-key auth (the public URL needs *some* protection)
    if not settings.app_api_key or x_api_key != settings.app_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class ExtractRequest(BaseModel):
    text: str
    provider: str | None = None  # optionally override the default per request


class ExtractResponse(BaseModel):
    result: ExtractionResult
    provider: str
    cost_usd: float
    latency_ms: float


@app.get("/healthz")  # Cloud Run / uptime checks hit this
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@observe(name="extract")  # wraps the call in a Langfuse trace
def _run(text: str, provider: str | None) -> ExtractResponse:
    parsed = extract(text, provider)
    # Because every provider returns the same normalized shape, we log one uniform span:
    with langfuse.start_as_current_observation(
        name=f"{parsed.provider}-extract",
        as_type="generation",
        model=parsed.model,
        input=text,
    ) as gen:
        gen.update(
            output=parsed.data.model_dump(),
            usage_details={
                "input": parsed.usage.input_tokens,
                "output": parsed.usage.output_tokens,
            },
            metadata={"cost_usd": parsed.cost_usd, "latency_ms": parsed.latency_ms},
        )
    return ExtractResponse(
        result=parsed.data,
        provider=parsed.provider,
        cost_usd=parsed.cost_usd,
        latency_ms=parsed.latency_ms,
    )


@app.post(
    "/extract", response_model=ExtractResponse, dependencies=[Depends(require_api_key)]
)
def extract_endpoint(req: ExtractRequest) -> ExtractResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    return _run(req.text, req.provider)
