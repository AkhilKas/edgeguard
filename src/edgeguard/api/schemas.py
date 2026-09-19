"""API data contracts.

All request and response shapes live here.
No business logic — pure data definitions.
"""

from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    context: str | None = Field(
        default=None,
        description="Optional deployment context override",
    )


class PipelineResponse(BaseModel):
    verdict: str
    confidence: float
    matched_pattern: str | None
    llm_response: str | None
    guardrail_latency_ms: float
    llm_latency_ms: float | None
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    llm_loaded: bool
    guardrail_mode: str
    guardrail_enabled: bool
    deployment_context: str
