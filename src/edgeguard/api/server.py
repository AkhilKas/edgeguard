"""FastAPI application entry point.

Wires together: settings → LLM → classifier → pipeline → MCU → HTTP routes.
Uses FastAPI lifespan for clean startup/shutdown of hardware resources.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status

from edgeguard.api.schemas import HealthResponse, PipelineResponse, PromptRequest
from edgeguard.config.settings import Settings, get_settings
from edgeguard.guardrail.factory import build_classifier
from edgeguard.llm.engine import LlamaCppEngine
from edgeguard.mcu.interface import MCUInterface
from edgeguard.pipeline.pipeline import EdgeGuardPipeline

log = structlog.get_logger(__name__)

# Module-level singletons initialised in lifespan
_pipeline: EdgeGuardPipeline | None = None
_mcu: MCUInterface | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    global _pipeline, _mcu
    settings: Settings = get_settings()

    log.info("startup.begin", context=settings.deployment_context)

    # LLM
    engine = LlamaCppEngine(settings.llm)
    engine.load()

    # Guardrail
    classifier = build_classifier(settings.guardrail)

    # Pipeline
    _pipeline = EdgeGuardPipeline(
        classifier=classifier,
        llm=engine,
        guardrail_enabled=settings.guardrail.enabled,
    )

    # MCU (skipped if disabled, e.g. in CI)
    _mcu = MCUInterface(settings.mcu)
    _mcu.on_prompt(lambda text: _run_and_notify(text))
    _mcu.connect()

    log.info("startup.done")
    yield

    # Shutdown
    _mcu.disconnect()
    engine.unload()
    log.info("shutdown.done")


def _run_and_notify(prompt: str) -> None:
    """Called by the MCU reader thread on incoming prompts."""
    if _pipeline is None or _mcu is None:
        return
    result = _pipeline.run(prompt)
    _mcu.send_status(result.verdict.value)


app = FastAPI(
    title="EdgeGuard",
    description="On-device prompt injection detection for edge LLMs",
    version="0.1.0",
    lifespan=lifespan,
)


def get_pipeline() -> EdgeGuardPipeline:
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialised",
        )
    return _pipeline


def get_settings_dep() -> Settings:
    return get_settings()


@app.post("/prompt", response_model=PipelineResponse)
def run_prompt(
    req: PromptRequest,
    pipeline: EdgeGuardPipeline = Depends(get_pipeline),
) -> PipelineResponse:
    """Submit a prompt through the full guardrail → LLM pipeline."""
    result = pipeline.run(req.text)
    return PipelineResponse(
        verdict=result.verdict.value,
        confidence=result.confidence,
        matched_pattern=result.matched_pattern,
        llm_response=result.llm_response,
        guardrail_latency_ms=result.guardrail_latency_ms,
        llm_latency_ms=result.llm_latency_ms,
        total_latency_ms=result.total_latency_ms,
    )


@app.get("/health", response_model=HealthResponse)
def health(
    pipeline: EdgeGuardPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings_dep),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_loaded=pipeline._llm.is_loaded(),
        guardrail_mode=settings.guardrail.mode.value,
        guardrail_enabled=settings.guardrail.enabled,
        deployment_context=settings.deployment_context,
    )


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "edgeguard.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.log_level.value.lower(),
    )


if __name__ == "__main__":
    main()
