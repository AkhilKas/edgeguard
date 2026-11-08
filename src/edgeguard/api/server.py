"""FastAPI application entry point.

Wires together: settings → LLM → classifier → pipeline → MCU → HTTP routes.
Uses FastAPI lifespan for clean startup/shutdown of hardware resources.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from edgeguard.api.schemas import (
    HealthResponse,
    MCUStatusResponse,
    MCUToggleRequest,
    PipelineResponse,
    PromptRequest,
)
from edgeguard.config.settings import Settings, get_settings
from edgeguard.guardrail.factory import build_classifier
from edgeguard.llm.engine import LlamaCppEngine
from edgeguard.mcu.interface import MCUInterface
from edgeguard.pipeline.pipeline import EdgeGuardPipeline

log = structlog.get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

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
        photo_dir=settings.actions.photo_dir,
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

def _maybe_add_cors(app: FastAPI, settings: Settings) -> None:
    """Allow cross-origin requests only when explicitly configured.

    Needed once the dashboard is hosted somewhere other than this API
    (API_CORS_ORIGINS unset keeps the default same-origin-only behaviour).
    """
    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )


_maybe_add_cors(app, get_settings())


def get_pipeline() -> EdgeGuardPipeline:
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialised",
        )
    return _pipeline


def get_settings_dep() -> Settings:
    return get_settings()


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings_dep),
) -> None:
    """Gate a route behind API_KEY, when one is configured.

    No-op when API_KEY is unset, so local/LAN use is unaffected — this only
    matters once the API is reachable from outside the local network.
    """
    if settings.api.key and x_api_key != settings.api.key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Serve the browser dashboard for prompting the pipeline manually."""
    return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post(
    "/prompt", response_model=PipelineResponse, dependencies=[Depends(require_api_key)]
)
def run_prompt(
    req: PromptRequest,
    pipeline: EdgeGuardPipeline = Depends(get_pipeline),
) -> PipelineResponse:
    """Submit a prompt through the full guardrail → LLM pipeline.

    Also notifies the MCU of the verdict, same as an MCU-originated prompt,
    so the board's LED responds regardless of where the prompt came from.
    """
    result = pipeline.run(req.text)
    if _mcu is not None:
        _mcu.send_status(result.verdict.value)
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
        mcu_enabled=settings.mcu.enabled,
    )


@app.post(
    "/mcu/toggle",
    response_model=MCUStatusResponse,
    dependencies=[Depends(require_api_key)],
)
def toggle_mcu(req: MCUToggleRequest) -> MCUStatusResponse:
    """Turn the MCU bridge on or off at runtime, from the dashboard toggle."""
    if _mcu is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCU interface not initialised",
        )
    if req.enabled:
        connected = _mcu.enable()
    else:
        _mcu.disable()
        connected = False
    return MCUStatusResponse(enabled=_mcu.enabled, connected=connected)


@app.get(
    "/mcu/status",
    response_model=MCUStatusResponse,
    dependencies=[Depends(require_api_key)],
)
def mcu_status() -> MCUStatusResponse:
    """Report whether the bridge is on, and whether the board is responding."""
    if _mcu is None:
        return MCUStatusResponse(enabled=False, connected=False)
    connected = _mcu.ping() if _mcu.enabled else False
    return MCUStatusResponse(enabled=_mcu.enabled, connected=connected)


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
