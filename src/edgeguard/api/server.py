"""FastAPI application entry point.

Wires together: settings → LLM → classifier → pipeline → MCU → HTTP routes.
Optional Whisper STT powers /voice for the dashboard mic button.
Uses FastAPI lifespan for clean startup/shutdown of hardware resources.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse

from edgeguard.api.schemas import (
    HealthResponse,
    PhotoInfo,
    PhotoListResponse,
    PipelineResponse,
    PromptRequest,
    VoiceResponse,
)
from edgeguard.config.settings import Settings, get_settings
from edgeguard.guardrail.factory import build_classifier
from edgeguard.llm.engine import LlamaCppEngine
from edgeguard.mcu.interface import MCUInterface
from edgeguard.pipeline.pipeline import EdgeGuardPipeline, PipelineResult
from edgeguard.stt.whisper_engine import WhisperSTT

log = structlog.get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# Module-level singletons initialised in lifespan
_pipeline: EdgeGuardPipeline | None = None
_mcu: MCUInterface | None = None
_stt: WhisperSTT | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    global _pipeline, _mcu, _stt
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
        camera_warmup_seconds=settings.actions.camera_warmup_seconds,
        camera_discard_frames=settings.actions.camera_discard_frames,
    )

    # MCU (skipped if disabled, e.g. in CI)
    _mcu = MCUInterface(settings.mcu)
    _mcu.on_prompt(lambda text: _run_and_notify(text))
    _mcu.connect()

    # Optional Whisper STT for /voice
    _stt = None
    if settings.stt.enabled:
        _stt = WhisperSTT(settings.stt)
        _stt.load()
    else:
        log.info("stt.disabled", reason="STT_ENABLED=false")

    log.info("startup.done")
    yield

    # Shutdown
    if _stt is not None:
        _stt.unload()
    _mcu.disconnect()
    engine.unload()
    log.info("shutdown.done")


def _run_and_notify(prompt: str) -> None:
    """Called by the MCU reader thread on incoming prompts."""
    if _pipeline is None or _mcu is None:
        return
    result = _pipeline.run(prompt)
    _mcu.send_status(result.verdict.value)


def _photo_url_for(path: Path | None, photo_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        resolved = path.resolve()
        if resolved.parent.resolve() != photo_dir.resolve():
            return None
        if resolved.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            return None
        return f"/photos/{resolved.name}"
    except OSError:
        return None


def _pipeline_response(
    result: PipelineResult, photo_dir: Path | None = None
) -> PipelineResponse:
    directory = photo_dir
    if directory is None and _pipeline is not None:
        directory = _pipeline._photo_dir
    photo_url = (
        _photo_url_for(result.photo_path, directory) if directory is not None else None
    )
    return PipelineResponse(
        verdict=result.verdict.value,
        confidence=result.confidence,
        matched_pattern=result.matched_pattern,
        llm_response=result.llm_response,
        guardrail_latency_ms=result.guardrail_latency_ms,
        llm_latency_ms=result.llm_latency_ms,
        total_latency_ms=result.total_latency_ms,
        photo_url=photo_url,
    )


def _safe_photo_path(photo_dir: Path, name: str) -> Path:
    """Resolve a photo filename under photo_dir; reject path traversal."""
    if not name or name != Path(name).name:
        raise HTTPException(status_code=400, detail="Invalid photo name")
    if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Unsupported photo type")
    path = (photo_dir / name).resolve()
    if path.parent != photo_dir.resolve():
        raise HTTPException(status_code=400, detail="Invalid photo name")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Photo not found")
    return path


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


def get_stt() -> WhisperSTT:
    if _stt is None or not _stt.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Speech-to-text is not available. Set STT_ENABLED=true in .env "
                'and install with: pip install -e ".[stt]"'
            ),
        )
    return _stt


def get_settings_dep() -> Settings:
    return get_settings()


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Serve the browser dashboard for prompting the pipeline manually."""
    return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/prompt", response_model=PipelineResponse)
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
    return _pipeline_response(result)


def _voice_from_audio(
    raw: bytes,
    filename: str,
    pipeline: EdgeGuardPipeline,
    stt: WhisperSTT,
) -> VoiceResponse:
    try:
        heard, stt_ms = stt.transcribe(raw, filename=filename)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except Exception as e:
        log.exception("stt.failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {e}",
        ) from e

    if not heard:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="I didn't catch any speech — try again a bit closer to the mic.",
        )

    result = pipeline.run(heard)
    if _mcu is not None:
        _mcu.send_status(result.verdict.value)

    pipe = _pipeline_response(result)
    return VoiceResponse(
        heard=heard,
        stt_latency_ms=stt_ms,
        verdict=pipe.verdict,
        confidence=pipe.confidence,
        matched_pattern=pipe.matched_pattern,
        llm_response=pipe.llm_response,
        guardrail_latency_ms=pipe.guardrail_latency_ms,
        llm_latency_ms=pipe.llm_latency_ms,
        total_latency_ms=pipe.total_latency_ms,
        photo_url=pipe.photo_url,
    )


@app.post("/voice", response_model=VoiceResponse)
async def run_voice(
    audio: UploadFile = File(..., description="Short recorded clip (webm/wav/ogg)"),
    pipeline: EdgeGuardPipeline = Depends(get_pipeline),
    stt: WhisperSTT = Depends(get_stt),
) -> VoiceResponse:
    """Transcribe a browser-uploaded clip with Whisper, then run the pipeline.

    Needs a secure context (https or localhost) for the browser mic. Prefer
    ``POST /voice/device`` when opening the dashboard over plain LAN HTTP.
    """
    raw = await audio.read()
    filename = audio.filename or "audio.webm"
    return _voice_from_audio(raw, filename, pipeline, stt)


@app.post("/voice/device", response_model=VoiceResponse)
def run_voice_device(
    pipeline: EdgeGuardPipeline = Depends(get_pipeline),
    stt: WhisperSTT = Depends(get_stt),
    settings: Settings = Depends(get_settings_dep),
) -> VoiceResponse:
    """Record from the board's ALSA mic (Brio), transcribe, run the pipeline.

    Works over plain ``http://<board-ip>:8000/`` — no browser microphone /
    HTTPS / SSH tunnel required.
    """
    from edgeguard.stt.device_record import record_wav

    try:
        raw = record_wav(
            seconds=settings.stt.record_seconds,
            alsa_device=settings.stt.alsa_device,
            sample_rate=settings.stt.sample_rate,
        )
    except Exception as e:
        log.exception("stt.device_record.failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Board mic record failed: {e}",
        ) from e

    return _voice_from_audio(raw, "device.wav", pipeline, stt)


@app.get("/photos", response_model=PhotoListResponse)
def list_photos(
    settings: Settings = Depends(get_settings_dep),
    limit: int = 24,
) -> PhotoListResponse:
    """List recent JPEGs in ACTIONS_PHOTO_DIR for the dashboard gallery."""
    photo_dir = settings.actions.photo_dir
    if not photo_dir.exists():
        return PhotoListResponse(photos=[])
    limit = max(1, min(limit, 100))
    files = sorted(
        [
            p
            for p in photo_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    photos = [
        PhotoInfo(
            name=p.name,
            url=f"/photos/{p.name}",
            size_bytes=p.stat().st_size,
            modified_at=p.stat().st_mtime,
        )
        for p in files
    ]
    return PhotoListResponse(photos=photos)


@app.get("/photos/{name}")
def get_photo(
    name: str,
    settings: Settings = Depends(get_settings_dep),
) -> FileResponse:
    """Serve one captured frame so the laptop browser can view it (no scp)."""
    path = _safe_photo_path(settings.actions.photo_dir, name)
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media, filename=path.name)


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
        stt_enabled=settings.stt.enabled,
        stt_loaded=_stt is not None and _stt.is_loaded,
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
