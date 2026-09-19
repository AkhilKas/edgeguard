"""
Central configuration for EdgeGuard.
All tunable values live here — nothing is hardcoded elsewhere.
Reads from environment variables or a .env file.
"""

from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GuardrailMode(str, Enum):
    HEURISTIC = "heuristic"  # fast rule-based baseline (no model needed)
    ML = "ml"  # distilled ML classifier (Week 4+)


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    model_path: Path = Field(
        default=Path("models/qwen3.5-0.8b-q4_k_m.gguf"),
        description="Path to the GGUF model file",
    )
    context_window: int = Field(default=2048, ge=512, le=8192)
    max_tokens: int = Field(default=512, ge=64, le=2048)
    n_threads: int = Field(default=4, ge=1, le=8)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stop_tokens: list[str] = Field(default=["</s>", "<|endoftext|>"])

    @field_validator("model_path")
    @classmethod
    def model_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(
                f"Model file not found at {v}. Run: bash scripts/download_model.sh"
            )
        return v


class GuardrailSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GUARDRAIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    mode: GuardrailMode = Field(
        default=GuardrailMode.HEURISTIC,
        description="Switch to 'ml' after Week 4 distillation",
    )
    model_path: Path | None = Field(
        default=None,
        description="Path to distilled classifier (required when mode=ml)",
    )
    confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Injection probability above this triggers a block",
    )
    enabled: bool = Field(default=True)


class MCUSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MCU_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    port: str = Field(default="/dev/ttyACM0", description="UART serial port")
    baud_rate: int = Field(default=115200)
    timeout: float = Field(default=2.0, description="Serial read timeout in seconds")
    enabled: bool = Field(
        default=True,
        description="Set False in CI/CD or local dev without hardware",
    )


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    reload: bool = Field(default=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    log_level: LogLevel = Field(default=LogLevel.INFO)
    deployment_context: str = Field(
        default="home_assistant",
        description="One of: home_assistant | security_log | iot_controller",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    guardrail: GuardrailSettings = Field(default_factory=GuardrailSettings)
    mcu: MCUSettings = Field(default_factory=MCUSettings)
    api: APISettings = Field(default_factory=APISettings)


def get_settings() -> Settings:
    """Factory used for dependency injection throughout the app."""
    return Settings()
