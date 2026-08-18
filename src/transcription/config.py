"""Central configuration. Every tunable lives here, nothing hardcoded downstream."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIPTION_", env_file=".env", extra="ignore"
    )

    # ASR engine
    asr_engine: str = "faster-whisper"
    whisper_model: str = "base"          # tiny | base | small | medium | large-v3
    whisper_device: str = "cpu"          # cpu | cuda
    whisper_compute_type: str = "int8"   # int8 | float16 | float32
    beam_size: int = 5
    vad_filter: bool = True
    language: str | None = None          # None means auto detect

    # Audio normalization target (canonical form the ASR always sees)
    target_sample_rate: int = 16_000
    target_channels: int = 1
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

    # Job execution
    max_concurrent_jobs: int = 2
    max_attempts: int = 3

    # Long file chunking
    chunk_length_s: float = 300.0        # 5 minute windows
    chunk_overlap_s: float = 5.0         # overlap so no word is cut at a seam

    # IO and limits
    workdir: Path = Path("/tmp/transcription")
    max_upload_bytes: int = 200 * 1024 * 1024

    # Observability
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()