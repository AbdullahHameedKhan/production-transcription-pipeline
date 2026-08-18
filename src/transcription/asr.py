"""ASR abstraction. The pipeline knows the protocol, not the vendor."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .config import Settings
from .models import ASRResult, Segment


@runtime_checkable
class ASREngine(Protocol):
    def transcribe(self, audio_path: Path, *, language: str | None = None) -> ASRResult: ...


class FasterWhisperEngine:
    """Local default. Import is deferred so nothing else pays for the model dependency."""

    def __init__(
        self,
        *,
        model_size: str,
        device: str,
        compute_type: str,
        beam_size: int,
        vad_filter: bool,
    ) -> None:
        from faster_whisper import WhisperModel  # deferred on purpose

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._beam_size = beam_size
        self._vad_filter = vad_filter

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> ASRResult:
        seg_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
        )
        segments = [
            Segment(start=s.start, end=s.end, text=s.text.strip(), id=i)
            for i, s in enumerate(seg_iter)
        ]
        return ASRResult(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
        )


def build_engine(settings: Settings) -> ASREngine:
    """Resolve an engine from config. Adding a vendor is a new branch, no pipeline change."""
    if settings.asr_engine == "faster-whisper":
        return FasterWhisperEngine(
            model_size=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
        )
    raise ValueError(f"unknown asr engine: {settings.asr_engine!r}")