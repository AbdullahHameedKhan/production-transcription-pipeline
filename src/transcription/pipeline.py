"""End to end orchestration: probe, normalize, chunk, transcribe, stitch.

Expects from the audio package (built next):
    probe(path: Path, settings) -> AudioMetadata
    normalize(src: Path, dst: Path, settings) -> Path        # writes 16 kHz mono wav
    chunk(wav: Path, meta: AudioMetadata, settings, workdir: Path) -> list[Chunk]
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .asr import ASREngine
from .audio.chunk import chunk
from .audio.normalize import normalize
from .audio.probe import probe
from .config import Settings
from .models import Transcript, TranscribedChunk, TranscriptionResult
from .stitch import stitch


def transcribe_file(
    path: Path,
    *,
    settings: Settings,
    engine: ASREngine,
    cleanup: bool = True,
) -> TranscriptionResult:
    path = Path(path)
    settings.workdir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=settings.workdir))

    try:
        meta = probe(path, settings)
        normalized = normalize(path, work / "normalized.wav", settings)
        chunks = chunk(normalized, meta, settings, work)

        language = settings.language
        transcribed: list[TranscribedChunk] = []
        for c in chunks:
            result = engine.transcribe(c.path, language=language)
            # Pin the language after the first chunk so later chunks stay consistent.
            if language is None:
                language = result.language
            transcribed.append(TranscribedChunk(chunk=c, segments=result.segments))

        segments = stitch(transcribed)
        transcript = Transcript(segments=segments, language=language, duration=meta.duration)
        return TranscriptionResult(transcript=transcript, audio=meta)
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)