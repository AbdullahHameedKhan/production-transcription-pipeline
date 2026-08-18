"""Domain types shared across the pipeline. No IO, no framework, pure data."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Segment:
    """One transcribed span. Times are seconds relative to whatever audio produced it."""
    start: float
    end: float
    text: str
    id: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def shifted(self, offset: float) -> "Segment":
        """Return a copy moved along the timeline. Used to place chunk times on the source timeline."""
        return Segment(start=self.start + offset, end=self.end + offset, text=self.text, id=self.id)


@dataclass(frozen=True, slots=True)
class ASRResult:
    """What an engine returns for one piece of audio."""
    segments: list[Segment]
    language: str | None = None
    language_probability: float | None = None


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Ground truth about the source file, from ffprobe."""
    duration: float
    sample_rate: int
    channels: int
    codec: str
    container: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A planned window of the source, materialized as its own wav file."""
    index: int
    start: float          # absolute seconds in the source
    end: float            # absolute seconds in the source
    path: Path

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class TranscribedChunk:
    """A chunk paired with its segments, whose times are relative to the chunk start."""
    chunk: Chunk
    segments: list[Segment]


@dataclass(frozen=True, slots=True)
class Transcript:
    segments: list[Segment]
    language: str | None = None
    duration: float | None = None

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    transcript: Transcript
    audio: AudioMetadata