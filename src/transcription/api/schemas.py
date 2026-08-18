"""Public API contract. Kept separate from domain models so internal refactors
cannot silently change the wire format."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class SegmentOut(BaseModel):
    id: int
    start: float
    end: float
    text: str


class AudioOut(BaseModel):
    duration: float
    sample_rate: int
    channels: int
    codec: str
    container: str


class JobCreated(BaseModel):
    """202 response. The client polls the status url."""
    job_id: str
    status: JobStatus
    status_url: str


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    attempts: int = 0
    error: str | None = None
    duration: float | None = None
    created_at: float
    updated_at: float


class TranscriptOut(BaseModel):
    job_id: str
    language: str | None
    duration: float | None
    text: str
    segments: list[SegmentOut] = Field(default_factory=list)
    audio: AudioOut


class Health(BaseModel):
    status: str
    engine: str
    model: str
    active_jobs: int