"""HTTP surface. Routing and translation between the wire format and the pipeline."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import PlainTextResponse

from ..config import Settings, get_settings
from ..exporters import EXPORTERS
from ..models import TranscriptionResult
from .jobs import Job, JobStore, TranscriptionWorker, new_job_id
from .schemas import (
    AudioOut,
    Health,
    JobCreated,
    JobState,
    JobStatus,
    SegmentOut,
    TranscriptOut,
)

router = APIRouter()

ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".webm", ".aac", ".mp4"}


def get_store(request: Request) -> JobStore:
    return request.app.state.store


def get_worker(request: Request) -> TranscriptionWorker:
    return request.app.state.worker


@router.get("/health", response_model=Health)
async def health(
    store: JobStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> Health:
    return Health(
        status="ok",
        engine=settings.asr_engine,
        model=settings.whisper_model,
        active_jobs=await store.count_active(),
    )


@router.post(
    "/transcriptions",
    response_model=JobCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_transcription(
    background: BackgroundTasks,
    file: UploadFile,
    store: JobStore = Depends(get_store),
    worker: TranscriptionWorker = Depends(get_worker),
    settings: Settings = Depends(get_settings),
) -> JobCreated:
    """Accept audio and return immediately. Transcription can take minutes, so
    holding the connection open would be wrong at any real scale."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"unsupported extension {suffix!r}; ffprobe still validates content",
        )

    job_id = new_job_id()
    uploads = settings.workdir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / f"{job_id}{suffix}"

    # Stream to disk. Never read an unbounded upload into memory, and stop the
    # moment it exceeds the cap rather than after paying for the whole transfer.
    written = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"upload exceeds {settings.max_upload_bytes} bytes",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="empty upload")

    job = Job(id=job_id, filename=file.filename or dest.name, source=dest)
    await store.create(job)
    background.add_task(worker.run, job)

    return JobCreated(
        job_id=job_id,
        status=JobStatus.queued,
        status_url=f"/v1/transcriptions/{job_id}",
    )


@router.get("/transcriptions/{job_id}", response_model=JobState)
async def get_job(job_id: str, store: JobStore = Depends(get_store)) -> JobState:
    job = await _require(store, job_id)
    return JobState(
        job_id=job.id,
        status=job.status,
        filename=job.filename,
        attempts=job.attempts,
        error=job.error,
        duration=job.result.audio.duration if job.result else None,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/transcriptions/{job_id}/result")
async def get_result(
    job_id: str,
    format: str = "json",
    store: JobStore = Depends(get_store),
):
    """JSON by default. srt, vtt and text come back as plain text so a browser
    or curl can consume them directly."""
    if format not in EXPORTERS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"format must be one of {', '.join(EXPORTERS)}",
        )

    job = await _require(store, job_id)
    if job.status is JobStatus.failed:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=job.error or "transcription failed")
    if job.status is not JobStatus.completed or job.result is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"job is {job.status.value}; poll {job_id} until completed",
        )

    if format == "json":
        return _to_transcript_out(job.id, job.result)
    return PlainTextResponse(EXPORTERS[format](job.result.transcript))


async def _require(store: JobStore, job_id: str) -> Job:
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="job not found")
    return job


def _to_transcript_out(job_id: str, result: TranscriptionResult) -> TranscriptOut:
    t = result.transcript
    return TranscriptOut(
        job_id=job_id,
        language=t.language,
        duration=t.duration,
        text=t.full_text,
        segments=[
            SegmentOut(id=s.id, start=round(s.start, 3), end=round(s.end, 3), text=s.text)
            for s in t.segments
        ],
        audio=AudioOut(
            duration=result.audio.duration,
            sample_rate=result.audio.sample_rate,
            channels=result.audio.channels,
            codec=result.audio.codec,
            container=result.audio.container,
        ),
    )