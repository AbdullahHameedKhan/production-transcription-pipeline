"""Job lifecycle: store, bounded worker pool, retry policy.

The store sits behind a protocol. The in-memory implementation is correct for a
single process and is deliberately the only one built here. Production swaps in
Redis or Postgres by implementing the same four methods; nothing else changes.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..asr import ASREngine
from ..config import Settings
from ..models import TranscriptionResult
from ..pipeline import transcribe_file
from .schemas import JobStatus

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    filename: str
    source: Path
    status: JobStatus = JobStatus.queued
    attempts: int = 0
    error: str | None = None
    result: TranscriptionResult | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobStore(Protocol):
    async def create(self, job: Job) -> None: ...
    async def get(self, job_id: str) -> Job | None: ...
    async def update(self, job: Job) -> None: ...
    async def count_active(self) -> int: ...


class InMemoryJobStore:
    """Single process store. Jobs die with the process, which is the documented
    limitation of this take home. See README for the durable design."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.id] = job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job: Job) -> None:
        async with self._lock:
            job.updated_at = time.time()
            self._jobs[job.id] = job

    async def count_active(self) -> int:
        async with self._lock:
            return sum(
                1 for j in self._jobs.values()
                if j.status in (JobStatus.queued, JobStatus.running)
            )


class TranscriptionWorker:
    """Runs jobs off the event loop with a hard concurrency ceiling and retries.

    Concurrency: uploads are accepted immediately and always. The semaphore caps
    how many actually decode and transcribe at once, so N concurrent uploads
    never become N concurrent ffmpeg plus model processes.

    Retries: transient failures (ffmpeg died, model hiccup) get retried with
    backoff. Bad input is not retried, since replaying it produces the same error.
    """

    def __init__(
        self,
        *,
        store: JobStore,
        engine: ASREngine,
        settings: Settings,
        max_concurrency: int = 2,
        max_attempts: int = 3,
    ) -> None:
        self._store = store
        self._engine = engine
        self._settings = settings
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max_attempts = max_attempts

    async def run(self, job: Job) -> None:
        async with self._sem:
            job.status = JobStatus.running
            await self._store.update(job)

            for attempt in range(1, self._max_attempts + 1):
                job.attempts = attempt
                try:
                    # Transcription is blocking CPU work. Off the loop it goes,
                    # otherwise one job freezes every other request.
                    result = await asyncio.to_thread(
                        transcribe_file,
                        job.source,
                        settings=self._settings,
                        engine=self._engine,
                    )
                    job.result = result
                    job.status = JobStatus.completed
                    job.error = None
                    await self._store.update(job)
                    self._discard_source(job)
                    return

                except ValueError as exc:
                    # Bad input. Deterministic, so retrying is pointless.
                    log.warning("job %s rejected: %s", job.id, exc)
                    job.status = JobStatus.failed
                    job.error = str(exc)
                    await self._store.update(job)
                    self._discard_source(job)
                    return

                except Exception as exc:  # noqa: BLE001 transient, worth another try
                    log.warning("job %s attempt %d failed: %s", job.id, attempt, exc)
                    job.error = str(exc)
                    await self._store.update(job)
                    if attempt == self._max_attempts:
                        job.status = JobStatus.failed
                        await self._store.update(job)
                        self._discard_source(job)
                        return
                    await asyncio.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s

    @staticmethod
    def _discard_source(job: Job) -> None:
        """Uploads are scratch. Durable storage is the README's answer, not this."""
        try:
            job.source.unlink(missing_ok=True)
        except OSError:
            log.warning("could not remove upload for job %s", job.id)


def new_job_id() -> str:
    return uuid.uuid4().hex