"""App construction. The ASR model is loaded once at startup and shared, since
loading it per request would dominate latency."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..config import get_settings
from .endpoints import router
from .jobs import InMemoryJobStore, TranscriptionWorker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings.workdir.mkdir(parents=True, exist_ok=True)

    engine = app.state.__dict__.get("engine")
    if engine is None:                       # tests inject a fake before startup
        from ..asr import build_engine
        engine = build_engine(settings)

    app.state.engine = engine
    app.state.store = InMemoryJobStore()
    app.state.worker = TranscriptionWorker(
        store=app.state.store,
        engine=engine,
        settings=settings,
        max_concurrency=settings.max_concurrent_jobs,
        max_attempts=settings.max_attempts,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Transcription Pipeline",
        version="0.1.0",
        description="Submit audio, poll for status, fetch the transcript.",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/v1")
    return app


app = create_app()