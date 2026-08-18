FROM python:3.11-slim

# ffmpeg and ffprobe are system dependencies, not pip packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir -e ".[whisper,api]"

ENV TRANSCRIPTION_WORKDIR=/tmp/transcription
EXPOSE 8000

CMD ["uvicorn", "transcription.api.main:app", "--host", "0.0.0.0", "--port", "8000"]