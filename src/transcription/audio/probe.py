"""Inspect a source file with ffprobe. Never trust the extension, ask the decoder."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..config import Settings
from ..models import AudioMetadata


def probe(path: Path, settings: Settings) -> AudioMetadata:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError("audio file is empty")
    if shutil.which(settings.ffprobe_bin) is None:
        raise RuntimeError(f"{settings.ffprobe_bin} not found on PATH")

    cmd = [
        settings.ffprobe_bin,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "a",          # audio streams only
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f"ffprobe could not read the file: {proc.stderr.strip()}")

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        raise ValueError("no audio stream found in file")

    stream = streams[0]
    fmt = data.get("format", {})

    # Duration may sit on the stream or the container. Prefer whichever is present.
    duration = _first_float(stream.get("duration"), fmt.get("duration"))
    if duration is None:
        raise ValueError("could not determine audio duration")

    return AudioMetadata(
        duration=duration,
        sample_rate=int(stream.get("sample_rate", 0)),
        channels=int(stream.get("channels", 0)),
        codec=stream.get("codec_name", "unknown"),
        container=fmt.get("format_name", "unknown"),
    )


def _first_float(*values: object) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None