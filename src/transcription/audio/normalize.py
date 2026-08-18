"""Erase format variance. Every file becomes 16 kHz mono PCM wav before ASR sees it."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..config import Settings


def normalize(src: Path, dst: Path, settings: Settings) -> Path:
    src, dst = Path(src), Path(dst)
    if shutil.which(settings.ffmpeg_bin) is None:
        raise RuntimeError(f"{settings.ffmpeg_bin} not found on PATH")

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        settings.ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",                                    # overwrite dst
        "-i", str(src),
        "-vn",                                   # drop video and cover art
        "-map", "0:a:0",                         # first audio stream only
        "-ac", str(settings.target_channels),    # downmix to mono
        "-ar", str(settings.target_sample_rate), # resample to 16 kHz
        "-c:a", "pcm_s16le",                     # uncompressed 16 bit PCM
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg failed to normalize audio: {proc.stderr.strip()}")
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError("normalization produced no output")
    return dst