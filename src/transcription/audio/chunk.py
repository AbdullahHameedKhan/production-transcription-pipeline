"""Split long audio into overlapping windows so no ASR pass sees an unbounded file.

Overlap means words near a seam appear in two chunks. stitch.py resolves the duplicate.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import Settings
from ..models import AudioMetadata, Chunk


def chunk(wav: Path, meta: AudioMetadata, settings: Settings, workdir: Path) -> list[Chunk]:
    wav, workdir = Path(wav), Path(workdir)
    length = settings.chunk_length_s
    overlap = settings.chunk_overlap_s
    if overlap >= length:
        raise ValueError("chunk_overlap_s must be smaller than chunk_length_s")

    windows = plan_windows(meta.duration, length, overlap)

    # Short file: one window covering everything. No cutting, reuse the normalized wav.
    if len(windows) == 1:
        start, end = windows[0]
        return [Chunk(index=0, start=start, end=end, path=wav)]

    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(windows):
        out = workdir / f"chunk_{i:04d}.wav"
        _cut(wav, out, start, end - start, settings)
        chunks.append(Chunk(index=i, start=start, end=end, path=out))
    return chunks


def plan_windows(duration: float, length: float, overlap: float) -> list[tuple[float, float]]:
    """Absolute (start, end) windows over the source. Pure, deterministic, unit tested."""
    if duration <= length:
        return [(0.0, duration)]
    step = length - overlap 
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + length, duration)
        windows.append((start, end))
        if end >= duration:                # last window reached the tail, stop
            break
        start += step
    return windows


def _cut(src: Path, dst: Path, start: float, duration: float, settings: Settings) -> None:
    cmd = [
        settings.ffmpeg_bin,
        "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}",             # seek before input: fast and sample accurate on PCM wav
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-ac", str(settings.target_channels),
        "-ar", str(settings.target_sample_rate),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to cut chunk: {proc.stderr.strip()}")