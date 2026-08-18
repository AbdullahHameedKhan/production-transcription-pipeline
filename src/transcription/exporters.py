"""Serialize a transcript into the formats downstream consumers expect."""
from __future__ import annotations

import json

from .models import Transcript


def _ts(seconds: float, sep: str) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_json(t: Transcript, *, indent: int | None = 2) -> str:
    payload = {
        "language": t.language,
        "duration": t.duration,
        "text": t.full_text,
        "segments": [
            {"id": s.id, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
            for s in t.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def to_srt(t: Transcript) -> str:
    blocks = []
    for i, s in enumerate(t.segments, start=1):
        stamp = f"{_ts(s.start, ',')} --> {_ts(s.end, ',')}"
        blocks.append(f"{i}\n{stamp}\n{s.text}")
    return "\n\n".join(blocks) + "\n"


def to_vtt(t: Transcript) -> str:
    lines = ["WEBVTT", ""]
    for s in t.segments:
        lines.append(f"{_ts(s.start, '.')} --> {_ts(s.end, '.')}")
        lines.append(s.text)
        lines.append("")
    return "\n".join(lines)


def to_text(t: Transcript) -> str:
    return t.full_text + "\n"


EXPORTERS = {"json": to_json, "srt": to_srt, "vtt": to_vtt, "text": to_text}