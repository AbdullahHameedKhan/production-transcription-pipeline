"""Reassemble chunk transcripts into one timeline. Deterministic, testable, no IO."""
from __future__ import annotations

from .models import Segment, TranscribedChunk


def stitch(chunks: list[TranscribedChunk]) -> list[Segment]:
    if not chunks:
        return []

    ordered = sorted(chunks, key=lambda tc: tc.chunk.start)
    n = len(ordered)

    # Move each chunk's segments from chunk local time onto the source timeline.
    absolute: list[list[Segment]] = [
        [s.shifted(tc.chunk.start) for s in tc.segments] for tc in ordered
    ]

    # Seam between chunk i and i+1 is the midpoint of their overlap (or of the gap,
    # if there is one). A segment belongs to whichever side its center falls on.
    left = [float("-inf")] * n
    right = [float("inf")] * n
    for i in range(n - 1):
        seam = (ordered[i].chunk.end + ordered[i + 1].chunk.start) / 2.0
        right[i] = seam
        left[i + 1] = seam

    kept: list[Segment] = []
    for i in range(n):
        lo, hi = left[i], right[i]
        for seg in absolute[i]:
            center = (seg.start + seg.end) / 2.0
            if lo <= center < hi:
                kept.append(seg)

    kept.sort(key=lambda s: s.start)
    return [
        Segment(start=s.start, end=s.end, text=s.text, id=i)
        for i, s in enumerate(kept)
    ]