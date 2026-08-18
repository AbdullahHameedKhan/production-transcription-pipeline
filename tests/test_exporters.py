"""Output format contracts: JSON shape and SRT/VTT timestamp correctness."""
from __future__ import annotations

import json

from transcription.exporters import to_json, to_srt, to_text, to_vtt
from transcription.models import Segment, Transcript


def make() -> Transcript:
    return Transcript(
        segments=[Segment(0.0, 1.5, "hello", 0), Segment(1.5, 3.0, "world", 1)],
        language="en",
        duration=3.0,
    )


class TestJson:
    def test_parses_and_has_expected_shape(self):
        data = json.loads(to_json(make()))
        assert data["language"] == "en"
        assert data["duration"] == 3.0
        assert data["text"] == "hello world"
        assert len(data["segments"]) == 2

    def test_segment_fields_and_rounding(self):
        seg = json.loads(to_json(make()))["segments"][0]
        assert seg == {"id": 0, "start": 0.0, "end": 1.5, "text": "hello"}

    def test_unicode_is_preserved(self):
        t = Transcript(segments=[Segment(0.0, 1.0, "مرحبا", 0)], language="ar", duration=1.0)
        data = json.loads(to_json(t))
        assert data["segments"][0]["text"] == "مرحبا"


class TestSrt:
    def test_structure(self):
        out = to_srt(make())
        blocks = out.strip().split("\n\n")
        assert len(blocks) == 2
        first = blocks[0].splitlines()
        assert first[0] == "1"                                   # 1 based index
        assert first[1] == "00:00:00,000 --> 00:00:01,500"       # comma decimal
        assert first[2] == "hello"

    def test_hours_are_formatted(self):
        t = Transcript(segments=[Segment(3661.5, 3662.0, "late", 0)], language="en", duration=3662.0)
        assert to_srt(t).splitlines()[1].startswith("01:01:01,500 -->")

    def test_empty_transcript(self):
        assert to_srt(Transcript(segments=[], language=None, duration=0.0)) == "\n"


class TestVtt:
    def test_header_and_separator(self):
        out = to_vtt(make())
        lines = out.splitlines()
        assert lines[0] == "WEBVTT"
        assert "00:00:00.000 --> 00:00:01.500" in lines      # dot decimal, not comma


class TestText:
    def test_joined_with_trailing_newline(self):
        assert to_text(make()) == "hello world\n"