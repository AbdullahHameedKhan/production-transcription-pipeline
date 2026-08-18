"""The reassembly arithmetic: window planning and seam resolution.

No audio, no model. Pure functions over plain numbers. These are the only
pieces of logic that can produce output that looks fine but is silently wrong,
so they get real assertions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from transcription.audio.chunk import plan_windows
from transcription.models import Chunk, Segment, TranscribedChunk
from transcription.stitch import stitch


def tc(index: int, start: float, end: float, segs: list[Segment]) -> TranscribedChunk:
    """Build a TranscribedChunk with a dummy path. Segment times are chunk local."""
    return TranscribedChunk(chunk=Chunk(index, start, end, Path(f"c{index}")), segments=segs)


class TestPlanWindows:
    def test_short_file_is_a_single_window(self):
        assert plan_windows(duration=120.0, length=300.0, overlap=5.0) == [(0.0, 120.0)]

    def test_file_equal_to_window_is_single(self):
        assert plan_windows(300.0, 300.0, 5.0) == [(0.0, 300.0)]

    def test_long_file_known_layout(self):
        # 650s, 300s windows, 5s overlap, step 295.
        assert plan_windows(650.0, 300.0, 5.0) == [(0.0, 300.0), (295.0, 595.0), (590.0, 650.0)]

    def test_first_starts_at_zero_last_ends_at_duration(self):
        w = plan_windows(1000.0, 300.0, 5.0)
        assert w[0][0] == 0.0
        assert w[-1][1] == 1000.0

    def test_consecutive_windows_overlap_never_gap(self):
        w = plan_windows(1000.0, 300.0, 5.0)
        for (_, prev_end), (next_start, _) in zip(w, w[1:]):
            assert next_start < prev_end          # every seam is covered by both sides

    def test_no_window_runs_past_the_end(self):
        w = plan_windows(1000.0, 300.0, 5.0)
        assert all(end <= 1000.0 for _, end in w)

    @pytest.mark.parametrize("duration", [301.0, 600.0, 605.0, 899.0, 12_345.0])
    def test_every_point_is_covered(self, duration: float):
        w = plan_windows(duration, 300.0, 5.0)
        # walk the timeline in 1s steps, each instant must fall inside some window
        t = 0.0
        while t <= duration:
            assert any(s <= t <= e for s, e in w), f"{t} uncovered"
            t += 1.0


class TestStitch:
    def test_empty_input(self):
        assert stitch([]) == []

    def test_single_chunk_passthrough(self):
        out = stitch([tc(0, 0.0, 300.0, [Segment(10.0, 11.0, "solo")])])
        assert [s.text for s in out] == ["solo"]
        assert out[0].start == 10.0

    def test_chunk_local_times_lifted_to_source_timeline(self):
        # segment at 5s inside a chunk that starts at 295s belongs at 300s absolute
        out = stitch([tc(1, 295.0, 595.0, [Segment(5.0, 6.0, "later")])])
        assert out[0].start == 300.0
        assert out[0].end == 301.0

    def test_overlapping_word_kept_exactly_once(self):
        # seam between the two chunks sits at 297.5
        left = tc(0, 0.0, 300.0, [Segment(296.0, 297.0, "hello")])   # abs center 296.5
        right = tc(1, 295.0, 595.0, [Segment(1.0, 2.0, "hello")])    # abs 296..297, center 296.5
        out = stitch([left, right])
        assert [s.text for s in out] == ["hello"]

    def test_word_past_the_seam_goes_to_the_right_chunk(self):
        # a word whose center is after 297.5 must be owned by the right chunk, not dropped
        left = tc(0, 0.0, 300.0, [Segment(299.0, 299.5, "edge")])    # abs center 299.25 > 297.5
        right = tc(1, 295.0, 595.0, [Segment(4.0, 4.5, "edge")])     # abs center 299.25
        out = stitch([left, right])
        assert [s.text for s in out] == ["edge"]     # kept once, owned by right

    def test_order_preserved_across_chunks(self):
        out = stitch([
            tc(0, 0.0, 300.0, [Segment(10.0, 11.0, "first")]),
            tc(1, 295.0, 595.0, [Segment(100.0, 101.0, "second")]),  # abs 395
        ])
        assert [s.text for s in out] == ["first", "second"]
        assert out[0].start < out[1].start

    def test_input_order_does_not_matter(self):
        a = tc(0, 0.0, 300.0, [Segment(10.0, 11.0, "first")])
        b = tc(1, 295.0, 595.0, [Segment(100.0, 101.0, "second")])
        assert [s.text for s in stitch([b, a])] == ["first", "second"]

    def test_ids_are_resequenced(self):
        out = stitch([
            tc(0, 0.0, 300.0, [Segment(10.0, 11.0, "a", id=99)]),
            tc(1, 295.0, 595.0, [Segment(100.0, 101.0, "b", id=42)]),
        ])
        assert [s.id for s in out] == [0, 1]

    def test_three_chunks_from_real_planner(self):
        # Tie the two functions together: plan real windows, put one word solidly
        # inside each chunk's OWN territory (clear of the leading overlap), and
        # confirm all three survive in order. A local time inside the leading
        # overlap would land in the previous chunk's zone and be dropped by design.
        windows = plan_windows(650.0, 300.0, 5.0)          # 3 windows
        chunks = [
            tc(i, s, e, [Segment(10.0, 11.0, f"w{i}")])    # local 10s clears the 5s overlap
            for i, (s, e) in enumerate(windows)
        ]
        out = stitch(chunks)
        assert [s.text for s in out] == ["w0", "w1", "w2"]