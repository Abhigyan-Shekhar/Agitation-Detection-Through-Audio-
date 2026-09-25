from __future__ import annotations

import pytest

from agitation_heatmap import DEFAULT_AGITATION_EVIDENCE_SCORE, build_agitation_heatmap_intervals


def _event(**overrides):
    event = {
        "behaviour": "Complaining", "start": 10.0, "end": 20.0,
        "confidence": 0.7, "validated": True, "transcript": "This is terrible.",
    }
    event.update(overrides)
    return event


def _scores(intervals):
    return [interval.score for interval in intervals]


def test_no_events_produces_zero_scores_across_audio_duration():
    intervals = build_agitation_heatmap_intervals([], 30, window_seconds=10)

    assert [(interval.start, interval.end) for interval in intervals] == [(0, 10), (10, 20), (20, 30)]
    assert _scores(intervals) == [0, 0, 0]


def test_one_event_contributes_evidence_to_its_timestamped_interval():
    intervals = build_agitation_heatmap_intervals([_event()], 30, window_seconds=10)

    assert _scores(intervals) == [0, pytest.approx(0.7), 0]
    assert intervals[1].behaviours == ("Complaining",)
    assert intervals[1].transcripts == ("This is terrible.",)


def test_overlapping_events_are_probabilistically_combined_and_normalized():
    intervals = build_agitation_heatmap_intervals([
        _event(start=10, end=20, confidence=0.6),
        _event(behaviour="Vocal agitation", start=15, end=25, confidence=0.8),
    ], 30, window_seconds=5)

    # The shared 15–20 interval uses 1 - (1 - .6) * (1 - .8), not addition.
    assert intervals[3].score == pytest.approx(0.92)
    assert all(0 <= interval.score <= 1 for interval in intervals)


def test_events_are_clipped_to_the_valid_audio_duration():
    intervals = build_agitation_heatmap_intervals([_event(start=-4, end=8)], 20, window_seconds=10)

    assert _scores(intervals) == [pytest.approx(0.7), 0]


def test_missing_score_uses_documented_fallback_without_crashing():
    intervals = build_agitation_heatmap_intervals([_event(confidence=None, initial_score=None)], 30, window_seconds=10)

    assert intervals[1].score == pytest.approx(DEFAULT_AGITATION_EVIDENCE_SCORE)


def test_only_agitation_taxonomy_behaviours_contribute():
    intervals = build_agitation_heatmap_intervals([
        _event(behaviour="Complaining"),
        _event(behaviour="Normal conversation", confidence=1.0),
    ], 30, window_seconds=10)

    assert intervals[1].score == pytest.approx(0.7)
    assert intervals[1].behaviours == ("Complaining",)


@pytest.mark.parametrize("start,end", [(10, 10), (15, 10), (float("nan"), 12)])
def test_zero_or_invalid_duration_events_are_ignored_safely(start, end):
    intervals = build_agitation_heatmap_intervals([_event(start=start, end=end)], 30, window_seconds=10)

    assert _scores(intervals) == [0, 0, 0]
