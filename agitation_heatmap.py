"""Pure aggregation helpers for the dashboard's agitation evidence heatmap.

This module intentionally derives its display values from existing final
behaviour records.  It does not classify audio or estimate a clinical
probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from audio_behaviour_taxonomy import get_supported_behaviours, map_observed_behaviour


DEFAULT_AGITATION_EVIDENCE_SCORE = 0.5
"""Conservative display fallback when an otherwise usable event has no score."""

AGITATION_RELEVANT_BEHAVIOURS = frozenset(
    entry.canonical_label for entry in get_supported_behaviours()
)


@dataclass(frozen=True)
class HeatmapInterval:
    """One fixed audio-relative heatmap interval and its contributing evidence."""

    start: float
    end: float
    score: float
    behaviours: tuple[str, ...] = ()
    transcripts: tuple[str, ...] = ()


def _value(event: FinalBehaviourResult | dict[str, Any], name: str, default: Any = None) -> Any:
    return getattr(event, name, default) if not isinstance(event, dict) else event.get(name, default)


def _score(event: Any) -> float:
    """Use final confidence, then Person 2 score, then a conservative fallback."""
    for name in ("confidence", "initial_score", "score"):
        try:
            value = float(_value(event, name))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return min(1.0, max(0.0, value))
    return DEFAULT_AGITATION_EVIDENCE_SCORE


def _is_agitation_event(event: Any) -> bool:
    """Accept only validated final records mapped to the audio agitation taxonomy."""
    if not bool(_value(event, "validated", True)):
        return False
    behaviour = _value(event, "behaviour", "")
    mapped = map_observed_behaviour(str(behaviour))
    return mapped.canonical_label in AGITATION_RELEVANT_BEHAVIOURS


def _window_size(duration: float, requested_window_seconds: float | None) -> float:
    if requested_window_seconds is not None and requested_window_seconds > 0:
        return requested_window_seconds
    # Approximately 240 cells keeps long recordings legible and responsive.
    return max(1.0, duration / 240.0)


def build_agitation_heatmap_intervals(
    events: Iterable[Any], audio_duration: float, *, window_seconds: float | None = None
) -> list[HeatmapInterval]:
    """Aggregate timestamped final events into bounded evidence intervals.

    Each event is clipped to the original audio bounds.  Overlapping evidence
    is combined with a probabilistic union (``1 - product(1 - score)``), which
    is deterministic, remains in 0–1, and avoids unbounded score addition.
    """
    try:
        duration = float(audio_duration)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(duration) or duration <= 0:
        return []

    size = _window_size(duration, window_seconds)
    interval_count = max(1, math.ceil(duration / size))
    intervals = [
        HeatmapInterval(index * size, min(duration, (index + 1) * size), 0.0)
        for index in range(interval_count)
    ]
    contributions: list[list[tuple[float, str, str]]] = [[] for _ in intervals]
    for event in events:
        if not _is_agitation_event(event):
            continue
        try:
            start, end = float(_value(event, "start")), float(_value(event, "end"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            continue
        first = max(0, int(start // size))
        last = min(interval_count - 1, int(math.nextafter(end, -math.inf) // size))
        behaviour, transcript, score = str(_value(event, "behaviour", "")), str(_value(event, "transcript", "") or _value(event, "evidence", "")), _score(event)
        for index in range(first, last + 1):
            contributions[index].append((score, behaviour, transcript))

    result: list[HeatmapInterval] = []
    for interval, items in zip(intervals, contributions):
        score = 1.0 - math.prod(1.0 - item[0] for item in items)
        behaviours = tuple(sorted({item[1] for item in items if item[1]}))
        transcripts = tuple(dict.fromkeys(item[2] for item in items if item[2]))
        result.append(HeatmapInterval(interval.start, interval.end, min(1.0, max(0.0, score)), behaviours, transcripts))
    return result
