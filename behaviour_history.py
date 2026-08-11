"""Rolling behaviour history helpers for dashboard analytics.

These helpers operate on already-detected behaviour events/records. They do
not classify audio, change thresholds, or compute an agitation score.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Iterable

from event_models import BehaviourEvent

DEFAULT_WINDOW_MINUTES = 30


def normalise_event_timestamp(value: Any) -> datetime | None:
    """Return a datetime for supported event timestamp formats."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    return None


def event_record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Build a stable key for preventing duplicate dashboard records."""
    event_id = record.get("event_id")
    if event_id:
        return ("event_id", event_id)
    return (
        "record",
        record.get("timestamp"),
        record.get("resident"),
        record.get("behaviour"),
        record.get("severity"),
        record.get("source"),
    )


def append_record_once(
    history: list[dict[str, Any]],
    record: dict[str, Any],
    seen_keys: set[tuple[Any, ...]],
) -> bool:
    """Append a behaviour record only if its stable key has not been seen."""
    key = event_record_key(record)
    if key in seen_keys:
        return False
    seen_keys.add(key)
    history.append(record)
    return True


def _event_timestamp(event: BehaviourEvent | dict[str, Any]) -> datetime | None:
    if isinstance(event, BehaviourEvent):
        return normalise_event_timestamp(event.timestamp)
    return normalise_event_timestamp(event.get("timestamp"))


def _event_behaviour(event: BehaviourEvent | dict[str, Any]) -> str:
    if isinstance(event, BehaviourEvent):
        return event.canonical_label or event.behaviour_type or "Unmapped audio behaviour"
    return str(event.get("behaviour") or event.get("canonical_label") or "Unmapped audio behaviour")


def get_recent_events(
    events: Iterable[BehaviourEvent | dict[str, Any]],
    *,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    now: datetime | None = None,
) -> list[BehaviourEvent | dict[str, Any]]:
    """Return events whose timestamps fall inside the rolling time window."""
    current_time = now or datetime.now()
    cutoff = current_time - timedelta(minutes=window_minutes)
    recent: list[BehaviourEvent | dict[str, Any]] = []
    for event in events:
        timestamp = _event_timestamp(event)
        if timestamp is not None and cutoff <= timestamp <= current_time:
            recent.append(event)
    return recent


def count_behaviours(events: Iterable[BehaviourEvent | dict[str, Any]]) -> Counter[str]:
    """Count canonical behaviour labels in the provided events."""
    return Counter(_event_behaviour(event) for event in events)


def get_most_common_behaviour(
    events: Iterable[BehaviourEvent | dict[str, Any]],
) -> tuple[str | None, int]:
    """Return the highest-frequency behaviour label and count."""
    counts = count_behaviours(events)
    if not counts:
        return None, 0
    return counts.most_common(1)[0]


def behaviour_breakdown(
    events: Iterable[BehaviourEvent | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return sorted behaviour-count rows for dashboard display."""
    counts = count_behaviours(events)
    return [
        {"behaviour": behaviour, "events": count}
        for behaviour, count in counts.most_common()
    ]
