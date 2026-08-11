"""Rolling behaviour-history analytics for dashboard event records.

This module operates on already-created behaviour event records. It does not
perform classification and does not alter the real-time detection pipeline.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Iterable


DEFAULT_WINDOW_MINUTES = 30


def _coerce_datetime(value: Any) -> datetime | None:
    """Return ``value`` as a naive ``datetime`` when possible."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    try:
        if isinstance(value, str):
            return datetime.fromisoformat(value).replace(tzinfo=None)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if hasattr(value, "to_pydatetime"):
            dt = value.to_pydatetime()
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except (TypeError, ValueError, OSError):
        return None
    return None


def event_fingerprint(record: dict[str, Any]) -> tuple[Any, ...]:
    """Build a stable fingerprint for avoiding duplicate dashboard records."""
    timestamp = _coerce_datetime(record.get("timestamp"))
    timestamp_key = timestamp.isoformat(timespec="microseconds") if timestamp else record.get("timestamp")
    return (
        timestamp_key,
        record.get("resident"),
        record.get("behaviour"),
        record.get("severity"),
        record.get("location"),
        record.get("source"),
        record.get("notes"),
    )


def append_unique_event(history: list[dict[str, Any]], record: dict[str, Any]) -> bool:
    """Append ``record`` to ``history`` unless an equivalent event is present."""
    fingerprint = event_fingerprint(record)
    if any(event_fingerprint(existing) == fingerprint for existing in history):
        return False
    history.append(record)
    return True


def get_recent_events(
    events: Iterable[dict[str, Any]],
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return events with timestamps inside the rolling lookback window.

    The cutoff is inclusive: ``now - window_minutes <= timestamp <= now``.
    Events outside the window remain in the caller's original history.
    """
    current_time = (now or datetime.now()).replace(tzinfo=None)
    cutoff = current_time - timedelta(minutes=window_minutes)
    recent: list[dict[str, Any]] = []
    for event in events:
        timestamp = _coerce_datetime(event.get("timestamp"))
        if timestamp is not None and cutoff <= timestamp <= current_time:
            recent.append(event)
    return recent


def count_behaviours(events: Iterable[dict[str, Any]]) -> Counter[str]:
    """Count canonical behaviour labels in ``events``."""
    return Counter(
        str(event.get("behaviour"))
        for event in events
        if event.get("behaviour")
    )


def get_most_common_behaviour(events: Iterable[dict[str, Any]]) -> tuple[str, int] | None:
    """Return the most frequent behaviour label and count, or ``None``."""
    counts = count_behaviours(events)
    if not counts:
        return None
    return counts.most_common(1)[0]


def build_behaviour_timeline(
    events: Iterable[dict[str, Any]],
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    now: datetime | None = None,
    bucket_minutes: int = 5,
) -> list[dict[str, Any]]:
    """Build bucketed counts for a rolling-window behaviour timeline chart."""
    recent = get_recent_events(events, window_minutes=window_minutes, now=now)
    buckets: Counter[tuple[datetime, str]] = Counter()
    for event in recent:
        timestamp = _coerce_datetime(event.get("timestamp"))
        behaviour = event.get("behaviour")
        if timestamp is None or not behaviour:
            continue
        bucket_minute = (timestamp.minute // bucket_minutes) * bucket_minutes
        bucket = timestamp.replace(minute=bucket_minute, second=0, microsecond=0)
        buckets[(bucket, str(behaviour))] += 1

    return [
        {"time": bucket, "behaviour": behaviour, "events": count}
        for (bucket, behaviour), count in sorted(buckets.items())
    ]
