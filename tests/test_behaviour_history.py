from __future__ import annotations

from datetime import datetime, timedelta

from behaviour_history import (
    append_record_once,
    behaviour_breakdown,
    get_most_common_behaviour,
    get_recent_events,
)


def _record(behaviour: str, timestamp: datetime, event_id: str | None = None) -> dict:
    return {
        "event_id": event_id,
        "timestamp": timestamp,
        "resident": "Resident A",
        "behaviour": behaviour,
        "severity": "Low",
        "source": "Detected",
    }


def test_event_from_5_minutes_ago_is_included():
    now = datetime(2026, 8, 10, 18, 40)
    event = _record("Negativism", now - timedelta(minutes=5))

    assert get_recent_events([event], window_minutes=30, now=now) == [event]


def test_event_from_29_minutes_ago_is_included():
    now = datetime(2026, 8, 10, 18, 40)
    event = _record("Complaining", now - timedelta(minutes=29))

    assert get_recent_events([event], window_minutes=30, now=now) == [event]


def test_event_older_than_30_minutes_is_excluded():
    now = datetime(2026, 8, 10, 18, 40)
    event = _record("Screaming", now - timedelta(minutes=30, seconds=1))

    assert get_recent_events([event], window_minutes=30, now=now) == []


def test_boundary_timestamps_are_inclusive():
    now = datetime(2026, 8, 10, 18, 40)
    at_cutoff = _record("Negativism", now - timedelta(minutes=30))
    at_now = _record("Complaining", now)

    assert get_recent_events([at_cutoff, at_now], window_minutes=30, now=now) == [
        at_cutoff,
        at_now,
    ]


def test_multiple_behaviour_types_are_counted_correctly():
    now = datetime(2026, 8, 10, 18, 40)
    events = [
        _record("Negativism", now - timedelta(minutes=5)),
        _record("Complaining", now - timedelta(minutes=4)),
        _record("Negativism", now - timedelta(minutes=3)),
        _record("Screaming", now - timedelta(minutes=2)),
        _record("Complaining", now - timedelta(minutes=1)),
        _record("Negativism", now),
    ]

    assert behaviour_breakdown(events) == [
        {"behaviour": "Negativism", "events": 3},
        {"behaviour": "Complaining", "events": 2},
        {"behaviour": "Screaming", "events": 1},
    ]


def test_most_repeated_behaviour_is_calculated_correctly():
    now = datetime(2026, 8, 10, 18, 40)
    events = [
        _record("Complaining", now - timedelta(minutes=4)),
        _record("Negativism", now - timedelta(minutes=3)),
        _record("Negativism", now - timedelta(minutes=2)),
    ]

    assert get_most_common_behaviour(events) == ("Negativism", 2)


def test_empty_30_minute_window_returns_zero_events():
    now = datetime(2026, 8, 10, 18, 40)
    old_event = _record("Negativism", now - timedelta(minutes=31))

    assert len(get_recent_events([old_event], window_minutes=30, now=now)) == 0


def test_empty_window_returns_no_most_repeated_behaviour():
    assert get_most_common_behaviour([]) == (None, 0)


def test_repeated_dashboard_execution_does_not_duplicate_events():
    now = datetime(2026, 8, 10, 18, 40)
    history: list[dict] = []
    seen_keys: set[tuple] = set()
    event = _record("Negativism", now, event_id="behaviour-123")

    first_append = append_record_once(history, event, seen_keys)
    second_append = append_record_once(history, event, seen_keys)

    assert first_append is True
    assert second_append is False
    assert history == [event]
