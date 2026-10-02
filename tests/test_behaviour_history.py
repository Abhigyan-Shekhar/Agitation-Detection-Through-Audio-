from datetime import datetime, timedelta

from behaviour_history import (
    append_unique_event,
    build_behaviour_timeline,
    count_behaviours,
    get_most_common_behaviour,
    get_recent_events,
)


def event(minutes_ago: float, behaviour: str = "Negativism", now: datetime | None = None):
    base = now or datetime(2026, 8, 11, 18, 40)
    return {
        "timestamp": base - timedelta(minutes=minutes_ago),
        "resident": "Unassigned resident",
        "behaviour": behaviour,
        "severity": "Mild",
        "location": "Observation area",
        "source": "Detected",
        "notes": "evidence",
    }


def test_recent_events_include_5_and_29_minutes():
    now = datetime(2026, 8, 11, 18, 40)
    recent = get_recent_events([event(5, now=now), event(29, now=now)], now=now)
    assert len(recent) == 2


def test_recent_events_exclude_older_than_30_minutes():
    now = datetime(2026, 8, 11, 18, 40)
    assert get_recent_events([event(30.01, now=now)], now=now) == []


def test_recent_events_include_inclusive_boundary_timestamps():
    now = datetime(2026, 8, 11, 18, 40)
    recent = get_recent_events([event(30, now=now), event(0, now=now)], now=now)
    assert [item["timestamp"] for item in recent] == [now - timedelta(minutes=30), now]


def test_multiple_behaviour_types_are_counted_correctly():
    counts = count_behaviours([
        event(1, "Negativism"),
        event(2, "Complaining"),
        event(3, "Negativism"),
    ])
    assert counts == {"Negativism": 2, "Complaining": 1}


def test_most_repeated_behaviour_is_calculated_correctly():
    most_common = get_most_common_behaviour([
        event(1, "Complaining"),
        event(2, "Negativism"),
        event(3, "Negativism"),
    ])
    assert most_common == ("Negativism", 2)


def test_empty_window_returns_zero_events_and_no_most_common():
    now = datetime(2026, 8, 11, 18, 40)
    recent = get_recent_events([event(31, now=now)], now=now)
    assert recent == []
    assert get_most_common_behaviour(recent) is None


def test_append_unique_event_prevents_duplicate_rerun_records():
    history = []
    record = event(1, "Negativism")
    assert append_unique_event(history, record) is True
    assert append_unique_event(history, dict(record)) is False
    assert len(history) == 1


def test_build_behaviour_timeline_groups_recent_events_by_time_and_behaviour():
    now = datetime(2026, 8, 11, 18, 40)
    timeline = build_behaviour_timeline([
        event(5, "Negativism", now=now),
        event(6, "Negativism", now=now),
        event(31, "Complaining", now=now),
    ], now=now)
    assert sum(row["events"] for row in timeline) == 2
    assert {row["behaviour"] for row in timeline} == {"Negativism"}
