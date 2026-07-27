from __future__ import annotations

from datetime import date, datetime, time

import pytest

from behaviour_dashboard_model import (
    can_log_behaviour,
    can_view_dashboard,
    event_to_record,
    filter_records,
    get_behavior_types,
    make_manual_event_record,
    records_dataframe,
    summarize_records,
    validate_manual_event,
)
from event_models import BehaviourEvent, FusedResult


def test_behavior_type_reference_data_comes_from_taxonomy():
    behaviour_types = get_behavior_types()
    labels = {item.label for item in behaviour_types}
    codes = {item.code for item in behaviour_types}

    assert "Making verbal sexual advances" in labels
    assert "Making strange noises" in labels
    assert "AUDIO_VERBAL_SEXUAL_ADVANCES" in codes
    assert all(item.active for item in behaviour_types)


def test_manual_event_validation_requires_core_fields():
    errors = validate_manual_event(
        resident="",
        behaviour="Screaming",
        severity="Low",
        location="",
        duration=-1,
    )

    assert "Resident / person is required." in errors
    assert "Location is required." in errors
    assert "Duration cannot be negative." in errors


def test_manual_event_record_uses_canonical_taxonomy_fields():
    record = make_manual_event_record(
        timestamp=datetime(2026, 7, 26, 9, 30),
        resident="Room 12",
        behaviour="weird laughter",
        severity="Medium",
        location="Hallway",
        duration=3,
        trigger="shift change",
        intervention="Reassurance",
        outcome="Improved",
        notes="Settled after staff check-in.",
    )

    assert record["behaviour"] == "Making strange noises"
    assert record["internal_code"] == "AUDIO_STRANGE_NOISE"
    assert record["cmai_category"] == "Verbally non-aggressive: strange noises"
    assert record["source"] == "Manual"


def test_manual_event_record_rejects_unknown_behaviour():
    with pytest.raises(ValueError, match="supported taxonomy"):
        make_manual_event_record(
            timestamp=datetime(2026, 7, 26, 9, 30),
            resident="Room 12",
            behaviour="pacing",
            severity="Medium",
            location="Hallway",
            duration=3,
        )


def test_detected_event_record_matches_dashboard_storage_shape():
    event = BehaviourEvent(
        internal_code="AUDIO_COMPLAINING",
        canonical_label="Complaining",
        cmai_category="Verbally non-aggressive: complaining",
        severity=None,
        timestamp=datetime(2026, 7, 26, 10, 0),
        notes="Complaint terms detected.",
    )
    result = FusedResult(severity="High")

    record = event_to_record(event, result)

    assert record["behaviour"] == "Complaining"
    assert record["severity"] == "High"
    assert record["source"] == "Detected"


def test_detected_event_record_preserves_unix_timestamp():
    event = BehaviourEvent(
        canonical_label="Complaining",
        timestamp=datetime(2026, 7, 26, 10, 0).timestamp(),
    )

    record = event_to_record(event)

    assert record["timestamp"] == datetime(2026, 7, 26, 10, 0)


def test_permissions_allow_logging_only_for_care_roles():
    assert can_view_dashboard("Read only")
    assert not can_log_behaviour("Read only")
    assert can_log_behaviour("Care staff")
    assert can_log_behaviour("Reviewer")


def test_filter_records_and_summary_metrics():
    records = [
        make_manual_event_record(
            timestamp=datetime(2026, 7, 26, 9, 30),
            resident="Room 12",
            behaviour="Screaming",
            severity="High",
            location="Hallway",
            duration=2,
            notes="loud call for help",
        ),
        make_manual_event_record(
            timestamp=datetime(2026, 7, 25, 15, 0),
            resident="Room 8",
            behaviour="Complaining",
            severity="Low",
            location="Dining room",
            duration=5,
        ),
    ]
    df = records_dataframe(records)

    filtered = filter_records(
        df,
        {
            "residents": ["Room 12"],
            "behaviours": ["Screaming"],
            "severities": ["High"],
            "locations": ["Hallway"],
            "date_range": (date(2026, 7, 26), date(2026, 7, 26)),
            "time_range": (time(0, 0), time(23, 59)),
            "search": "help",
        },
    )
    summary = summarize_records(filtered, today=date(2026, 7, 26))

    assert len(filtered) == 1
    assert summary["today_events"] == 1
    assert summary["high_severity_events"] == 1
    assert summary["most_common_behaviour"] == "Screaming"
