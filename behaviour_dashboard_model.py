"""Dashboard-facing behaviour reference data, storage records, and filters."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Iterable

import pandas as pd

from audio_behaviour_taxonomy import get_supported_behaviours, map_observed_behaviour
from event_models import BehaviourEvent, FusedResult


SEVERITY_OPTIONS: tuple[str, ...] = ("Low", "Medium", "High", "Critical")
USER_ROLES: tuple[str, ...] = ("Care staff", "Reviewer", "Read only")
BEHAVIOUR_RECORD_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "resident",
    "behaviour",
    "internal_code",
    "cmai_category",
    "severity",
    "location",
    "duration",
    "trigger",
    "intervention",
    "outcome",
    "notes",
    "source",
)


@dataclass(frozen=True)
class BehaviorType:
    """Dashboard reference row derived from the canonical taxonomy."""

    code: str
    label: str
    cmai_category: str
    modality: str
    description: str
    aliases: tuple[str, ...]
    active: bool = True


def get_behavior_types() -> tuple[BehaviorType, ...]:
    """Return configurable dashboard behaviour labels from the taxonomy."""
    return tuple(
        BehaviorType(
            code=entry.internal_code,
            label=entry.canonical_label,
            cmai_category=entry.cmai_category,
            modality=entry.modality,
            description=entry.description,
            aliases=entry.aliases,
        )
        for entry in get_supported_behaviours()
    )


def can_view_dashboard(role: str | None) -> bool:
    return role in USER_ROLES


def can_log_behaviour(role: str | None) -> bool:
    return role in {"Care staff", "Reviewer"}


def validate_manual_event(
    *,
    resident: str | None,
    behaviour: str | None,
    severity: str | None,
    location: str | None,
    duration: int | float | None,
) -> list[str]:
    """Return validation errors for a manual dashboard event."""
    errors: list[str] = []
    if not resident or not resident.strip():
        errors.append("Resident / person is required.")
    mapped = map_observed_behaviour(behaviour)
    if mapped.mapping_status != "mapped":
        errors.append("Behaviour must be selected from the supported taxonomy.")
    if severity not in SEVERITY_OPTIONS:
        errors.append("Severity must be one of the configured dashboard severities.")
    if not location or not location.strip():
        errors.append("Location is required.")
    if duration is None:
        errors.append("Duration is required.")
    else:
        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            errors.append("Duration must be numeric.")
        else:
            if duration_value < 0:
                errors.append("Duration cannot be negative.")
    return errors


def make_manual_event_record(
    *,
    timestamp: datetime,
    resident: str,
    behaviour: str,
    severity: str,
    location: str,
    duration: int | float,
    trigger: str = "",
    intervention: str = "",
    outcome: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Validate and create the canonical dict stored by the dashboard."""
    errors = validate_manual_event(
        resident=resident,
        behaviour=behaviour,
        severity=severity,
        location=location,
        duration=duration,
    )
    if errors:
        raise ValueError(" ".join(errors))

    mapped = map_observed_behaviour(behaviour)
    return {
        "timestamp": timestamp,
        "resident": resident.strip(),
        "behaviour": mapped.canonical_label,
        "internal_code": mapped.internal_code,
        "cmai_category": mapped.cmai_category,
        "severity": severity,
        "location": location.strip(),
        "duration": float(duration),
        "trigger": trigger.strip(),
        "intervention": intervention.strip(),
        "outcome": outcome.strip(),
        "notes": notes.strip(),
        "source": "Manual",
    }


def event_to_record(event: BehaviourEvent, result: FusedResult | None = None) -> dict[str, Any]:
    """Convert a detected BehaviourEvent into the dashboard storage shape."""
    timestamp = event.timestamp
    if isinstance(timestamp, (int, float)):
        timestamp = datetime.fromtimestamp(timestamp)
    if not isinstance(timestamp, datetime):
        timestamp = datetime.now()
    return {
        "timestamp": timestamp,
        "resident": event.person or "Unassigned resident",
        "behaviour": event.canonical_label or event.behaviour_type or "Unmapped audio behaviour",
        "internal_code": event.internal_code,
        "cmai_category": event.cmai_category,
        "severity": event.severity or (result.severity if result else "Low"),
        "location": event.location or "Observation area",
        "duration": event.duration,
        "trigger": event.trigger or "",
        "intervention": event.intervention or "",
        "outcome": event.outcome or "",
        "notes": event.notes or "",
        "source": "Detected",
    }


def records_dataframe(records: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Build the canonical events DataFrame used by dashboard tabs."""
    if not records:
        return pd.DataFrame(columns=BEHAVIOUR_RECORD_COLUMNS)
    df = pd.DataFrame(records)
    for column in BEHAVIOUR_RECORD_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.loc[:, BEHAVIOUR_RECORD_COLUMNS].sort_values("timestamp", ascending=False)


def _selected(values: Iterable[Any] | None) -> list[Any]:
    return [value for value in values or [] if value not in (None, "")]


def _range_pair(value: Any) -> tuple[Any, Any] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]
    return None


def filter_records(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Apply sidebar filter values to dashboard records."""
    if df.empty:
        return df
    filtered = df.copy()
    for column, selected_values in [
        ("resident", filters.get("residents")),
        ("behaviour", filters.get("behaviours")),
        ("severity", filters.get("severities")),
        ("location", filters.get("locations")),
    ]:
        selected = _selected(selected_values)
        if selected:
            filtered = filtered[filtered[column].isin(selected)]

    date_range = filters.get("date_range")
    date_pair = _range_pair(date_range)
    if date_pair is not None:
        start_date, end_date = date_pair
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start_date)
            & (filtered["timestamp"].dt.date <= end_date)
        ]

    time_range = filters.get("time_range")
    time_pair = _range_pair(time_range)
    if time_pair is not None:
        start_time, end_time = time_pair
        if isinstance(start_time, time) and isinstance(end_time, time):
            filtered = filtered[
                (filtered["timestamp"].dt.time >= start_time)
                & (filtered["timestamp"].dt.time <= end_time)
            ]

    search = str(filters.get("search", "")).strip().lower()
    if search:
        haystack = (
            filtered["notes"].fillna("") + " "
            + filtered["outcome"].fillna("") + " "
            + filtered["trigger"].fillna("")
        ).str.lower()
        filtered = filtered[haystack.str.contains(search, regex=False)]
    return filtered


def summarize_records(df: pd.DataFrame, today: date | None = None) -> dict[str, Any]:
    """Return high-level dashboard metrics for filtered event records."""
    if df.empty:
        return {
            "today_events": 0,
            "high_severity_events": 0,
            "most_common_behaviour": "-",
            "most_active_resident": "-",
            "average_severity": "-",
        }

    today_value = today or date.today()
    weights = {"Low": 1, "Mild": 1.5, "Medium": 2, "Moderate": 2.5, "High": 3, "Critical": 4}
    return {
        "today_events": len(df[df["timestamp"].dt.date == today_value]),
        "high_severity_events": len(df[df["severity"].isin(["High", "Critical"])]),
        "most_common_behaviour": df["behaviour"].mode().iat[0],
        "most_active_resident": df["resident"].mode().iat[0],
        "average_severity": f"{df['severity'].map(weights).fillna(0).mean():.1f}/4",
    }
