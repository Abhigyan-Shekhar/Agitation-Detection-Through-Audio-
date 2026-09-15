"""Minimal Supabase persistence for validated behaviour-event metadata."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import os
from typing import Any, Iterable

from person2_module import canonicalize_person2_behaviour


class SupabaseEventStoreError(RuntimeError):
    """Base error for configuration, validation, or Supabase failures."""


class MissingSupabaseCredentialsError(SupabaseEventStoreError):
    """Raised when Supabase credentials are not configured."""


@dataclass(frozen=True)
class BehaviourEventMetadata:
    """The complete allow-list of data persisted for one event."""
    recorded_at: datetime
    audio_timestamp: str
    behaviour: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "recorded_at": self.recorded_at.astimezone(timezone.utc).isoformat(),
            "audio_timestamp": self.audio_timestamp,
            "behaviour": self.behaviour,
        }


def validate_event(*, recorded_at: datetime, audio_timestamp: str, behaviour: str) -> BehaviourEventMetadata:
    if not isinstance(recorded_at, datetime):
        raise ValueError("recorded_at must be a datetime")
    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        raise ValueError("recorded_at must be timezone-aware")
    if not isinstance(audio_timestamp, str) or not audio_timestamp.strip():
        raise ValueError("audio_timestamp must be a non-empty string")
    if not isinstance(behaviour, str) or not behaviour.strip():
        raise ValueError("behaviour must be a non-empty string")
    try:
        canonical_behaviour = canonicalize_person2_behaviour(behaviour.strip())
    except ValueError as exc:
        raise SupabaseEventStoreError(f"Supabase persistence skipped: {exc}") from exc
    return BehaviourEventMetadata(recorded_at.astimezone(timezone.utc), audio_timestamp.strip(), canonical_behaviour)


class SupabaseEventStore:
    """Small server-side adapter around the Supabase Python client."""
    def __init__(self, *, url: str | None = None, key: str | None = None, client: Any | None = None) -> None:
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key)

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.configured:
            raise MissingSupabaseCredentialsError("SUPABASE_URL and SUPABASE_KEY are not configured; event was not stored.")
        if importlib.util.find_spec("supabase") is None:
            raise SupabaseEventStoreError("The supabase package is not installed.")
        from supabase import create_client
        self._client = create_client(self.url, self.key)
        return self._client

    def insert_event(self, *, recorded_at: datetime, audio_timestamp: str, behaviour: str) -> dict[str, Any]:
        metadata = validate_event(recorded_at=recorded_at, audio_timestamp=audio_timestamp, behaviour=behaviour)
        try:
            response = self.client.table("behaviour_events").insert(metadata.as_dict()).execute()
        except Exception as exc:  # noqa: BLE001 - preserve pipeline and expose a clear persistence error
            raise SupabaseEventStoreError(f"Supabase insert failed: {exc}") from exc
        rows = getattr(response, "data", None)
        if not isinstance(rows, list) or not rows:
            raise SupabaseEventStoreError("Supabase did not return the inserted behaviour event.")
        return {key: rows[0].get(key) for key in ("id", "recorded_at", "audio_timestamp", "behaviour")}

    def insert_events(self, events: Iterable[BehaviourEventMetadata]) -> list[dict[str, Any]]:
        payload = [validate_event(recorded_at=e.recorded_at, audio_timestamp=e.audio_timestamp, behaviour=e.behaviour).as_dict() for e in events]
        if not payload:
            return []
        try:
            response = self.client.table("behaviour_events").insert(payload).execute()
        except Exception as exc:  # noqa: BLE001
            raise SupabaseEventStoreError(f"Supabase insert failed: {exc}") from exc
        rows = getattr(response, "data", None)
        if not isinstance(rows, list):
            raise SupabaseEventStoreError("Supabase did not return inserted behaviour events.")
        return [{key: row.get(key) for key in ("id", "recorded_at", "audio_timestamp", "behaviour")} for row in rows]

    def list_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        try:
            response = self.client.table("behaviour_events").select("id,recorded_at,audio_timestamp,behaviour").order("recorded_at", desc=True).limit(limit).execute()
        except Exception as exc:  # noqa: BLE001
            raise SupabaseEventStoreError(f"Supabase retrieval failed: {exc}") from exc
        rows = getattr(response, "data", None)
        if not isinstance(rows, list):
            raise SupabaseEventStoreError("Supabase did not return behaviour events.")
        return [{key: row.get(key) for key in ("id", "recorded_at", "audio_timestamp", "behaviour")} for row in rows]
