from datetime import datetime, timezone

import pytest

from supabase_event_store import MissingSupabaseCredentialsError, SupabaseEventStore, SupabaseEventStoreError, validate_event


class _Response:
    def __init__(self, data): self.data = data


class _Table:
    def __init__(self, rows): self.rows, self.inserted = rows, None
    def insert(self, payload):
        self.inserted = payload
        values = payload if isinstance(payload, list) else [payload]
        self.rows[:] = [{"id": f"id-{i}", **value} for i, value in enumerate(values)]
        return self
    def select(self, _fields): return self
    def order(self, *_args, **_kwargs): return self
    def limit(self, *_args, **_kwargs): return self
    def execute(self): return _Response(self.rows)


class _Client:
    def __init__(self): self.table_instance = _Table([])
    def table(self, _name): return self.table_instance


def test_insert_and_retrieve_are_allowlisted_metadata_only():
    client = _Client()
    store = SupabaseEventStore(client=client)
    now = datetime(2026, 9, 15, 3, 12, 31, tzinfo=timezone.utc)
    inserted = store.insert_event(recorded_at=now, audio_timestamp="00:03.2 – 00:05.4", behaviour="complaining")
    assert set(client.table_instance.inserted) == {"recorded_at", "audio_timestamp", "behaviour"}
    assert inserted["recorded_at"].endswith("+00:00")
    assert inserted["audio_timestamp"] == "00:03.2 – 00:05.4"
    assert inserted["behaviour"] == "Complaining"
    assert set(store.list_events()[0]) == {"id", "recorded_at", "audio_timestamp", "behaviour"}


def test_multiple_events_and_invalid_input():
    client = _Client()
    store = SupabaseEventStore(client=client)
    now = datetime.now(timezone.utc)
    assert len(store.insert_events([
        validate_event(recorded_at=now, audio_timestamp="00:01.2", behaviour="Complaining"),
        validate_event(recorded_at=now, audio_timestamp="00:02.8", behaviour="Negativism"),
    ])) == 2
    with pytest.raises(ValueError):
        validate_event(recorded_at=now.replace(tzinfo=None), audio_timestamp="00:01", behaviour="pacing")


def test_missing_credentials_does_not_create_client(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    with pytest.raises(MissingSupabaseCredentialsError):
        _ = SupabaseEventStore().client


def test_connection_failure_is_reported_as_persistence_error():
    class FailingTable(_Table):
        def execute(self): raise RuntimeError("database unavailable")

    class FailingClient:
        def table(self, _name): return FailingTable([])

    with pytest.raises(SupabaseEventStoreError, match="database unavailable"):
        SupabaseEventStore(client=FailingClient()).insert_event(
            recorded_at=datetime.now(timezone.utc), audio_timestamp="00:01", behaviour="Complaining"
        )


@pytest.mark.parametrize("physical_label", ["pacing", "kicking", "punching", "pushing", "grabbing"])
def test_physical_behaviour_is_rejected_before_insert(physical_label):
    client = _Client()
    with pytest.raises(SupabaseEventStoreError, match="persistence skipped"):
        SupabaseEventStore(client=client).insert_event(
            recorded_at=datetime.now(timezone.utc), audio_timestamp="00:01", behaviour=physical_label
        )
    assert client.table_instance.inserted is None
