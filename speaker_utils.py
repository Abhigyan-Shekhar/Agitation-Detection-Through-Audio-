"""Small speaker and timestamp helpers shared by the streaming pipeline."""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Mapping

import config

logger = logging.getLogger(__name__)
_WLK_TIME_RE = re.compile(r"^(?P<h>\d+):(?P<m>[0-5]?\d):(?P<s>[0-5]?\d(?:\.\d+)?)$")


def parse_wlk_timestamp(value: Any) -> float | None:
    """Return seconds from a WLK H:MM:SS timestamp or numeric value."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    match = _WLK_TIME_RE.fullmatch(str(value).strip())
    if not match:
        return None
    return (
        int(match.group("h")) * 3600.0
        + int(match.group("m")) * 60.0
        + float(match.group("s"))
    )


def wlk_relative_to_wallclock(stream_start: float | None, value: Any) -> float | None:
    """Convert a WLK stream-relative timestamp to the audio wall-clock domain."""
    relative = parse_wlk_timestamp(value)
    if stream_start is None or relative is None:
        return None
    return stream_start + relative


def normalize_speaker_id(value: Any) -> int | str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def default_speaker_label(speaker_id: int | str | None) -> str | None:
    return None if speaker_id is None else f"Speaker {speaker_id}"


def configured_speaker_aliases(raw: str = config.SPEAKER_ALIASES_JSON) -> dict[int | str, str]:
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring malformed SPEAKER_ALIASES_JSON: %s", exc)
        return {}
    if not isinstance(data, Mapping):
        logger.warning("Ignoring SPEAKER_ALIASES_JSON because it is not an object")
        return {}
    aliases: dict[int | str, str] = {}
    for key, value in data.items():
        speaker_id = normalize_speaker_id(key)
        label = str(value).strip()
        if speaker_id is not None and label:
            aliases[speaker_id] = label
    return aliases


class SpeakerRegistry:
    """Session-local speaker labels with optional human-readable aliases."""

    def __init__(self, aliases: Mapping[int | str, str] | None = None) -> None:
        self._aliases = dict(aliases or configured_speaker_aliases())
        self._seen: set[int | str] = set()
        self._lock = threading.Lock()

    def observe(self, speaker_id: int | str | None) -> str | None:
        if speaker_id is None:
            return None
        with self._lock:
            self._seen.add(speaker_id)
        return self._aliases.get(speaker_id) or default_speaker_label(speaker_id)

    @property
    def speakers_seen(self) -> tuple[int | str, ...]:
        with self._lock:
            return tuple(sorted(self._seen, key=str))

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()
