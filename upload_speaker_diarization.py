"""Optional injectable speaker attribution for uploaded audio.

The default implementation is intentionally dependency-free and returns unknown
speaker attribution. Deployments that install a diarization backend can inject
an implementation of ``UploadSpeakerDiarizer`` without changing the Person 1 ->
Person 2 contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SpeakerAttribution:
    speaker_id: int | str | None = None
    speaker_label: str | None = None


class UploadSpeakerDiarizer(Protocol):
    def identify(self, audio: np.ndarray, sample_rate: int, start: float, end: float) -> SpeakerAttribution:
        """Return speaker attribution for one uploaded transcript/audio span."""


class UnknownSpeakerDiarizer:
    def identify(self, audio: np.ndarray, sample_rate: int, start: float, end: float) -> SpeakerAttribution:
        return SpeakerAttribution()
