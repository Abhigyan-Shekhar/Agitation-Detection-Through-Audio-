"""Dataset-derived labels for non-speech human vocalizations.

The labels below come from public human vocal sound datasets used as label
taxonomies for this prototype:

* OpenSLR SLR99 / Deeply Nonverbal Vocalization Dataset
* VocalSound
* Nonspeech7k
* EmoGator

This module does not train or load an audio classifier. It provides a curated,
auditable label vocabulary so transcript annotations or upstream audio-caption
outputs such as "[moaning]" or "throat clearing" can be mapped into the CMAI
"making strange noises" behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrangeNoiseLabel:
    """One non-speech human vocalization label and its CMAI handling."""

    canonical: str
    variants: tuple[str, ...]
    datasets: tuple[str, ...]
    confidence: float
    map_to_strange_noise: bool = True
    notes: str = ""


DATASET_SLR99 = "OpenSLR SLR99 / Deeply Nonverbal Vocalization Dataset"
DATASET_VOCALSOUND = "VocalSound"
DATASET_NONSPEECH7K = "Nonspeech7k"
DATASET_EMOGATOR = "EmoGator"


STRANGE_NOISE_LABELS: tuple[StrangeNoiseLabel, ...] = (
    StrangeNoiseLabel("moaning", ("moan", "moaning", "moans"), (DATASET_SLR99, DATASET_EMOGATOR), 0.85),
    StrangeNoiseLabel("groaning", ("groan", "groaning", "groans"), (DATASET_EMOGATOR,), 0.85),
    StrangeNoiseLabel("crying", ("cry", "crying", "cries", "sob", "sobbing", "weeping"), (DATASET_SLR99, DATASET_NONSPEECH7K, DATASET_EMOGATOR), 0.85),
    StrangeNoiseLabel("laughing", ("laugh", "laughing", "laughter", "laughs", "chuckle", "chuckling", "giggle", "giggling"), (DATASET_SLR99, DATASET_VOCALSOUND, DATASET_NONSPEECH7K, DATASET_EMOGATOR), 0.75),
    StrangeNoiseLabel("sighing", ("sigh", "sighing", "sighs"), (DATASET_SLR99, DATASET_VOCALSOUND, DATASET_EMOGATOR), 0.70),
    StrangeNoiseLabel("panting", ("pant", "panting", "pants", "gasp", "gasping"), (DATASET_SLR99,), 0.75),
    StrangeNoiseLabel("yawning", ("yawn", "yawning", "yawns"), (DATASET_SLR99, DATASET_NONSPEECH7K), 0.70),
    StrangeNoiseLabel("throat clearing", ("throat clearing", "clearing throat", "clears throat", "ahem"), (DATASET_SLR99, DATASET_VOCALSOUND), 0.70),
    StrangeNoiseLabel("coughing", ("cough", "coughing", "coughs"), (DATASET_SLR99, DATASET_VOCALSOUND, DATASET_NONSPEECH7K), 0.65),
    StrangeNoiseLabel("sneezing", ("sneeze", "sneezing", "sneezes", "achoo"), (DATASET_SLR99, DATASET_VOCALSOUND, DATASET_NONSPEECH7K), 0.65),
    StrangeNoiseLabel("sniffing", ("sniff", "sniffing", "sniffs", "sniffle", "sniffling"), (DATASET_VOCALSOUND,), 0.60),
    StrangeNoiseLabel("breathing", ("breath", "breathing", "heavy breathing"), (DATASET_NONSPEECH7K,), 0.60),
    StrangeNoiseLabel("teeth chattering", ("teeth chattering", "teeth-chattering", "chattering teeth"), (DATASET_SLR99,), 0.80),
    StrangeNoiseLabel("teeth grinding", ("teeth grinding", "teeth-grinding", "grinding teeth"), (DATASET_SLR99,), 0.80),
    StrangeNoiseLabel("tongue clicking", ("tongue clicking", "tongue-clicking", "clicking tongue", "tongue click", "clicks tongue"), (DATASET_SLR99,), 0.75),
    StrangeNoiseLabel("nose blowing", ("nose blowing", "nose-blowing", "blowing nose", "blows nose"), (DATASET_SLR99,), 0.65),
    StrangeNoiseLabel("lip popping", ("lip popping", "lip-popping", "popping lips", "lip pop"), (DATASET_SLR99,), 0.75),
    StrangeNoiseLabel("lip smacking", ("lip smacking", "lip-smacking", "smacking lips", "lip smack"), (DATASET_SLR99,), 0.75),
    StrangeNoiseLabel("screaming", ("scream", "screaming", "screams", "shriek", "shrieking"), (DATASET_SLR99, DATASET_NONSPEECH7K), 0.0, False, "Handled by AUDIO_SCREAMING."),
)


STRANGE_NOISE_DATASET_LABELS: tuple[str, ...] = tuple(
    label.canonical for label in STRANGE_NOISE_LABELS if label.map_to_strange_noise
)
