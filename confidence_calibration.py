"""Calibration infrastructure for final behaviour confidence.

Raw verifier scores are uncalibrated support scores. A calibrated confidence is
only produced after fitting this lightweight logistic calibrator on labelled
calibration data that is separate from evaluation/test annotations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json, math
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CalibrationExample:
    features: dict[str, float]
    label: int


class LogisticConfidenceCalibrator:
    def __init__(self, weights: dict[str, float] | None = None, bias: float = 0.0, feature_names: list[str] | None = None) -> None:
        self.weights = weights or {}
        self.bias = float(bias)
        self.feature_names = feature_names or sorted(self.weights)

    def fit(self, examples: Iterable[CalibrationExample], *, epochs: int = 300, lr: float = 0.1, l2: float = 0.001) -> "LogisticConfidenceCalibrator":
        data = list(examples)
        if not data:
            raise ValueError("Calibration requires labelled examples; do not fit on evaluation/test data.")
        self.feature_names = sorted({k for ex in data for k in ex.features})
        self.weights = {k: 0.0 for k in self.feature_names}
        self.bias = 0.0
        for _ in range(epochs):
            grad = {k: 0.0 for k in self.feature_names}; gb = 0.0
            for ex in data:
                p = self.predict_proba(ex.features)
                err = p - float(ex.label)
                gb += err
                for k in self.feature_names:
                    grad[k] += err * float(ex.features.get(k, 0.0))
            n = float(len(data))
            for k in self.feature_names:
                self.weights[k] -= lr * (grad[k] / n + l2 * self.weights[k])
            self.bias -= lr * gb / n
        return self

    def predict_proba(self, features: dict[str, float]) -> float:
        z = self.bias + sum(self.weights.get(k, 0.0) * float(features.get(k, 0.0)) for k in self.feature_names)
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"weights": self.weights, "bias": self.bias, "feature_names": self.feature_names}, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "LogisticConfidenceCalibrator":
        data = json.loads(Path(path).read_text())
        return cls(weights={str(k): float(v) for k, v in data["weights"].items()}, bias=float(data["bias"]), feature_names=list(data["feature_names"]))


def calibration_features(final_result: Any, source_record: dict[str, Any] | None = None) -> dict[str, float]:
    source_record = source_record or {}
    support = getattr(final_result, "support", "supported" if getattr(final_result, "validated", False) else "unsupported")
    acoustic = source_record.get("acoustic") or {}
    rep = source_record.get("repetition") or {}
    return {
        "person2_score": float(source_record.get("score") or 0.0),
        "semantic_similarity": float(source_record.get("semantic_similarity") or 0.0),
        "asr_confidence": float(source_record.get("transcript_confidence") or 0.0),
        "acoustic_score": float(acoustic.get("agitation_score") or acoustic.get("scream_score") or 0.0),
        "repetition_count": float(rep.get("count") or 0.0),
        "verifier_supported": 1.0 if support == "supported" else 0.0,
        "verifier_insufficient": 1.0 if support == "insufficient" else 0.0,
        "model_support_score": float(getattr(final_result, "model_support_score", getattr(final_result, "confidence", 0.0)) or 0.0),
    }


def apply_calibration(final_result: Any, calibrator: LogisticConfidenceCalibrator | None, source_record: dict[str, Any] | None = None) -> Any:
    if calibrator is None:
        return final_result
    from dataclasses import replace
    support = getattr(final_result, "support", "supported" if getattr(final_result, "validated", False) else "unsupported")
    calibrated = 0.0 if support != "supported" else calibrator.predict_proba(calibration_features(final_result, source_record))
    return replace(final_result, calibrated_confidence=max(0.0, min(1.0, float(calibrated))))
