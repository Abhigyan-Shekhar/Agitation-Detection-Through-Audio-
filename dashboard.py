"""Audio + Linguistic Agitation Dashboard with local faster-whisper transcription.

Architecture overview
---------------------
Microphone (sounddevice)
    │
    ├─► transcription_queue ──► TranscriptionWorker ──► partial_queue  ─► live caption
    │                                               └──► committed_queue ─► UtteranceAggregator
    │                                                                          │
    └─► acoustic_queue ──► AcousticWorker                                      ▼
                               (rolling ring buffer)                 completed utterance_queue
                                     │                                         │
                                     └──────────────► ScoreFusion ◄────────────┘
                                                        │
                                                 BehaviourClassifier
                                                        │
                                                 Streamlit Dashboard
"""
from __future__ import annotations

import logging
import queue
import time
from datetime import date, datetime, time as datetime_time, timedelta
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

import config
from dashboard_manager import DashboardManager
from baseline_manager import BaselineManager
from linguistic_features import LinguisticAnalyzer
from score_fusion import ScoreFusion
from behaviour_classifier import BehaviourClassifier
from audio_behaviour_taxonomy import get_supported_behaviours
from audio_pipeline import LoudnessSnapshot

from behaviour_history import (
    DEFAULT_WINDOW_MINUTES,
    append_record_once,
    behaviour_breakdown,
    get_most_common_behaviour,
    get_recent_events,
    normalise_event_timestamp,
)

from event_models import (
    AcousticFeatureWindow,
    BehaviourEvent,
    FusedResult,
    LinguisticFeatures,
    Utterance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Agitation Dashboard", layout="wide")

# Minimal role options for dashboard role selectors. No shared USER_ROLES
# definition exists elsewhere in this project.
USER_ROLES: tuple[str, ...] = (
    "Care staff",
    "Clinician",
    "Administrator",
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init() -> None:
    defaults: dict[str, Any] = {
        "manager": None,
        "baseline_manager": None,
        "linguistic_analyzer": None,
        "score_fusion": None,
        "behaviour_classifier": None,
        # Queues
        "partial_queue": queue.Queue(maxsize=5),
        "committed_queue": queue.Queue(maxsize=100),
        "utterance_queue": queue.Queue(maxsize=50),
        # Display state
        "partial_caption": "",
        "transcription_metadata": {},
        "committed_lines": [],      # list[str]
        "timeline": [],             # list[dict]
        "behaviour_log": [],        # list[dict]
        "behaviour_event_keys": set(),
        "latest_result": None,      # FusedResult | None
        "last_acoustic_scream_ts": 0.0,
        "error": None,
        # Calibration
        "calibrating": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _ensure_services() -> None:
    """Create long-lived service objects once per session."""
    if st.session_state.baseline_manager is None:
        st.session_state.baseline_manager = BaselineManager()

    bm: BaselineManager = st.session_state.baseline_manager

    if st.session_state.linguistic_analyzer is None:
        st.session_state.linguistic_analyzer = LinguisticAnalyzer()

    if st.session_state.score_fusion is None:
        st.session_state.score_fusion = ScoreFusion(bm)

    if st.session_state.behaviour_classifier is None:
        st.session_state.behaviour_classifier = BehaviourClassifier()


def _get_manager() -> DashboardManager:
    """Return the single runtime manager stored in Streamlit session state."""
    manager = st.session_state.get("manager")
    if manager is None:
        manager = DashboardManager(
            partial_queue=st.session_state.partial_queue,
            committed_queue=st.session_state.committed_queue,
            utterance_queue=st.session_state.utterance_queue,
            baseline_manager=st.session_state.baseline_manager,
        )
        st.session_state.manager = manager
    return manager


def _pipeline_running() -> bool:
    """Return True when the live runtime manager has active microphone capture."""
    manager = st.session_state.get("manager")
    return bool(manager and manager.is_running)


@st.fragment(run_every=1.0)
def _render_baseline_calibration_panel() -> None:
    """Render auto-refreshing calibration controls and progress."""
    st.subheader("📐 Baseline Calibration")
    bm: BaselineManager | None = st.session_state.baseline_manager
    if bm is None:
        st.warning("Baseline manager is not initialised yet.")
        return

    pipeline_running = _pipeline_running()
    if bm.has_personal_baseline:
        st.success(f"Personal baseline set ({bm._personal_n} windows)")
        if st.button("Reset baseline"):
            bm.reset_calibration()
    elif bm.is_calibrating:
        progress = bm.calibration_progress
        st.progress(
            progress,
            text=(
                f"Calibrating… {int(progress * 100)}% "
                f"({bm.calibration_window_count}/{bm.minimum_windows_for_personal} windows)"
            ),
        )
        if st.button("Stop calibration"):
            ok = bm.stop_calibration()
            st.session_state.calibrating = False
            if ok:
                st.success("Baseline saved!")
            else:
                st.warning("Not enough data — keep recording and try again")
    else:
        st.info(f"No personal baseline. Collect ~{config.BASELINE_COLLECT_MIN:.0f} min of calm speech.")
        if st.button("Start calibration", disabled=not pipeline_running):
            bm.start_calibration()
            st.session_state.calibrating = True


def _start_pipeline() -> None:
    st.session_state.error = None
    st.session_state.partial_caption = ""
    _get_manager().start()


def _stop_pipeline() -> None:
    manager = st.session_state.get("manager")
    if manager is not None:
        manager.stop()
    st.session_state.manager = None



# ---------------------------------------------------------------------------
# Fragment — runs every 1 second
# ---------------------------------------------------------------------------

def _consume() -> None:
    """Drain queues and run analysis on completed utterances."""
    # Partial caption (display only — no analysis)
    try:
        while True:
            text = st.session_state.partial_queue.get_nowait()
            st.session_state.partial_caption = text
    except queue.Empty:
        pass

    # Do not drain committed_queue here: the UtteranceAggregator owns it.
    # Committed transcript display is updated from completed utterances below.

    # Completed utterances → full analysis pipeline
    manager = st.session_state.get("manager")
    acoustic_worker = manager.acoustic_worker if manager is not None else None
    analyzer: LinguisticAnalyzer = st.session_state.linguistic_analyzer
    fusion: ScoreFusion = st.session_state.score_fusion
    classifier: BehaviourClassifier = st.session_state.behaviour_classifier

    try:
        while True:
            utterance: Utterance = st.session_state.utterance_queue.get_nowait()
            logger.info("Processing utterance: %r", utterance.full_text[:60])
            logger.info(
                "BEHAVIOUR_TRACE dashboard_received_utterance transcript=%r start=%.3f end=%.3f utterance_q=%d",
                utterance.full_text,
                utterance.start_time,
                utterance.end_time,
                st.session_state.utterance_queue.qsize(),
            )

            trace = utterance.latency_trace

            # Aggregate acoustic features for this utterance's time span
            acoustic = None
            if acoustic_worker is not None:
                acoustic = acoustic_worker.aggregate(
                    utterance.start_time, utterance.end_time
                )
            if trace is not None:
                trace.feature_extraction_ts = time.monotonic()

            # Linguistic features
            linguistic = analyzer.analyze(utterance)
            logger.info(
                "BEHAVIOUR_TRACE dashboard_classifier_input transcript=%r acoustic_available=%s linguistic=%s",
                utterance.full_text,
                acoustic is not None,
                linguistic,
            )

            # Fusion
            result = fusion.fuse(utterance, acoustic, linguistic)
            result.linguistic_features = linguistic

            # Behaviour classification
            result = classifier.classify(result)
            logger.info(
                "BEHAVIOUR_TRACE dashboard_classifier_output transcript=%r behaviours=%s event_labels=%s severity=%s",
                utterance.full_text,
                result.behaviours,
                [event.canonical_label for event in result.behaviour_events],
                result.severity,
            )

            if result.latency_trace is not None:
                result.latency_trace.dashboard_render_ts = time.monotonic()
                logger.info("Dashboard latency diagnostics: %s", result.latency_trace.durations_ms())
            st.session_state.latest_result = result
            st.session_state.committed_lines.extend(line.text for line in utterance.lines)
            if len(st.session_state.committed_lines) > 50:
                st.session_state.committed_lines = st.session_state.committed_lines[-50:]
            for event in result.behaviour_events:
                append_record_once(
                    st.session_state.behaviour_log,
                    _event_to_record(event, result),
                    st.session_state.behaviour_event_keys,
                )
            st.session_state.timeline.append({
                "time": time.strftime("%H:%M:%S"),
                "timestamp": datetime.now(),
                "acoustic_score": result.acoustic_score,
                "linguistic_score": result.linguistic_score,
                "smoothed_score": result.smoothed_score,
                "severity": result.severity,
            })

            # Optional Gemini ablation
            if config.ENABLE_GEMINI_COMPARISON and acoustic is not None:
                try:
                    from person3_module import analyze_person3
                    gemini_result = analyze_person3(
                        transcript=utterance.full_text,
                        acoustic_features=acoustic.to_dict() if acoustic else {},
                    )
                    result.gemini_result = gemini_result
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Gemini comparison failed: %s", exc)

    except queue.Empty:
        pass

    pipeline = manager.pipeline if manager is not None else None
    _consume_acoustic_only_screaming(acoustic_worker, classifier, pipeline.latest_loudness if pipeline else None)
    _consume_acoustic_only_strange_noise(acoustic_worker, classifier)


def _acoustic_scream_score(acoustic: AcousticFeatureWindow | None) -> float:
    if acoustic is None:
        return 0.0
    energy_score = min(1.0, acoustic.rms_mean / max(config.BEHAVIOUR_ABSOLUTE_RMS_SHOUT, 1e-6))
    peak_score = min(1.0, acoustic.rms_max / max(config.BEHAVIOUR_ABSOLUTE_PEAK_SHOUT, 1e-6))
    clipping_score = min(1.0, acoustic.clipping_ratio / max(config.BEHAVIOUR_CLIPPING_SHOUT, 1e-6))
    if clipping_score > 0 and energy_score >= 0.65:
        return max(energy_score, peak_score, clipping_score)
    if energy_score >= 1.0 and peak_score >= 1.0:
        return max(energy_score, peak_score)
    return 0.0


def _loudness_scream_score(loudness: LoudnessSnapshot | None) -> float:
    if loudness is None:
        return 0.0
    age = time.time() - loudness.timestamp
    if age > 1.0:
        return 0.0
    rms_score = min(1.0, loudness.rms / max(config.BEHAVIOUR_ABSOLUTE_RMS_SHOUT, 1e-6))
    peak_score = min(1.0, loudness.peak / max(config.BEHAVIOUR_ABSOLUTE_PEAK_SHOUT, 1e-6))
    clipping_score = min(1.0, loudness.clipping_ratio / max(config.BEHAVIOUR_CLIPPING_SHOUT, 1e-6))
    if rms_score >= 1.0 and peak_score >= 0.75:
        return max(rms_score, peak_score)
    if clipping_score > 0 and rms_score >= 0.50:
        return max(rms_score, peak_score, clipping_score)
    return 0.0


def _consume_acoustic_only_screaming(
    acoustic_worker: Any,
    classifier: BehaviourClassifier,
    loudness: LoudnessSnapshot | None = None,
) -> None:
    acoustic = acoustic_worker.latest_window() if acoustic_worker is not None else None
    loudness_score = _loudness_scream_score(loudness)
    acoustic_score = _acoustic_scream_score(acoustic)
    scream_score = max(loudness_score, acoustic_score)
    if scream_score < 0.65:
        return

    now = time.time()
    last_ts = float(st.session_state.get("last_acoustic_scream_ts", 0.0) or 0.0)
    if now - last_ts < 2.0:
        return

    severity = "High" if scream_score >= 0.85 else "Moderate"
    acoustic_for_result = acoustic or AcousticFeatureWindow(
        start_time=(loudness.timestamp if loudness else now),
        end_time=(loudness.timestamp if loudness else now),
        rms_mean=(loudness.rms if loudness else 0.0),
        rms_max=(loudness.peak if loudness else 0.0),
        clipping_ratio=(loudness.clipping_ratio if loudness else 0.0),
        voiced_ratio=1.0 if loudness else 0.0,
    )
    result = FusedResult(
        acoustic_score=round(scream_score, 4),
        linguistic_score=0.0,
        raw_final_score=round(scream_score, 4),
        smoothed_score=round(scream_score, 4),
        severity=severity,
        reliability=max(0.0, 1.0 - (0.25 if acoustic_for_result.clipping_ratio > 0.10 else 0.0)),
        utterance=None,
        acoustic_features=acoustic_for_result,
        linguistic_features=LinguisticFeatures(),
        acoustic_contributions={
            "raw_callback_rms": round(loudness.rms if loudness else 0.0, 4),
            "raw_callback_peak": round(loudness.peak if loudness else 0.0, 4),
            "raw_callback_clipping": round(loudness.clipping_ratio if loudness else 0.0, 4),
            "absolute_rms": round(acoustic.rms_mean if acoustic else 0.0, 4),
            "absolute_peak": round(acoustic.rms_max if acoustic else 0.0, 4),
            "clipping": round(acoustic.clipping_ratio if acoustic else 0.0, 4),
        },
        linguistic_contributions={},
    )
    result = classifier.classify(result)
    if "Screaming" not in result.behaviours:
        return

    logger.info(
        "BEHAVIOUR_TRACE acoustic_only_screaming raw_rms=%.3f raw_peak=%.3f raw_clipping=%.3f window_rms=%.3f window_peak=%.3f window_clipping=%.3f labels=%s severity=%s",
        loudness.rms if loudness else 0.0,
        loudness.peak if loudness else 0.0,
        loudness.clipping_ratio if loudness else 0.0,
        acoustic.rms_mean if acoustic else 0.0,
        acoustic.rms_max if acoustic else 0.0,
        acoustic.clipping_ratio if acoustic else 0.0,
        result.behaviours,
        result.severity,
    )
    st.session_state.last_acoustic_scream_ts = now
    st.session_state.latest_result = result
    for event in result.behaviour_events:
        st.session_state.behaviour_log.append(_event_to_record(event, result))
    st.session_state.timeline.append({
        "time": time.strftime("%H:%M:%S"),
        "timestamp": datetime.now(),
        "acoustic_score": result.acoustic_score,
        "linguistic_score": result.linguistic_score,
        "smoothed_score": result.smoothed_score,
        "severity": result.severity,
    })


def _consume_acoustic_only_strange_noise(
    acoustic_worker: Any,
    classifier: BehaviourClassifier,
) -> None:
    acoustic = acoustic_worker.latest_window() if acoustic_worker is not None else None
    if acoustic is None:
        return

    score = acoustic.non_speech_vocalization_score
    if score < config.BEHAVIOUR_STRANGE_NOISE_THRESHOLD:
        return

    now = time.time()
    last_ts = float(st.session_state.get("last_acoustic_strange_noise_ts", 0.0) or 0.0)
    if now - last_ts < 2.0:
        return

    severity = "Moderate" if score >= 0.80 else "Mild"
    result = FusedResult(
        acoustic_score=round(score, 4),
        linguistic_score=0.0,
        raw_final_score=round(score, 4),
        smoothed_score=round(score, 4),
        severity=severity,
        reliability=0.85,
        utterance=None,
        acoustic_features=acoustic,
        linguistic_features=LinguisticFeatures(),
        acoustic_contributions={
            "non_speech_vocalization": round(score, 4),
            "rms": round(acoustic.rms_mean, 4),
            "peak": round(acoustic.rms_max, 4),
        },
        linguistic_contributions={},
    )
    result = classifier.classify(result)
    if "Making strange noises" not in result.behaviours:
        return

    logger.info(
        "BEHAVIOUR_TRACE acoustic_only_strange_noise score=%.3f label=%s evidence=%s labels=%s severity=%s",
        score,
        acoustic.non_speech_vocalization_label,
        acoustic.non_speech_vocalization_evidence,
        result.behaviours,
        result.severity,
    )
    st.session_state.last_acoustic_strange_noise_ts = now
    st.session_state.latest_result = result
    for event in result.behaviour_events:
        st.session_state.behaviour_log.append(_event_to_record(event, result))
    st.session_state.timeline.append({
        "time": time.strftime("%H:%M:%S"),
        "timestamp": datetime.now(),
        "acoustic_score": result.acoustic_score,
        "linguistic_score": result.linguistic_score,
        "smoothed_score": result.smoothed_score,
        "severity": result.severity,
    })


def _taxonomy_labels() -> list[str]:
    """Return canonical behaviour labels from the shared taxonomy."""
    return [entry.canonical_label for entry in get_supported_behaviours()]


def _severity_options() -> list[str]:
    return ["Low", "Medium", "High", "Critical"]


def _severity_badge(severity: str | None) -> str:
    colors = {
        "Low": "🟢 Low",
        "Mild": "🟡 Mild",
        "Medium": "🟡 Medium",
        "Moderate": "🟠 Moderate",
        "High": "🟠 High",
        "Critical": "🔴 Critical",
    }
    return colors.get(severity or "", f"⚪ {severity or 'Unknown'}")


def _event_to_record(event: BehaviourEvent, result: FusedResult | None = None) -> dict[str, Any]:
    """Convert a BehaviourEvent into a dashboard-friendly record."""
    timestamp = normalise_event_timestamp(event.timestamp) or datetime.now()
    return {
        "event_id": event.event_id,
        "timestamp": timestamp,
        "resident": event.person or "Unassigned resident",
        "behaviour": event.canonical_label or event.behaviour_type or "Unmapped audio behaviour",
        "severity": event.severity or (result.severity if result else "Low"),
        "reliability": result.reliability if result else None,
        "location": event.location or "Observation area",
        "duration": event.duration,
        "trigger": event.trigger or "",
        "intervention": event.intervention or "",
        "outcome": event.outcome or "",
        "notes": event.notes or "",
        "source": "Detected",
    }


def _records_dataframe(records: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """Build the canonical events DataFrame used by dashboard tabs."""
    data = records if records is not None else st.session_state.behaviour_log
    columns = [
        "event_id", "timestamp", "resident", "behaviour", "severity", "reliability", "location", "duration",
        "trigger", "intervention", "outcome", "notes", "source",
    ]
    if not data:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp", ascending=False)


def _sidebar_filters(df: pd.DataFrame) -> dict[str, Any]:
    """Render interactive filters and return selected values."""
    st.subheader("🔎 Filters")
    residents = sorted(df["resident"].dropna().unique().tolist()) if not df.empty else []
    behaviours = sorted(set(_taxonomy_labels()) | set(df["behaviour"].dropna().unique().tolist())) if not df.empty else _taxonomy_labels()
    severities = sorted(df["severity"].dropna().unique().tolist()) if not df.empty else _severity_options()
    locations = sorted(df["location"].dropna().unique().tolist()) if not df.empty else []
    today = date.today()
    return {
        "residents": st.multiselect("Resident", residents, default=residents, help="Limit dashboard cards, charts, and tables to selected residents."),
        "behaviours": st.multiselect("Behaviour", behaviours, default=behaviours),
        "severities": st.multiselect("Severity", severities, default=severities),
        "locations": st.multiselect("Location", locations, default=locations),
        "date_range": st.date_input("Date range", value=(today - timedelta(days=7), today)),
        "time_range": st.slider(
            "Time range",
            value=(datetime_time(0, 0), datetime_time(23, 59)),
            help="Filters events by local event time.",
        ),
        "search": st.text_input("Search notes/outcomes", placeholder="Type to search…"),
    }


def _default_filters(df: pd.DataFrame) -> dict[str, Any]:
    """Return safe filter defaults when the sidebar has not rendered yet."""
    today = date.today()
    return {
        "residents": sorted(df["resident"].dropna().unique().tolist()) if not df.empty else [],
        "behaviours": sorted(set(_taxonomy_labels()) | set(df["behaviour"].dropna().unique().tolist())) if not df.empty else _taxonomy_labels(),
        "severities": sorted(df["severity"].dropna().unique().tolist()) if not df.empty else _severity_options(),
        "locations": sorted(df["location"].dropna().unique().tolist()) if not df.empty else [],
        "date_range": (today - timedelta(days=7), today),
        "time_range": (datetime_time(0, 0), datetime_time(23, 59)),
        "search": "",
    }


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    for column, selected in [
        ("resident", filters["residents"]),
        ("behaviour", filters["behaviours"]),
        ("severity", filters["severities"]),
        ("location", filters["locations"]),
    ]:
        if selected:
            filtered = filtered[filtered[column].isin(selected)]
    date_range = filters["date_range"]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start_date)
            & (filtered["timestamp"].dt.date <= end_date)
        ]
    start_time, end_time = filters["time_range"]
    filtered = filtered[
        (filtered["timestamp"].dt.time >= start_time)
        & (filtered["timestamp"].dt.time <= end_time)
    ]
    search = filters["search"].strip().lower()
    if search:
        haystack = (
            filtered["notes"].fillna("") + " "
            + filtered["outcome"].fillna("") + " "
            + filtered["trigger"].fillna("")
        ).str.lower()
        filtered = filtered[haystack.str.contains(search, regex=False)]
    return filtered


_BASELINE_DEBUG_FEATURES: tuple[str, ...] = (
    "rms_mean",
    "rms_max",
    "rms_slope",
    "pitch_median",
    "pitch_range",
    "pitch_variance",
    "zcr_mean",
    "spectral_centroid",
    "voiced_ratio",
    "pause_ratio",
)


def _render_acoustic_baseline_debug() -> None:
    """Render live acoustic baseline diagnostics without changing scoring."""
    st.subheader("Acoustic Baseline Debug")
    manager = st.session_state.get("manager")
    acoustic_worker = manager.acoustic_worker if manager is not None else None
    bm: BaselineManager | None = st.session_state.baseline_manager
    fusion: ScoreFusion | None = st.session_state.score_fusion
    latest = acoustic_worker.latest_window() if acoustic_worker is not None else None
    result: FusedResult | None = st.session_state.latest_result

    if bm is None:
        st.warning("Baseline manager is not initialised.")
        return
    if latest is None:
        st.info("No acoustic feature window has been extracted yet.")
        return

    st.caption(
        "Live diagnostics for the latest acoustic window. Values are read-only "
        "and do not alter scoring, thresholds, or calibration."
    )
    status_cols = st.columns(4)
    status_cols[0].metric("Personal baseline", "Active" if bm.has_personal_baseline else "Rolling fallback")
    status_cols[1].metric("Calibration windows", bm.calibration_window_count)
    status_cols[2].metric("Latest window age", f"{(time.time() - latest.end_time):.2f}s")
    if acoustic_worker is not None:
        status_cols[3].metric("Pending extractions", acoustic_worker.pending_extractions)

    raw_rows = [
        {"feature": feat, "raw": round(float(getattr(latest, feat, 0.0)), 6)}
        for feat in _BASELINE_DEBUG_FEATURES
    ]
    st.markdown("**A. Raw acoustic features**")
    st.dataframe(pd.DataFrame(raw_rows), hide_index=True, use_container_width=True)

    personal_stats = bm.personal_baseline_stats()
    baseline_rows = []
    for feat in _BASELINE_DEBUG_FEATURES:
        mean, std = personal_stats.get(feat, (None, None))
        baseline_rows.append({
            "feature": feat,
            "personal_mean": None if mean is None else round(float(mean), 6),
            "personal_std": None if std is None else round(float(std), 6),
        })
    st.markdown("**B. Personal baseline**")
    if personal_stats:
        st.dataframe(pd.DataFrame(baseline_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No personal baseline is active yet; z-scores currently use the rolling fallback when enough rolling data exists.")

    z_rows = [
        {
            "feature": feat,
            "z_score": round(float(bm.z_score(feat, float(getattr(latest, feat, 0.0)))), 4),
        }
        for feat in _BASELINE_DEBUG_FEATURES
    ]
    st.markdown("**C. Z-scores from BaselineManager.z_score()**")
    st.dataframe(pd.DataFrame(z_rows), hide_index=True, use_container_width=True)

    debug_values = fusion.acoustic_debug_values(latest) if fusion is not None else {"score": 0.0, "z_scores": {}, "branch_values": {}}
    branch_z = debug_values.get("z_scores", {})
    branch_values = debug_values.get("branch_values", {})
    st.markdown("**D. Acoustic branch values used by score_fusion.py**")
    branch_rows = [
        {"name": name, "value": value}
        for name, value in {**branch_z, **branch_values}.items()
    ]
    st.dataframe(pd.DataFrame(branch_rows), hide_index=True, use_container_width=True)

    st.markdown("**E. Final scores**")
    score_cols = st.columns(5)
    score_cols[0].metric("Latest acoustic branch", debug_values.get("score", 0.0))
    if result is not None:
        score_cols[1].metric("Result acoustic", result.acoustic_score)
        score_cols[2].metric("Result linguistic", result.linguistic_score)
        score_cols[3].metric("Fused agitation", result.smoothed_score)
        score_cols[4].metric("Reliability", result.reliability)
        st.caption(f"Severity: {result.severity}")
    else:
        score_cols[1].metric("Result acoustic", "N/A")
        score_cols[2].metric("Result linguistic", "N/A")
        score_cols[3].metric("Fused agitation", "N/A")
        score_cols[4].metric("Reliability", "N/A")


def _render_summary_cards(df: pd.DataFrame) -> None:
    today_df = df[df["timestamp"].dt.date == date.today()] if not df.empty else df
    high_df = df[df["severity"].isin(["High", "Critical"])] if not df.empty else df
    most_common = "—" if df.empty else df["behaviour"].mode().iat[0]
    active_resident = "—" if df.empty else df["resident"].mode().iat[0]
    avg_severity = "—"
    if not df.empty:
        weights = {"Low": 1, "Mild": 1.5, "Medium": 2, "Moderate": 2.5, "High": 3, "Critical": 4}
        avg_severity = f"{df['severity'].map(weights).fillna(0).mean():.1f}/4"
    cols = st.columns(6)
    cols[0].metric("Today's Events", len(today_df), help="Events recorded today after filters.")
    cols[1].metric("High Severity", len(high_df), help="High and critical events.")
    cols[2].metric("Most Common", most_common)
    cols[3].metric("Avg Severity", avg_severity)
    cols[4].metric("Most Active Resident", active_resident)
    cols[5].metric("System Status", "Running" if _pipeline_running() else "Stopped")


def _render_empty(message: str) -> None:
    st.info(f"ℹ️ {message}")


def _render_timeline(df: pd.DataFrame) -> None:
    st.subheader("🕒 Behaviour Timeline")
    if df.empty:
        _render_empty("No matching records for the selected filters.")
        return
    for _, row in df.sort_values("timestamp", ascending=False).head(12).iterrows():
        st.markdown(
            f"**{row['timestamp'].strftime('%H:%M')}** &nbsp; "
            f"{_severity_badge(row['severity'])} — **{row['behaviour']}**  \n"
            f"<span style='color:#667085'>{row['resident']} · {row['location']}</span>",
            unsafe_allow_html=True,
        )


def _bar_chart(df: pd.DataFrame, column: str, title: str) -> None:
    st.subheader(title)
    if df.empty or df[column].dropna().empty:
        _render_empty(f"No data available for {title.lower()}.")
        return
    counts = df[column].value_counts().rename_axis(column).reset_index(name="events")
    st.bar_chart(counts, x=column, y="events")


def _render_charts(df: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    with c1:
        _bar_chart(df, "behaviour", "Behaviour Frequency")
        _bar_chart(df, "location", "Events by Location")
    with c2:
        _bar_chart(df, "severity", "Severity Distribution")
        _bar_chart(df, "resident", "Resident Breakdown")
    st.subheader("Events per Hour")
    if df.empty:
        _render_empty("No hourly event data for the selected filters.")
    else:
        hourly = df.assign(hour=df["timestamp"].dt.hour).groupby("hour").size().reset_index(name="events")
        st.line_chart(hourly, x="hour", y="events")


def _render_rolling_behaviour_history() -> None:
    window_minutes = DEFAULT_WINDOW_MINUTES
    st.subheader(f"Behaviours detected in last {window_minutes} minutes")

    now = datetime.now()
    recent_events = get_recent_events(
        st.session_state.behaviour_log,
        window_minutes=window_minutes,
        now=now,
    )
    most_common_behaviour, most_common_count = get_most_common_behaviour(recent_events)
    breakdown_rows = behaviour_breakdown(recent_events)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total events", len(recent_events))
    c2.metric("Most repeated behaviour", most_common_behaviour or "None")
    c3.metric("Occurrences", most_common_count)

    if not recent_events:
        _render_empty(f"No behaviours detected in the last {window_minutes} minutes.")
        return

    if most_common_behaviour:
        st.markdown("**Most Repeated Behaviour**")
        st.metric(most_common_behaviour.upper(), f"{most_common_count} occurrences")

    st.markdown("**Behaviour breakdown**")
    breakdown_df = pd.DataFrame(breakdown_rows)
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    st.markdown("**30-minute history graph**")
    recent_df = pd.DataFrame(recent_events)
    recent_df["timestamp"] = pd.to_datetime(recent_df["timestamp"], errors="coerce")
    recent_df = recent_df.dropna(subset=["timestamp"])
    if recent_df.empty:
        _render_empty("No timestamped behaviour events are available for the graph.")
        return

    graph_df = (
        recent_df
        .assign(time_bucket=recent_df["timestamp"].dt.floor("5min"))
        .groupby(["time_bucket", "behaviour"])
        .size()
        .reset_index(name="events")
        .sort_values("time_bucket")
    )
    st.bar_chart(graph_df, x="time_bucket", y="events", color="behaviour")


def _render_recent_events(df: pd.DataFrame) -> None:
    st.subheader("📋 Recent Behaviour Events")
    if df.empty:
        _render_empty("No behaviour events recorded today." if not st.session_state.behaviour_log else "No matching records.")
        return
    view = df[["timestamp", "resident", "behaviour", "severity", "location", "outcome", "notes", "source"]].copy()
    view["timestamp"] = view["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(view, use_container_width=True, hide_index=True)


def _render_logging_form() -> None:
    st.subheader("➕ Behaviour Logging Panel")
    st.caption("Record caregiver observations without changing the live audio detection pipeline.")
    behaviours = _taxonomy_labels()
    with st.form("behaviour_log_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        resident = c1.text_input("Resident / Person", placeholder="Resident name or room", help="Use your facility's preferred identifier.")
        behaviour = c2.selectbox("Behaviour", behaviours)
        severity = c3.radio("Severity", _severity_options(), horizontal=True)

        c4, c5, c6 = st.columns(3)
        event_date = c4.date_input("Date", value=date.today())
        event_time = c5.time_input("Time", value=datetime.now().time().replace(second=0, microsecond=0))
        location = c6.selectbox("Location", ["Bedroom", "Dining room", "Hallway", "Activity room", "Bathroom", "Observation area", "Other"])

        c7, c8 = st.columns(2)
        duration = c7.slider("Duration (minutes)", min_value=0, max_value=120, value=5)
        trigger = c8.text_input("Trigger", placeholder="e.g., care activity, noise, unknown")

        interventions = st.multiselect(
            "Intervention",
            ["Reassurance", "Redirection", "Quiet space", "Pain check", "Hydration/snack", "Medication review", "Family contact"],
            help="Select any response attempted by staff.",
        )
        if severity in {"High", "Critical"}:
            emergency = st.text_area("Emergency Intervention", placeholder="Document urgent steps taken and who was notified.")
        else:
            emergency = ""
        if "aggression" in behaviour.lower():
            target = st.text_input("Target of aggression", placeholder="Person, staff role, object, or unknown")
        else:
            target = ""
        outcome = ""
        if interventions or emergency:
            outcome = st.radio("Outcome", ["Resolved", "Improved", "Unchanged", "Escalated"], horizontal=True)
        notes = st.text_area("Notes", placeholder="Add concise clinical context.")

        submitted = st.form_submit_button("Save behaviour event")
        if submitted:
            timestamp = datetime.combine(event_date, event_time)
            extra_notes = notes
            if emergency:
                extra_notes = f"{extra_notes}\nEmergency intervention: {emergency}".strip()
            if target:
                extra_notes = f"{extra_notes}\nTarget of aggression: {target}".strip()
            record = {
                "event_id": f"manual-{uuid4().hex[:8]}",
                "timestamp": timestamp,
                "resident": resident or "Unassigned resident",
                "behaviour": behaviour,
                "severity": severity,
                "reliability": None,
                "location": location,
                "duration": duration,
                "trigger": trigger,
                "intervention": ", ".join(interventions),
                "outcome": outcome,
                "notes": extra_notes,
                "source": "Manual",
            }
            append_record_once(
                st.session_state.behaviour_log,
                record,
                st.session_state.behaviour_event_keys,
            )
            st.success("Behaviour event saved.")



def _displayed_behaviour_label(result: FusedResult) -> str:
    """Return the label shown in the current-behaviour card and log UI selection."""
    event_labels = [event.canonical_label for event in result.behaviour_events]
    candidate_labels = event_labels or [
        label for label in result.behaviours if label != "No audio agitation detected"
    ] or result.behaviours
    displayed = candidate_labels[0] if candidate_labels else "No audio agitation detected"
    logger.info(
        "BEHAVIOUR_TRACE ui_display transcript=%r behaviours=%s event_labels=%s severity=%s displayed_behavior=%r",
        result.utterance.full_text if result.utterance else "",
        result.behaviours,
        event_labels,
        result.severity,
        displayed,
    )
    return displayed

def _render_behaviour_events(result: FusedResult) -> None:
    """Render canonical behaviour events in a compact, research-friendly layout."""
    if result.behaviour_events:
        st.subheader("Detected Behaviours")
        for event in result.behaviour_events:
            with st.container():
                st.markdown(f"**{event.canonical_label}**")
                details: list[str] = []
                if event.internal_code:
                    details.append(f"Internal code: {event.internal_code}")
                if event.cmai_category:
                    details.append(f"CMAI: {event.cmai_category}")
                if event.mapping_status:
                    details.append(f"Mapping status: {event.mapping_status}")
                if event.timestamp is not None:
                    details.append(f"Timestamp: {event.timestamp}")
                if event.severity:
                    details.append(f"Severity: {event.severity}")
                if event.duration is not None:
                    details.append(f"Duration: {event.duration}")
                if event.raw_detected_behaviour:
                    details.append(f"Raw observation: {event.raw_detected_behaviour}")
                if event.notes:
                    details.append(f"Notes: {event.notes}")
                st.caption(" • ".join(details))
        return

    st.subheader("Detected Behaviours")
    non_event_labels = [label for label in result.behaviours if label != "No audio agitation detected"]
    if non_event_labels:
        for label in non_event_labels:
            st.info(label)
        return
    st.success("No audio agitation detected")


def _render() -> None:
    """Render the main dashboard from session state."""
    df = _records_dataframe()
    filters = st.session_state.get("dashboard_filters")
    if filters is None:
        filters = _default_filters(df)
    filtered_df = _apply_filters(df, filters)
    result: FusedResult | None = st.session_state.latest_result

    overview_tab, log_tab, analytics_tab, history_tab, settings_tab = st.tabs(
        ["Overview", "Behaviour Log", "Analytics", "History", "Settings"]
    )

    with overview_tab:
        _render_summary_cards(filtered_df)
        st.divider()
        st.subheader("🩺 System Status")
        status_cols = st.columns([1, 1, 2])
        status_cols[0].success("Microphone active" if _pipeline_running() else "Monitoring stopped")
        status_cols[1].caption("Local decision support only — not a clinical diagnosis.")
        status_cols[2].progress(1.0 if _pipeline_running() else 0.0, text="Audio pipeline status")
        if result is not None and result.latency_trace is not None:
            with st.expander("⏱️ Latency diagnostics", expanded=False):
                latency = result.latency_trace.durations_ms()
                st.json(latency if latency else {"status": "waiting for complete trace"})
        with st.expander("Acoustic Baseline Debug", expanded=False):
            _render_acoustic_baseline_debug()

        live_col, current_col = st.columns([1, 1])
        with live_col:
            st.subheader("🎙️ Current Recording")
            partial = st.session_state.partial_caption or "_Waiting for speech…_"
            st.markdown(f"> {partial}")
            manager = st.session_state.get("manager")
            worker = manager.transcription_worker if manager is not None else None
            tx = worker.latest_result if worker is not None else None
            if tx is not None:
                meta_cols = st.columns(3)
                meta_cols[0].metric("Transcript time", datetime.fromtimestamp(tx.timestamp).strftime("%H:%M:%S"))
                meta_cols[1].metric("Confidence", "N/A" if tx.confidence is None else f"{tx.confidence:.0%}")
                meta_cols[2].metric("Inference", f"{tx.inference_ms:.0f} ms")
            committed = st.session_state.committed_lines
            with st.expander("📝 Current Transcript", expanded=bool(committed)):
                st.write("  \n".join(committed[-20:]) if committed else "No committed transcript yet.")

        with current_col:
            st.subheader("🧭 Current Behaviour")
            if result is None:
                st.info("Waiting for a completed utterance…")
            else:
                behaviour_label = _displayed_behaviour_label(result)
                st.metric("Behaviour", behaviour_label)
                st.metric("Current Severity", _severity_badge(result.severity))
                st.metric("Current Confidence", f"{result.reliability:.0%}")
                with st.expander("Detected behaviour details", expanded=True):
                    _render_behaviour_events(result)

        st.divider()
        _render_timeline(filtered_df)
        st.divider()
        _render_recent_events(filtered_df.head(10))

    with log_tab:
        _render_logging_form()

    with analytics_tab:
        _render_rolling_behaviour_history()
        st.divider()
        _render_summary_cards(filtered_df)
        _render_charts(filtered_df)
        if st.session_state.timeline:
            st.subheader("📈 Detection Score Timeline")
            score_df = pd.DataFrame(st.session_state.timeline).set_index("time")
            st.line_chart(score_df[["acoustic_score", "linguistic_score", "smoothed_score"]])

        if result is not None:
            with st.expander("Why was the current behaviour detected?", expanded=False):
                all_contributions = {
                    **result.acoustic_contributions,
                    **{k: v for k, v in result.linguistic_contributions.items() if not k.startswith("[")},
                }
                if all_contributions:
                    max_val = max(abs(v) for v in all_contributions.values()) or 1.0
                    for feature, contrib in sorted(all_contributions.items(), key=lambda x: -abs(x[1])):
                        bar_len = int(abs(contrib) / max_val * 20)
                        st.text(f"  {feature:<35} {'+' if contrib >= 0 else '-'}{abs(contrib):.4f}  {'█' * bar_len}")
                else:
                    _render_empty("No explainability contributions are available yet.")

    with history_tab:
        _render_recent_events(filtered_df)
        if result and result.utterance:
            with st.expander("Analysed utterance text"):
                st.write(result.utterance.full_text)
        if config.ENABLE_GEMINI_COMPARISON and result and result.gemini_result:
            with st.expander("🤖 Gemini comparison (ablation)", expanded=False):
                st.json(result.gemini_result)

    with settings_tab:
        st.subheader("⚙️ Settings")
        st.info("Settings are a placeholder for future caregiver preferences and facility configuration.")
        st.caption("Detection pipeline configuration remains unchanged by this dashboard redesign.")


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

_init()
_ensure_services()

# ---- Sidebar ------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Controls")
    st.session_state.dashboard_role = st.selectbox(
        "Dashboard role",
        list(USER_ROLES),
        index=list(USER_ROLES).index(st.session_state.get("dashboard_role", "Care staff")),
        help="Controls whether manual behaviour events can be added.",
    )

    pipeline_running = _pipeline_running()
    col_start, col_stop = st.columns(2)

    if col_start.button("▶ Start mic", disabled=pipeline_running):
        st.session_state.error = None
        try:
            _start_pipeline()
        except Exception as exc:
            logger.exception("Failed to start pipeline")
            st.session_state.error = str(exc)

    if col_stop.button("⏹ Stop mic", disabled=not pipeline_running):
        _stop_pipeline()

    st.divider()

    # Baseline calibration
    bm: BaselineManager | None = st.session_state.baseline_manager
    _render_baseline_calibration_panel()

    st.divider()
    st.session_state.dashboard_filters = _sidebar_filters(_records_dataframe())

    st.divider()

    # Debug
    with st.expander("🔧 Debug"):
        st.write("Pipeline running:", pipeline_running)
        st.write("Transcription engine:", config.TRANSCRIPTION_ENGINE)
        st.write("Whisper model:", config.WHISPER_MODEL)
        st.write("Transcription window (s):", config.TRANSCRIPTION_WINDOW_SECONDS)
        st.write("Transcription interval (s):", config.TRANSCRIPTION_INTERVAL_SECONDS)
        st.write("Use GPU if available:", config.USE_GPU_IF_AVAILABLE)
        st.write("Gemini comparison:", config.ENABLE_GEMINI_COMPARISON)
        manager = st.session_state.get("manager")
        aw = manager.acoustic_worker if manager else None
        if aw:
            st.write("Acoustic windows extracted:", aw.windows_extracted)
            st.write("Last acoustic extraction (ms):", round(aw.last_extraction_ms, 2))
            st.write("Average acoustic extraction (ms):", round(aw.average_extraction_ms, 2))
            st.write("Acoustic extractions scheduled:", aw.windows_scheduled)
            st.write("Pending acoustic extractions:", aw.pending_extractions)
            st.write("Skipped acoustic windows (backpressure):", aw.windows_skipped_backpressure)
            latest = aw.latest_window()
            if latest:
                st.write("Latest acoustic window age (ms):", round((time.time() - latest.end_time) * 1000.0, 2))
        ua = manager.utterance_aggregator if manager else None
        if ua:
            st.write("Utterances emitted:", ua.emitted_count)
        if bm:
            st.write("Baseline manager id:", id(bm))
            st.write("Calibration active:", bm.is_calibrating)
            st.write("Calibration windows:", bm.calibration_window_count)
            st.write("Calibration min windows:", bm.minimum_windows_for_personal)
            st.write("Calibration progress:", round(bm.calibration_progress * 100.0, 1))
            st.write("Rolling baseline windows:", len(bm._rolling))
            if manager:
                st.write("Manager baseline id:", id(manager._baseline_manager))
                st.write("Manager uses session baseline:", manager._baseline_manager is bm)

# ---- Error banner --------------------------------------------------------
if st.session_state.error:
    st.error(st.session_state.error)

# ---- Main title ----------------------------------------------------------
st.title("🔊 Audio + Linguistic Agitation Dashboard")
st.caption(
    "Local, real-time, explainable audio-linguistic cue detection. "
    "CMAI-inspired labels are decision support only — not a clinical diagnosis."
)


# ---- Live fragment (polls every second) ----------------------------------
@st.fragment(run_every=1.0)
def _live() -> None:
    _consume()
    _render()


_live()
