"""Audio + Linguistic Agitation Dashboard — WhisperLiveKit edition.

Architecture overview
---------------------
Microphone (sounddevice)
    │
    ├─► wlk_queue ──► WhisperLiveKitClient ──► partial_queue  ─► live caption
    │                                      └──► committed_queue ─► UtteranceAggregator
    │                                                                     │
    └─► acoustic_queue ──► AcousticWorker                                 ▼
                               (rolling ring buffer)            completed utterance_queue
                                     │                                    │
                                     └─────────► ScoreFusion ◄────────────┘
                                                     │
                                              BehaviourClassifier
                                                     │
                                              Streamlit Dashboard

WLK server is launched as a subprocess if WLK_AUTO_LAUNCH=true (default).
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import time
from datetime import date, datetime, time as datetime_time, timedelta
from typing import Any

import pandas as pd
import streamlit as st

import config
from audio_pipeline import AudioPipeline
from whisperlivekit_client import WhisperLiveKitClient
from utterance_aggregator import UtteranceAggregator
from acoustic_features import AcousticWorker
from baseline_manager import BaselineManager
from linguistic_features import LinguisticAnalyzer
from score_fusion import ScoreFusion
from behaviour_classifier import BehaviourClassifier
from audio_behaviour_taxonomy import get_supported_behaviours
from event_models import BehaviourEvent, FusedResult, Utterance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Agitation Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init() -> None:
    defaults: dict[str, Any] = {
        "pipeline": None,
        "wlk_client": None,
        "utterance_aggregator": None,
        "acoustic_worker": None,
        "baseline_manager": None,
        "linguistic_analyzer": None,
        "score_fusion": None,
        "behaviour_classifier": None,
        "wlk_proc": None,
        # Queues
        "partial_queue": queue.Queue(maxsize=5),
        "committed_queue": queue.Queue(maxsize=100),
        "utterance_queue": queue.Queue(maxsize=50),
        # Display state
        "partial_caption": "",
        "committed_lines": [],      # list[str]
        "timeline": [],             # list[dict]
        "behaviour_log": [],        # list[dict]
        "latest_result": None,      # FusedResult | None
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


def _start_wlk_server() -> subprocess.Popen | None:
    """Launch WhisperLiveKit server as a subprocess if auto-launch is enabled."""
    if not config.WLK_AUTO_LAUNCH:
        return None
    cmd = [
        sys.executable, "-m", "whisperlivekit.server",
        "--backend", config.WLK_BACKEND,
        "--model", config.WLK_MODEL,
        "--lan", config.WLK_LANGUAGE,
        "--pcm-input",
        "--host", config.WLK_HOST,
        "--port", str(config.WLK_PORT),
    ]
    logger.info("Launching WLK server: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        time.sleep(2.5)     # give the server a moment to start
        return proc
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not auto-launch WLK: %s", exc)
        st.session_state.error = (
            f"Could not auto-launch WhisperLiveKit: {exc}\n"
            f"Start it manually: wlk --backend {config.WLK_BACKEND} "
            f"--model {config.WLK_MODEL} --lan {config.WLK_LANGUAGE} --pcm-input"
        )
        return None


def _start_pipeline() -> None:
    _ensure_services()

    # 1. WLK server
    if st.session_state.wlk_proc is None:
        st.session_state.wlk_proc = _start_wlk_server()

    # 2. Audio pipeline (fan-out)
    pipeline = AudioPipeline()

    # 3. WLK client
    wlk_client = WhisperLiveKitClient(
        wlk_queue=pipeline.wlk_queue,
        partial_queue=st.session_state.partial_queue,
        committed_queue=st.session_state.committed_queue,
    )

    # 4. Utterance aggregator
    aggregator = UtteranceAggregator(
        committed_queue=st.session_state.committed_queue,
        utterance_queue=st.session_state.utterance_queue,
    )

    # 5. Acoustic worker
    acoustic_worker = AcousticWorker(acoustic_queue=pipeline.acoustic_queue)
    # Feed new windows into baseline manager automatically
    bm: BaselineManager = st.session_state.baseline_manager

    def _patched_run():
        import time as _t
        while not acoustic_worker._stop_event.is_set():
            acoustic_worker._drain_queue()
            now = _t.time()
            if now - acoustic_worker._last_extraction_time >= acoustic_worker._hop_sec:
                records = acoustic_worker._ring.latest_window(acoustic_worker._window_sec)
                if records:
                    import time as tt
                    window_end = now
                    window_start = window_end - acoustic_worker._window_sec
                    feat = acoustic_worker._extractor.extract(records, window_start, window_end)
                    with acoustic_worker._lock:
                        acoustic_worker._windows.append(feat)
                    acoustic_worker._last_extraction_time = now
                    acoustic_worker._windows_extracted += 1
                    bm.feed(feat)
            _t.sleep(0.010)

    import threading
    acoustic_worker._thread = threading.Thread(
        target=_patched_run, name="acoustic-worker", daemon=True
    )

    # Start everything
    pipeline.start()
    wlk_client.start()
    aggregator.start()
    acoustic_worker._stop_event.clear()
    acoustic_worker._thread.start()

    st.session_state.pipeline = pipeline
    st.session_state.wlk_client = wlk_client
    st.session_state.utterance_aggregator = aggregator
    st.session_state.acoustic_worker = acoustic_worker
    st.session_state.score_fusion.reset()
    logger.info("All pipeline components started")


def _stop_pipeline() -> None:
    for key, attr in [
        ("utterance_aggregator", "stop"),
        ("wlk_client", "stop"),
        ("pipeline", "stop"),
        ("acoustic_worker", "stop"),
    ]:
        obj = st.session_state.get(key)
        if obj is not None:
            try:
                getattr(obj, attr)()
            except Exception:  # noqa: BLE001
                pass
            st.session_state[key] = None

    proc = st.session_state.get("wlk_proc")
    if proc is not None:
        proc.terminate()
        st.session_state.wlk_proc = None

    logger.info("All pipeline components stopped")


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

    # Committed lines (for the committed transcript display)
    try:
        while True:
            from event_models import CommittedLine
            line: CommittedLine = st.session_state.committed_queue.get_nowait()
            st.session_state.committed_lines.append(line.text)
            if len(st.session_state.committed_lines) > 50:
                st.session_state.committed_lines.pop(0)
    except queue.Empty:
        pass

    # Completed utterances → full analysis pipeline
    acoustic_worker: AcousticWorker | None = st.session_state.acoustic_worker
    analyzer: LinguisticAnalyzer = st.session_state.linguistic_analyzer
    fusion: ScoreFusion = st.session_state.score_fusion
    classifier: BehaviourClassifier = st.session_state.behaviour_classifier

    try:
        while True:
            utterance: Utterance = st.session_state.utterance_queue.get_nowait()
            logger.info("Processing utterance: %r", utterance.full_text[:60])

            # Aggregate acoustic features for this utterance's time span
            acoustic = None
            if acoustic_worker is not None:
                acoustic = acoustic_worker.aggregate(
                    utterance.start_time, utterance.end_time
                )

            # Linguistic features
            linguistic = analyzer.analyze(utterance)

            # Fusion
            result = fusion.fuse(utterance, acoustic, linguistic)
            result.linguistic_features = linguistic

            # Behaviour classification
            result = classifier.classify(result)

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
    timestamp = event.timestamp
    if not isinstance(timestamp, datetime):
        timestamp = datetime.now()
    return {
        "timestamp": timestamp,
        "resident": event.person or "Unassigned resident",
        "behaviour": event.canonical_label or event.behaviour_type or "Unmapped audio behaviour",
        "severity": event.severity or (result.severity if result else "Low"),
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
        "timestamp", "resident", "behaviour", "severity", "location", "duration",
        "trigger", "intervention", "outcome", "notes", "source",
    ]
    if not data:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp", ascending=False)


def _sidebar_filters(df: pd.DataFrame) -> dict[str, Any]:
    """Render interactive filters and return selected values."""
    st.sidebar.subheader("🔎 Filters")
    residents = sorted(df["resident"].dropna().unique().tolist()) if not df.empty else []
    behaviours = sorted(set(_taxonomy_labels()) | set(df["behaviour"].dropna().unique().tolist())) if not df.empty else _taxonomy_labels()
    severities = sorted(df["severity"].dropna().unique().tolist()) if not df.empty else _severity_options()
    locations = sorted(df["location"].dropna().unique().tolist()) if not df.empty else []
    today = date.today()
    return {
        "residents": st.sidebar.multiselect("Resident", residents, default=residents, help="Limit dashboard cards, charts, and tables to selected residents."),
        "behaviours": st.sidebar.multiselect("Behaviour", behaviours, default=behaviours),
        "severities": st.sidebar.multiselect("Severity", severities, default=severities),
        "locations": st.sidebar.multiselect("Location", locations, default=locations),
        "date_range": st.sidebar.date_input("Date range", value=(today - timedelta(days=7), today)),
        "time_range": st.sidebar.slider(
            "Time range",
            value=(datetime_time(0, 0), datetime_time(23, 59)),
            help="Filters events by local event time.",
        ),
        "search": st.sidebar.text_input("Search notes/outcomes", placeholder="Type to search…"),
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
    cols[5].metric("System Status", "Running" if st.session_state.pipeline is not None else "Stopped")


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
            st.session_state.behaviour_log.append({
                "timestamp": timestamp,
                "resident": resident or "Unassigned resident",
                "behaviour": behaviour,
                "severity": severity,
                "location": location,
                "duration": duration,
                "trigger": trigger,
                "intervention": ", ".join(interventions),
                "outcome": outcome,
                "notes": extra_notes,
                "source": "Manual",
            })
            st.success("Behaviour event saved.")


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
    st.success("No audio agitation detected")


def _render() -> None:
    """Render the main dashboard from session state."""
    df = _records_dataframe()
    filters = _sidebar_filters(df)
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
        status_cols[0].success("Microphone active" if st.session_state.pipeline is not None else "Monitoring stopped")
        status_cols[1].caption("Local decision support only — not a clinical diagnosis.")
        status_cols[2].progress(1.0 if st.session_state.pipeline is not None else 0.0, text="Audio pipeline status")

        live_col, current_col = st.columns([1, 1])
        with live_col:
            st.subheader("🎙️ Current Recording")
            partial = st.session_state.partial_caption or "_Waiting for speech…_"
            st.markdown(f"> {partial}")
            committed = st.session_state.committed_lines
            with st.expander("📝 Current Transcript", expanded=bool(committed)):
                st.write("  \n".join(committed[-20:]) if committed else "No committed transcript yet.")

        with current_col:
            st.subheader("🧭 Current Behaviour")
            if result is None:
                st.info("Waiting for a completed utterance…")
            else:
                behaviour_label = result.behaviour_events[0].canonical_label if result.behaviour_events else "No audio agitation detected"
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

    pipeline_running = st.session_state.pipeline is not None
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
    st.subheader("📐 Baseline Calibration")
    bm: BaselineManager | None = st.session_state.baseline_manager
    if bm:
        if bm.has_personal_baseline:
            st.success(f"Personal baseline set ({bm._personal_n} windows)")
            if st.button("Reset baseline"):
                bm.reset_calibration()
        elif bm.is_calibrating:
            progress = bm.calibration_progress
            st.progress(progress, text=f"Calibrating… {int(progress * 100)}%")
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

    st.divider()

    # Debug
    with st.expander("🔧 Debug"):
        st.write("Pipeline running:", pipeline_running)
        st.write("WLK auto-launch:", config.WLK_AUTO_LAUNCH)
        st.write("WLK backend:", config.WLK_BACKEND)
        st.write("WLK model:", config.WLK_MODEL)
        st.write("Gemini comparison:", config.ENABLE_GEMINI_COMPARISON)
        aw = st.session_state.acoustic_worker
        if aw:
            st.write("Acoustic windows extracted:", aw.windows_extracted)
        ua = st.session_state.utterance_aggregator
        if ua:
            st.write("Utterances emitted:", ua.emitted_count)
        if bm:
            st.write("Rolling baseline windows:", len(bm._rolling))

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
    if st.session_state.pipeline is not None:
        _consume()
    _render()


_live()
