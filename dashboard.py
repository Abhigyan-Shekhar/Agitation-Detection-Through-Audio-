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
from event_models import FusedResult, Utterance

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
    _original_run = acoustic_worker._run

    def _patched_run():
        import time as _t
        bm: BaselineManager = st.session_state.baseline_manager
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
            st.session_state.timeline.append({
                "time": time.strftime("%H:%M:%S"),
                "acoustic_score": result.acoustic_score,
                "linguistic_score": result.linguistic_score,
                "smoothed_score": result.smoothed_score,
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


def _render() -> None:
    """Render the main dashboard from session state."""
    # ---- Live caption ---------------------------------------------------
    st.subheader("🎙️ Live Caption")
    partial = st.session_state.partial_caption or "_Waiting for speech…_"
    st.markdown(f"> {partial}")

    # ---- Committed transcript ------------------------------------------
    committed = st.session_state.committed_lines
    if committed:
        with st.expander("📝 Committed Transcript (last 50 lines)", expanded=False):
            st.write("  \n".join(committed[-20:]))

    result: FusedResult | None = st.session_state.latest_result
    if result is None:
        st.info("Waiting for a completed utterance…")
        return

    # ---- Score cards ---------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acoustic score", f"{result.acoustic_score:.3f}")
    c2.metric("Linguistic score", f"{result.linguistic_score:.3f}")
    c3.metric("Final score (smoothed)", f"{result.smoothed_score:.3f}")
    c4.metric("Reliability", f"{result.reliability:.2f}")

    # ---- Severity badge ------------------------------------------------
    severity_color = {
        "Low": "🟢", "Mild": "🟡", "Moderate": "🟠", "High": "🔴"
    }.get(result.severity, "⚪")
    st.markdown(f"### {severity_color} Severity: **{result.severity}**")

    # ---- Behaviour tags ------------------------------------------------
    if result.behaviours:
        st.subheader("Detected Behaviours")
        cols = st.columns(len(result.behaviours))
        for col, b in zip(cols, result.behaviours):
            col.info(b)
    else:
        st.subheader("Detected Behaviours")
        st.success("No audio agitation detected")

    # ---- Explainability panel -----------------------------------------
    st.subheader("Why was this detected?")
    all_contributions = {
        **result.acoustic_contributions,
        **{k: v for k, v in result.linguistic_contributions.items() if not k.startswith("[")},
    }
    if all_contributions:
        max_val = max(abs(v) for v in all_contributions.values()) or 1.0
        for feature, contrib in sorted(all_contributions.items(), key=lambda x: -abs(x[1])):
            bar_len = int(abs(contrib) / max_val * 20)
            bar = "█" * bar_len
            sign = "+" if contrib >= 0 else "-"
            st.text(f"  {feature:<35} {sign}{abs(contrib):.4f}  {bar}")

    # ---- Utterance text -----------------------------------------------
    if result.utterance:
        with st.expander("Analysed utterance text"):
            st.write(result.utterance.full_text)

    # ---- Timeline chart -----------------------------------------------
    if st.session_state.timeline:
        st.subheader("📊 Score Timeline")
        df = pd.DataFrame(st.session_state.timeline).set_index("time")
        st.line_chart(df[["acoustic_score", "linguistic_score", "smoothed_score"]])

    # ---- Optional Gemini comparison -----------------------------------
    if config.ENABLE_GEMINI_COMPARISON and result.gemini_result:
        with st.expander("🤖 Gemini comparison (ablation)", expanded=False):
            st.json(result.gemini_result)


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
