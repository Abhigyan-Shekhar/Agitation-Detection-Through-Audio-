"""Continuously refreshed Streamlit view of AudioPipeline and Person 3 output."""
from __future__ import annotations

import logging
import queue
import time
from typing import Any

import pandas as pd
import streamlit as st

from audio_pipeline import AudioPipeline
from person3_module import (
    GeminiBehaviourAnalyzer,
    analyze_person3,
    resolve_acoustic_score,
    compute_final_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Audio Agitation Dashboard", layout="wide")
st.title("Audio + Linguistic Agitation Dashboard")
st.caption("Decision support only; CMAI-inspired labels are not clinical diagnoses.")


def initialise() -> None:
    st.session_state.setdefault("timeline", [])
    st.session_state.setdefault("latest", None)
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("person3_analyzer", None)
    st.session_state.setdefault("dashboard_error", None)
    st.session_state.setdefault("last_queue_size", 0)
    st.session_state.setdefault("last_gemini_call_time", 0.0)
    st.session_state.setdefault("last_gemini_result", None)


def stop_pipeline() -> None:
    pipeline = st.session_state.pipeline
    if pipeline is not None:
        pipeline.stop()
        logger.info("Audio pipeline stopped from dashboard")
    st.session_state.pipeline = None


def get_analyzer() -> GeminiBehaviourAnalyzer:
    if st.session_state.person3_analyzer is None:
        logger.info("Creating Gemini Person 3 analyzer")
        st.session_state.person3_analyzer = GeminiBehaviourAnalyzer()
    return st.session_state.person3_analyzer


def consume_chunks() -> None:
    """Drain all completed pipeline chunks on every timed fragment rerun."""
    pipeline = st.session_state.pipeline
    if pipeline is None:
        return

    st.session_state.last_queue_size = pipeline.output_queue.qsize()
    logger.info("Dashboard polling output queue; size=%s", st.session_state.last_queue_size)
    while True:
        try:
            chunk = pipeline.output_queue.get_nowait()
        except queue.Empty:
            return

        logger.info("Dashboard dequeued a pipeline result; starting Person 3 analysis")
        transcript_override = st.session_state.get("transcript_override", "").strip()
        transcript = transcript_override or chunk.get("transcript", "")
        try:
            # AudioPipeline does not currently enqueue an acoustic score. Resolve
            # the Person 3 feature-based fallback before storing dashboard state.
            chunk["acoustic_score"] = resolve_acoustic_score(
                chunk.get("acoustic_score"), chunk.get("acoustic_features")
            )
            
            current_time = time.time()
            # Enforce rate limit (1 call per 15 seconds) to avoid 429 RESOURCE_EXHAUSTED
            if transcript and (current_time - st.session_state.last_gemini_call_time) >= 15.0:
                try:
                    result = analyze_person3(
                        transcript=transcript,
                        acoustic_features=chunk["acoustic_features"],
                        acoustic_score=chunk.get("acoustic_score"),
                        analyzer=get_analyzer(),
                    )
                    st.session_state.last_gemini_call_time = current_time
                    st.session_state.last_gemini_result = result
                    st.session_state.dashboard_error = None
                except Exception as exc:
                    if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                        logger.warning("Rate limit hit, backing off.")
                        st.session_state.last_gemini_call_time = current_time + 30.0 # Force a longer backoff
                        st.session_state.dashboard_error = "Gemini rate limit (429) hit. Reusing previous/fallback linguistic score."
                    else:
                        raise
            
            # Use the actual result if we just computed it, or fallback
            if st.session_state.last_gemini_result is not None:
                result = dict(st.session_state.last_gemini_result)
                # Ensure the acoustic and final score are up-to-date with this chunk's acoustic features
                result["acoustic_score"] = chunk["acoustic_score"]
                result["final_score"] = compute_final_score(chunk["acoustic_score"], result["gemini"]["agitation_score"])
            else:
                # Neutral fallback if we haven't made a successful call yet
                result = {
                    "acoustic_score": chunk["acoustic_score"],
                    "gemini": {
                        "emotion": "Neutral",
                        "agitation_score": 0.0,
                        "behaviours": [],
                        "reasoning": "Awaiting initial Gemini analysis (rate limiting or cooldown in effect)."
                    },
                    "cmai_mapping": [],
                    "final_score": compute_final_score(chunk["acoustic_score"], 0.0)
                }
                
        except Exception as exc:
            logger.exception("Person 3 analysis failed")
            st.session_state.dashboard_error = f"Person 3 analysis failed: {exc}"
            return

        logger.info("Person 3 analysis completed; updating dashboard state")
        st.session_state.dashboard_error = None
        st.session_state.latest = {
            **chunk,
            **result,
            "transcript": transcript,
            "acoustic_score": chunk.get("acoustic_score"),
        }
        st.session_state.timeline.append(
            {
                "time": time.strftime("%H:%M:%S", time.localtime(chunk["timestamp"])),
                "acoustic_score": chunk.get("acoustic_score"),
                "linguistic_score": result["gemini"]["agitation_score"],
                "final_score": result["final_score"],
            }
        )


def render_results() -> None:
    latest: dict[str, Any] | None = st.session_state.latest
    if st.session_state.dashboard_error:
        st.error(st.session_state.dashboard_error)
    if latest is None:
        st.info("Waiting for a completed speech chunk. The queue is polled every second while the microphone runs.")
        return

    first, second, third = st.columns(3)
    first.metric("Acoustic score", latest.get("acoustic_score", "Not available"))
    second.metric("Gemini agitation score", latest["gemini"]["agitation_score"])
    third.metric("Final agitation score", latest["final_score"])
    st.subheader("Acoustic features")
    st.json(latest["acoustic_features"])
    st.subheader("Transcript")
    st.write(latest.get("transcript") or "No transcript available")
    st.subheader("Gemini emotion")
    st.write(latest["gemini"]["emotion"])
    st.subheader("Detected behaviours")
    st.write(latest["gemini"]["behaviours"] or "None detected")
    st.subheader("CMAI-inspired mapping")
    st.dataframe(latest["cmai_mapping"], use_container_width=True)
    st.subheader("Reasoning")
    st.write(latest["gemini"]["reasoning"])

    if st.session_state.timeline:
        st.subheader("Timeline of scores")
        timeline = pd.DataFrame(st.session_state.timeline).set_index("time")
        st.line_chart(timeline[["acoustic_score", "linguistic_score", "final_score"]])


initialise()
with st.sidebar:
    st.header("Controls")
    st.text_area(
        "Transcript override (optional)",
        key="transcript_override",
        help="Uses external text instead of Whisper output for newly dequeued chunks.",
    )
    start, stop = st.columns(2)
    if start.button("Start microphone", disabled=st.session_state.pipeline is not None):
        try:
            st.session_state.pipeline = AudioPipeline()
            st.session_state.pipeline.start()
            logger.info("Audio pipeline started from dashboard")
        except Exception as exc:
            logger.exception("Audio pipeline could not start")
            st.session_state.pipeline = None
            st.session_state.dashboard_error = f"Audio pipeline could not start: {exc}"
    if stop.button("Stop microphone", disabled=st.session_state.pipeline is None):
        stop_pipeline()
    with st.expander("Debug status"):
        st.write("Pipeline running:", st.session_state.pipeline is not None)
        st.write("Last observed queue size:", st.session_state.last_queue_size)
        st.write("Timeline points:", len(st.session_state.timeline))


@st.fragment(run_every=1.0)
def live_results() -> None:
    """Streamlit reruns this fragment every second without restarting audio capture."""
    consume_chunks()
    render_results()


live_results()
