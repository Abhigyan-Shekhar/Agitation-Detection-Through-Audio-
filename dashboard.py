"""Streamlit dashboard for existing pipeline output and Person 3 analysis."""
from __future__ import annotations

import queue
import time

import pandas as pd
import streamlit as st

from audio_pipeline import AudioPipeline
from person3_module import GeminiBehaviourAnalyzer, analyze_person3

st.set_page_config(page_title="Audio Agitation Dashboard", layout="wide")
st.title("Audio + Linguistic Agitation Dashboard")
st.caption("Decision support only; CMAI-inspired labels are not clinical diagnoses.")


def initialise() -> None:
    st.session_state.setdefault("timeline", [])
    st.session_state.setdefault("latest", None)
    st.session_state.setdefault("pipeline", None)


def stop_pipeline() -> None:
    if st.session_state.pipeline is not None:
        st.session_state.pipeline.stop()
    st.session_state.pipeline = None


def consume_chunks(transcript_override: str) -> None:
    pipeline = st.session_state.pipeline
    if pipeline is None:
        return
    while True:
        try:
            chunk = pipeline.output_queue.get_nowait()
        except queue.Empty:
            return
        transcript = transcript_override.strip() or chunk.get("transcript", "")
        try:
            result = analyze_person3(
                transcript, chunk["acoustic_features"], chunk.get("acoustic_score"), GeminiBehaviourAnalyzer()
            )
        except (ValueError, ImportError) as exc:
            st.warning(f"Person 3 analysis unavailable: {exc}")
            return
        st.session_state.latest = {**chunk, "transcript": transcript, **result}
        st.session_state.timeline.append({
            "time": time.strftime("%H:%M:%S", time.localtime(chunk["timestamp"])),
            "acoustic_score": chunk.get("acoustic_score"),
            "linguistic_score": result["gemini"]["agitation_score"],
            "final_score": result["final_score"],
        })


initialise()
with st.sidebar:
    st.header("Controls")
    override = st.text_area("Transcript override (optional)", help="Uses external text instead of Whisper output for the next analysis.")
    start, stop = st.columns(2)
    if start.button("Start microphone", disabled=st.session_state.pipeline is not None):
        st.session_state.pipeline = AudioPipeline()
        st.session_state.pipeline.start()
    if stop.button("Stop microphone", disabled=st.session_state.pipeline is None):
        stop_pipeline()
    st.button("Refresh results")

consume_chunks(override)
latest = st.session_state.latest
if latest is None:
    st.info("Start the microphone and speak until a speech window is emitted. Set GEMINI_API_KEY before analysis.")
else:
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
