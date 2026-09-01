"""Streamlit MVP dashboard for Person 1 -> Person 2 -> Qwen Person 3 analysis."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from html import escape
from io import BytesIO
import logging
from pathlib import Path
import time
from typing import Any
import wave

import pandas as pd
import streamlit as st

import numpy as np

import config
from batch_transcription import SUPPORTED_AUDIO_EXTENSIONS, inspect_upload, iter_transcription_chunks, transcribe_upload
from person2_module import analyze_person1_transcript, prepare_embedding_provider
from qwen_person3 import FinalBehaviourResult, Person3Error, analyze_person2_behaviours


ANALYSIS_RESULT_STATE_KEY = "mvp_analysis_result"
ANALYSIS_UPLOAD_STATE_KEY = "mvp_analysis_upload_key"
SELECTED_EVENT_STATE_KEY = "mvp_selected_behaviour_event"
BEHAVIOUR_SELECT_STATE_KEY = "mvp_behaviour_select"
LOGGER = logging.getLogger(__name__)


def upload_cache_key(data: bytes, filename: str) -> str:
    """Return a stable identity for the uploaded audio currently being viewed."""
    return f"{Path(filename).name}:{sha256(data).hexdigest()}"


def format_timestamp(seconds: float) -> str:
    """Format audio-relative seconds as MM:SS.s for display."""
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:04.1f}"


def result_timestamp(result: FinalBehaviourResult | dict[str, Any]) -> str:
    """Return a display timestamp range for a final behaviour result."""
    start = float(result.start if isinstance(result, FinalBehaviourResult) else result["start"])
    end = float(result.end if isinstance(result, FinalBehaviourResult) else result["end"])
    return f"{format_timestamp(start)} – {format_timestamp(end)}"


def transcript_rows(transcript_contract: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert Person 1 transcript records into dashboard display rows."""
    return [
        {
            "Timestamp": f"{format_timestamp(float(segment['start']))} – {format_timestamp(float(segment['end']))}",
            "Speaker": str(segment.get("speaker_label", "Unknown")),
            "Text": str(segment.get("text", "")),
        }
        for segment in transcript_contract
    ]


def final_results_table(results: list[FinalBehaviourResult]) -> pd.DataFrame:
    """Build the final behaviour results table shown in Streamlit."""
    return pd.DataFrame(
        [
            {
                "Behaviour": result.behaviour,
                "Timestamp": result_timestamp(result),
                "Severity": result.severity,
                "Confidence": result.confidence,
                "Validated": result.validated,
            }
            for result in results
        ]
    )


def timeline_table(results: list[FinalBehaviourResult]) -> pd.DataFrame:
    """Build an audio-relative behaviour timeline data frame."""
    return pd.DataFrame(
        [
            {
                "start_sec": result.start,
                "duration_sec": max(0.05, result.end - result.start),
                "behaviour": result.behaviour,
                "confidence": result.confidence,
            }
            for result in results
        ]
    )


def validation_status(result: FinalBehaviourResult) -> str:
    """Return the dashboard label for Person 3's validation decision."""
    if result.validated:
        return "Supported"
    if result.severity == "Insufficient":
        return "Insufficient"
    return "Not supported"


def relevant_transcript_segments(
    result: FinalBehaviourResult,
    person2_behaviours: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return this result's source transcript units in audio-time order.

    Person 3 retains the source IDs selected from Person 2.  Resolving those
    IDs here keeps the dashboard grounded in the pipeline evidence without
    changing either upstream contract.
    """
    selected_ids = set(result.evidence_segment_ids or [])
    segments: dict[str, dict[str, Any]] = {}
    for behaviour in person2_behaviours:
        for segment in behaviour.get("evidence_segments", []) or []:
            if not isinstance(segment, dict):
                continue
            segment_id = str(segment.get("id", ""))
            if selected_ids and segment_id not in selected_ids:
                continue
            if not selected_ids:
                if result.chunk_id and behaviour.get("chunk_id") != result.chunk_id:
                    continue
                if behaviour.get("behaviour") != result.initial_behaviour:
                    continue
            if segment_id:
                segments.setdefault(segment_id, {
                    "id": segment_id,
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": str(segment.get("text", "")),
                })
    if segments:
        return sorted(segments.values(), key=lambda segment: (segment["start"], segment["end"], segment["id"]))
    # This fallback is still pipeline-produced text; it supports legacy final
    # results that predate evidence-segment IDs.
    return [{"id": "result-transcript", "start": result.start, "end": result.end, "text": result.transcript}] if result.transcript else []


def timeline_events(
    results: list[FinalBehaviourResult], person2_behaviours: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build one independently selectable timeline event per final result."""
    events = []
    for event_id, result in enumerate(results):
        segments = relevant_transcript_segments(result, person2_behaviours)
        events.append({
            "event_id": event_id,
            "result": result,
            "start": float(result.start),
            "end": float(result.end),
            "transcript_segments": segments,
        })
    return events


def selected_timeline_event(events: list[dict[str, Any]], event_id: int) -> dict[str, Any]:
    """Resolve a click/dropdown event identity without conflating overlaps."""
    for event in events:
        if event["event_id"] == event_id:
            return event
    raise ValueError(f"Unknown timeline event: {event_id}")


def audio_segment_bounds(event: dict[str, Any]) -> tuple[float, float]:
    """Return the exact original-audio bounds shared by both selection paths."""
    return float(event["start"]), float(event["end"])


def _event_hover_text(event: dict[str, Any]) -> str:
    """Create Plotly hover content from actual final and source evidence."""
    result = event["result"]
    transcript = "<br>".join(
        f"{escape(segment['id'])} {escape(format_timestamp(segment['start']))}–{escape(format_timestamp(segment['end']))}: &quot;{escape(segment['text'])}&quot;"
        for segment in event["transcript_segments"]
    ) or "No source transcript available"
    return (
        f"<b>{escape(result.behaviour)}</b><br>{escape(result_timestamp(result))}<br>"
        f"Confidence: {result.confidence:.2f}<br>Severity: {escape(result.severity)}<br>"
        f"Status: {validation_status(result)}<br><br><b>Transcript</b><br>{transcript}<br><br>"
        f"<b>Evidence</b><br>{escape(result.evidence or result.person2_evidence)}<extra></extra>"
    )


def interactive_timeline_figure(events: list[dict[str, Any]], selected_event_id: int | None):
    """Build a clickable Plotly timeline while preserving one bar per event."""
    import plotly.graph_objects as go

    figure = go.Figure()
    for event in events:
        result = event["result"]
        selected = event["event_id"] == selected_event_id
        figure.add_trace(go.Bar(
            x=[max(0.05, event["end"] - event["start"])], base=[event["start"]],
            y=[f"{result.behaviour} #{event['event_id'] + 1}"], orientation="h",
            name=result.behaviour, customdata=[event["event_id"]],
            marker={"line": {"color": "#111827" if selected else "#ffffff", "width": 3 if selected else 1}},
            hovertemplate=_event_hover_text(event), showlegend=False,
        ))
    figure.update_layout(
        barmode="overlay", height=max(300, 54 * len(events) + 100),
        xaxis_title="Audio-relative time (seconds)", yaxis_title="Behaviour event",
        margin={"l": 180, "r": 30, "t": 25, "b": 55},
        clickmode="event+select",
    )
    return figure


def extract_audio_segment_wav(data: bytes, filename: str, start: float, end: float) -> bytes:
    """Return a WAV segment aligned to original audio-relative timestamps."""
    sample_rate = config.SAMPLE_RATE
    requested_start = max(0.0, float(start))
    requested_end = max(requested_start + 0.25, float(end))
    parts: list[np.ndarray] = []
    for chunk in iter_transcription_chunks(
        data,
        filename,
        target_sample_rate=sample_rate,
        chunk_seconds=max(30.0, requested_end - requested_start + 1.0),
        overlap_seconds=0.0,
    ):
        if chunk.primary_end <= requested_start:
            continue
        if chunk.primary_start >= requested_end:
            break
        local_start = max(0, int((requested_start - chunk.primary_start) * sample_rate))
        local_end = min(chunk.samples.size, int((requested_end - chunk.primary_start) * sample_rate))
        if local_end > local_start:
            parts.append(chunk.samples[local_start:local_end])
    if not parts:
        parts = [np.zeros(int(0.25 * sample_rate), dtype=np.float32)]
    samples = np.concatenate(parts)
    pcm = (samples.clip(-1.0, 1.0) * 32767).astype("<i2")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def run_pipeline(data: bytes, filename: str, *, progress_callback: Any | None = None) -> dict[str, Any]:
    """Run the complete MVP processing pipeline for one uploaded audio file."""
    started = time.monotonic()
    LOGGER.info("MVP pipeline started filename=%s bytes=%d", filename, len(data))
    if progress_callback is not None:
        progress_callback("preparing", 0, 1, "Validating audio...")
    person1 = transcribe_upload(data, filename, progress_callback=progress_callback)
    LOGGER.info(
        "Person 1 complete filename=%s duration=%.3fs transcript_segments=%d elapsed=%.2fs",
        filename,
        person1.duration,
        len(person1.segments),
        time.monotonic() - started,
    )
    transcript_contract = person1.transcript_contract()
    person2_started = time.monotonic()
    if progress_callback is not None:
        progress_callback("person2", 0, 1, "Analyzing behaviour evidence")
    embedding_provider = prepare_embedding_provider(
        progress_callback=(
            (lambda message: progress_callback("person2", 0, 1, message))
            if progress_callback is not None else None
        )
    )
    person2 = analyze_person1_transcript(person1.person2_contract(), embedding_provider=embedding_provider)
    LOGGER.info(
        "Person 2 complete filename=%s chunks=%d behaviours=%d elapsed=%.2fs",
        filename,
        len(person2.chunks),
        len(person2.behaviours),
        time.monotonic() - person2_started,
    )
    behaviour_contract = person2.behaviour_contract()
    if progress_callback is not None:
        progress_callback("person3", 0, max(1, len(behaviour_contract)), "Validating candidates with Qwen...")
    person3_started = time.monotonic()
    def person3_progress(completed: int, total: int, record: dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(
                "person3",
                completed,
                max(1, total),
                f"Validating {completed}/{total}: {record.get('behaviour', 'behaviour evidence')}",
            )

    final_results = analyze_person2_behaviours(behaviour_contract, progress_callback=person3_progress)
    if progress_callback is not None:
        progress_callback("person3", len(behaviour_contract), max(1, len(behaviour_contract)), "Validation complete")
    LOGGER.info(
        "Person 3 complete filename=%s final_results=%d elapsed=%.2fs",
        filename,
        len(final_results),
        time.monotonic() - person3_started,
    )
    LOGGER.info(
        "MVP pipeline complete filename=%s final_results=%d total_elapsed=%.2fs",
        filename,
        len(final_results),
        time.monotonic() - started,
    )
    return {
        "person1": person1,
        "transcript": transcript_contract,
        "person2": person2,
        "person2_behaviours": behaviour_contract,
        "final_results": final_results,
    }


def main() -> None:
    """Render the new MVP dashboard."""
    st.set_page_config(page_title="Audio Behaviour MVP Dashboard", layout="wide")
    st.title("Audio Behaviour MVP Dashboard")
    st.caption("Person 1 transcription → Person 2 initial evidence → Qwen/Groq Person 3 validation")

    uploaded = st.file_uploader(
        "Upload audio",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_AUDIO_EXTENSIONS)],
    )
    if uploaded is None:
        st.info("Upload an audio file to begin.")
        return

    data = uploaded.getvalue()
    filename = uploaded.name
    current_upload_key = upload_cache_key(data, filename)
    if st.session_state.get(ANALYSIS_UPLOAD_STATE_KEY) != current_upload_key:
        # A different upload must never show the previous file's analysis, but
        # interactions with the same file (such as timestamp selection) keep it.
        st.session_state[ANALYSIS_UPLOAD_STATE_KEY] = current_upload_key
        st.session_state.pop(ANALYSIS_RESULT_STATE_KEY, None)

    st.subheader("1. Audio Upload")
    try:
        metadata = inspect_upload(data, filename)
        st.write(f"Filename: **{metadata.filename}**")
        st.write(f"Duration: **{metadata.duration:.1f} seconds**")
        st.write(f"Upload size: **{metadata.source_bytes / (1024 * 1024):.1f} MB / 200 MB**")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Audio validation failed: {exc}")
        return

    run_analysis = st.button("Run MVP analysis", type="primary")
    pipeline = st.session_state.get(ANALYSIS_RESULT_STATE_KEY)
    if run_analysis:
        try:
            with st.status("Processing audio-analysis pipeline...", expanded=True) as status:
                progress_bar = st.progress(0.0)
                stage_line = st.empty()

                def update_progress(stage: str, completed: int, total: int, message: str) -> None:
                    stage_weights = {
                        "preparing": (0.00, 0.05),
                        "loading_model": (0.05, 0.08),
                        "transcribing": (0.13, 0.62),
                        "person2": (0.70, 0.15),
                        "person3": (0.85, 0.15),
                        "speaker_identification": (0.62, 0.08),
                    }
                    base, span = stage_weights.get(stage, (0.0, 1.0))
                    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, completed / total))
                    progress_bar.progress(min(1.0, base + span * fraction))
                    stage_line.write(message)

                pipeline = run_pipeline(data, filename, progress_callback=update_progress)
                status.update(label="Processing complete", state="complete")
            st.session_state[ANALYSIS_RESULT_STATE_KEY] = pipeline
        except Person3Error as exc:
            LOGGER.exception("Qwen/Groq analysis error for filename=%s", filename)
            st.error(f"Qwen/Groq analysis error: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("MVP pipeline failed for filename=%s", filename)
            st.error(f"Pipeline failed: {exc}")
            return
    elif pipeline is None:
        st.info("Click Run MVP analysis to process the upload.")
        return

    transcript = pipeline["transcript"]
    person2_behaviours = pipeline["person2_behaviours"]
    final_results = pipeline["final_results"]

    st.subheader("2. Transcript")
    person1 = pipeline["person1"]
    if person1.speaker_identification_error:
        st.warning(f"Speaker identification unavailable: {person1.speaker_identification_error}")
    elif person1.patient_speaker_enrolled:
        st.caption(
            f"Patient voice enrolled from the first {person1.speaker_enrollment_seconds:.1f}s; "
            "later speech is labeled by speaker-embedding similarity."
        )
    st.dataframe(pd.DataFrame(transcript_rows(transcript)), use_container_width=True, hide_index=True)

    st.subheader("3. Initial Behaviour Detection")
    st.dataframe(pd.DataFrame(person2_behaviours), use_container_width=True, hide_index=True)

    st.subheader("4. Final Behaviour Results")
    table = final_results_table(final_results)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("6. Behaviour Timeline")
    events = timeline_events(final_results, person2_behaviours)
    if not events:
        st.info("No final behaviours were detected.")
        return
    selected_event_id = st.session_state.get(SELECTED_EVENT_STATE_KEY, 0)
    if not any(event["event_id"] == selected_event_id for event in events):
        selected_event_id = 0
        st.session_state[SELECTED_EVENT_STATE_KEY] = selected_event_id

    # A key that changes after each selection clears Plotly's old selection on
    # the next rerun, so a later dropdown change is not mistaken for a click.
    chart_state = st.plotly_chart(
        interactive_timeline_figure(events, selected_event_id),
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=f"behaviour_timeline_{selected_event_id}",
    )
    selection = getattr(chart_state, "selection", None)
    points = selection.get("points", []) if isinstance(selection, dict) else getattr(selection, "points", [])
    if points:
        clicked_event_id = int(points[0]["customdata"])
        if clicked_event_id != selected_event_id:
            selected_event_id = clicked_event_id
            st.session_state[SELECTED_EVENT_STATE_KEY] = selected_event_id
            st.session_state[BEHAVIOUR_SELECT_STATE_KEY] = selected_event_id
            st.rerun()

    st.caption("Hover for evidence and transcript context. Click or tap a bar to select its exact audio segment.")

    st.subheader("5. Evidence / Explanation and 7. Audio Segment Playback")
    choice = st.selectbox(
        "Select behaviour",
        options=[event["event_id"] for event in events],
        index=selected_event_id,
        format_func=lambda event_id: (
            f"{selected_timeline_event(events, event_id)['result'].behaviour} "
            f"({result_timestamp(selected_timeline_event(events, event_id)['result'])})"
        ),
        key=BEHAVIOUR_SELECT_STATE_KEY,
    )
    if int(choice) != selected_event_id:
        selected_event_id = int(choice)
        st.session_state[SELECTED_EVENT_STATE_KEY] = selected_event_id
        st.rerun()

    selected_event = selected_timeline_event(events, selected_event_id)
    selected = selected_event["result"]
    start, end = audio_segment_bounds(selected_event)
    st.markdown("#### Selected behaviour")
    st.write(f"**{selected.behaviour}**  ")
    st.write(f"**{result_timestamp(selected)}**")
    st.write(f"Confidence: **{selected.confidence:.0%}** | Severity: **{selected.severity}** | Validation: **{validation_status(selected)}**")
    st.markdown("**Relevant transcript**")
    if selected_event["transcript_segments"]:
        for segment in selected_event["transcript_segments"]:
            st.write(f"`{segment['id']}` {format_timestamp(segment['start'])}–{format_timestamp(segment['end'])}: “{segment['text']}”")
    else:
        st.caption("No source transcript is available for this event.")
    st.markdown("**Supporting evidence**")
    st.write(selected.evidence or selected.person2_evidence)
    st.json(asdict(selected))
    st.write(f"**Person 2 evidence:** {selected.person2_evidence}")
    st.write(f"**Qwen explanation:** {selected.explanation}")
    st.write(f"Start: {start:.2f} sec | End: {end:.2f} sec")
    try:
        st.audio(extract_audio_segment_wav(data, filename, start, end), format="audio/wav")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not extract playback segment: {exc}")


if __name__ == "__main__":
    main()
