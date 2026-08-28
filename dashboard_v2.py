"""Streamlit MVP dashboard for Person 1 -> Person 2 -> Qwen Person 3 analysis."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
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
from person2_module import analyze_person1_transcript
from qwen_person3 import FinalBehaviourResult, Person3Error, analyze_person2_behaviours


ANALYSIS_RESULT_STATE_KEY = "mvp_analysis_result"
ANALYSIS_UPLOAD_STATE_KEY = "mvp_analysis_upload_key"
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
        progress_callback("preparing", 0, 1, "Preparing audio")
    person1 = transcribe_upload(data, filename, progress_callback=progress_callback)
    LOGGER.info(
        "Person 1 complete filename=%s duration=%.3fs transcript_segments=%d elapsed=%.2fs",
        filename,
        person1.duration,
        len(person1.segments),
        time.monotonic() - started,
    )
    transcript_contract = person1.transcript_contract()
    if progress_callback is not None:
        progress_callback("person2", 0, 1, "Analyzing behaviour evidence")
    person2 = analyze_person1_transcript(person1.person2_contract())
    LOGGER.info(
        "Person 2 complete filename=%s chunks=%d behaviours=%d elapsed=%.2fs",
        filename,
        len(person2.chunks),
        len(person2.behaviours),
        time.monotonic() - started,
    )
    behaviour_contract = person2.behaviour_contract()
    if progress_callback is not None:
        progress_callback("person3", 0, max(1, len(behaviour_contract)), "Validating Person 2 results")
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
        "MVP pipeline complete filename=%s final_results=%d elapsed=%.2fs",
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
                        "transcribing": (0.05, 0.70),
                        "person2": (0.70, 0.15),
                        "person3": (0.85, 0.15),
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
    st.dataframe(pd.DataFrame(transcript_rows(transcript)), use_container_width=True, hide_index=True)

    st.subheader("3. Initial Behaviour Detection")
    st.dataframe(pd.DataFrame(person2_behaviours), use_container_width=True, hide_index=True)

    st.subheader("4. Final Behaviour Results")
    table = final_results_table(final_results)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("6. Behaviour Timeline")
    timeline = timeline_table(final_results)
    if not timeline.empty:
        st.bar_chart(timeline, x="start_sec", y="duration_sec", color="behaviour")
    else:
        st.info("No final behaviours were detected.")

    st.subheader("5. Evidence / Explanation and 7. Audio Segment Playback")
    if not final_results:
        return
    choice = st.selectbox(
        "Select behaviour",
        options=list(range(len(final_results))),
        format_func=lambda idx: f"{final_results[idx].behaviour} ({result_timestamp(final_results[idx])})",
    )
    selected = final_results[int(choice)]
    st.json(asdict(selected))
    st.write(f"**Person 2 evidence:** {selected.person2_evidence}")
    st.write(f"**Qwen explanation:** {selected.explanation}")
    st.write(f"**Relevant transcript:** {selected.transcript}")
    st.write(f"Start: {selected.start:.1f} sec | End: {selected.end:.1f} sec")
    try:
        st.audio(extract_audio_segment_wav(data, filename, selected.start, selected.end), format="audio/wav")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not extract playback segment: {exc}")


if __name__ == "__main__":
    main()
