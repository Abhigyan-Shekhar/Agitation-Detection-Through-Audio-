"""Streamlit MVP dashboard for Person 1 -> Person 2 -> Qwen Person 3 analysis."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
import wave

import pandas as pd
import streamlit as st

from batch_transcription import SUPPORTED_AUDIO_EXTENSIONS, preprocess_upload, transcribe_upload
from person2_module import analyze_person1_transcript
from qwen_person3 import FinalBehaviourResult, Person3Error, analyze_person2_behaviours


ANALYSIS_RESULT_STATE_KEY = "mvp_analysis_result"
ANALYSIS_UPLOAD_STATE_KEY = "mvp_analysis_upload_key"


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
    processed = preprocess_upload(data, filename)
    start_index = max(0, int(start * processed.sample_rate))
    end_index = min(processed.samples.size, int(end * processed.sample_rate))
    if end_index <= start_index:
        end_index = min(processed.samples.size, start_index + int(0.25 * processed.sample_rate))
    pcm = (processed.samples[start_index:end_index].clip(-1.0, 1.0) * 32767).astype("<i2")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(processed.sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def run_pipeline(data: bytes, filename: str) -> dict[str, Any]:
    """Run the complete MVP processing pipeline for one uploaded audio file."""
    person1 = transcribe_upload(data, filename)
    transcript_contract = person1.transcript_contract()
    person2 = analyze_person1_transcript(person1.person2_contract())
    behaviour_contract = person2.behaviour_contract()
    final_results = analyze_person2_behaviours(behaviour_contract)
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
        processed = preprocess_upload(data, filename)
        st.write(f"Filename: **{processed.filename}**")
        st.write(f"Duration: **{processed.duration:.1f} seconds**")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Audio validation failed: {exc}")
        return

    run_analysis = st.button("Run MVP analysis", type="primary")
    pipeline = st.session_state.get(ANALYSIS_RESULT_STATE_KEY)
    if run_analysis:
        try:
            with st.status("Processing audio-analysis pipeline...", expanded=True) as status:
                st.write("Person 1: transcription with timestamps")
                pipeline = run_pipeline(data, filename)
                st.write("Person 2: contextual chunks, embeddings, initial behaviour evidence")
                st.write("Person 3: Qwen validation through Groq")
                status.update(label="Processing complete", state="complete")
            st.session_state[ANALYSIS_RESULT_STATE_KEY] = pipeline
        except Person3Error as exc:
            st.error(f"Qwen/Groq analysis error: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
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
