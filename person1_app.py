"""Streamlit upload UI for the Person 1 batch transcription module."""
from __future__ import annotations

import json

import streamlit as st

import config
from batch_transcription import AudioValidationError, transcribe_upload
from transcriber import DirectWhisperTranscriber


st.set_page_config(page_title="Timestamped audio transcription", page_icon="🎙️", layout="wide")


@st.cache_resource(show_spinner="Loading the local speech-to-text model…")
def _transcriber() -> DirectWhisperTranscriber:
    return DirectWhisperTranscriber()


st.title("🎙️ Uploaded audio → timestamped transcript")
st.caption(
    "Person 1 module: validates and resamples uploaded audio to mono 16 kHz, "
    "then transcribes it locally with faster-whisper. Original audio never leaves this process."
)

uploaded = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg", "oga", "webm"],
    help="Maximum 200 MB and 120 minutes.",
)

if uploaded is not None:
    data = uploaded.getvalue()
    st.audio(data, format=uploaded.type or "audio/wav")
    if st.button("Transcribe audio", type="primary"):
        try:
            with st.spinner("Preprocessing and transcribing audio…"):
                result = transcribe_upload(data, uploaded.name, transcriber=_transcriber())
        except AudioValidationError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
        else:
            payload = result.transcript_contract()
            st.success(f"Created {len(payload)} timestamped segments from {result.duration:.1f} seconds of audio.")
            st.dataframe(payload, use_container_width=True, hide_index=True)
            st.download_button(
                "Download transcript JSON",
                data=json.dumps(payload, indent=2, ensure_ascii=False),
                file_name=f"{result.filename}.transcript.json",
                mime="application/json",
            )
            with st.expander("Processing metadata"):
                st.json(
                    {
                        "filename": result.filename,
                        "duration": round(result.duration, 3),
                        "sample_rate": result.sample_rate,
                        "model": result.model,
                        "language": config.WHISPER_LANGUAGE or "auto",
                    }
                )
