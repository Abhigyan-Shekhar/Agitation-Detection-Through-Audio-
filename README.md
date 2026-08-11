# Audio + Linguistic Agitation Dashboard

Local Streamlit prototype for real-time microphone transcription, acoustic and
linguistic cue extraction, and CMAI-inspired audio behaviour labels. Detected
speech can be attributed to session-local speakers through WhisperLiveKit's
native streaming diarization. This is decision support, not a clinical
diagnosis or biometric identity system.

## Speaker-aware architecture

```text
microphone ─┬─> acoustic feature windows ───────────────┐
            └─> WhisperLiveKit (PCM + Sortformer)       │
                         └─> committed speaker lines ─┬─> transcript display
                                                     └─> utterance aggregator
                                                          └─> per-speaker linguistic history
                                                               └─> score fusion / behaviour event
```

WLK `lines` are the source of truth for speaker identity. Full snapshots and
diff messages are supported, silence speaker `-2` is ignored, and repeated
snapshots are deduplicated by speaker, timestamps, and text. WLK's stream-time
timestamps are converted to Unix time before acoustic windows are aggregated.

`Speaker 1`, `Speaker 2`, and so on mean only that speech was clustered as the
same session-local voice. IDs may reset after a server/client restart and do
not identify a named person. Optional display aliases can be supplied without
changing the underlying ID:

```bash
SPEAKER_ALIASES_JSON='{"1":"Resident","2":"Caregiver"}'
```

## Installation

The base application and single-speaker fallback support the repository's
existing environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Speaker diarization uses WLK's optional NeMo/Sortformer stack. Install it in a
Python 3.11-3.12 Linux environment (an NVIDIA CUDA system is the practical
real-time target):

```bash
python -m pip install -r requirements-diarization.txt
```

Current WhisperLiveKit releases require Python below 3.14. Sortformer/NeMo is
not treated as a supported Apple Silicon runtime by this project; on macOS or
Python 3.14 the application defaults to the existing faster-whisper path. This
is an explicit fallback, not simulated diarization. A WLK initialization or
missing-extra failure is shown with the launch command and server log tail.

## Run

On a supported diarization machine, the exact application command is:

```bash
ENABLE_SPEAKER_DIARIZATION=true \
TRANSCRIPTION_ENGINE=whisperlivekit \
DIARIZATION_BACKEND=sortformer \
python -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501
```

The dashboard auto-launches a server equivalent to:

```bash
python -m whisperlivekit.basic_server \
  --backend auto --model small --language en --pcm-input \
  --host 127.0.0.1 --port 8000 \
  --diarization --diarization-backend sortformer
```

Disable diarization while retaining WLK transcription:

```bash
ENABLE_SPEAKER_DIARIZATION=false TRANSCRIPTION_ENGINE=whisperlivekit \
python -m streamlit run dashboard.py
```

Use the pre-feature, single-speaker fallback (also the automatic unsupported-
platform/dependency default):

```bash
ENABLE_SPEAKER_DIARIZATION=false TRANSCRIPTION_ENGINE=faster-whisper \
python -m streamlit run dashboard.py
```

For an externally managed WLK server, set `WLK_AUTO_LAUNCH=false` and configure
`WLK_HOST`, `WLK_PORT`, and `WLK_PATH`. `WLK_OUTPUT_MODE` may be `diff` (default)
or `full`.

## Tests

Parser and pipeline tests do not download or load speech models:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```
