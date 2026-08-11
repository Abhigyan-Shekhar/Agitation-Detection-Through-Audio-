# Local audio behaviour dashboard

This branch adds real-time, session-local speaker distinction to the existing
`fix/baseline-calibration` architecture. It uses the microphone audio already
queued for local `faster-whisper`; it does **not** use WhisperLiveKit, create a
second microphone stream, or send audio to a cloud service.

## Speaker diarization setup

```bash
pip install -r requirements.txt
pip install -r requirements-diarization.txt
streamlit run dashboard.py
```

The first diarized segment downloads SpeechBrain's ECAPA-TDNN VoxCeleb model
from `speechbrain/spkrec-ecapa-voxceleb`. Speaker embeddings are clustered
online with cosine similarity and centroids are reset whenever a monitoring
session starts.

Configuration:

```bash
ENABLE_SPEAKER_DIARIZATION=true
DIARIZATION_BACKEND=speechbrain-ecapa
DIARIZATION_SIMILARITY_THRESHOLD=0.22
DIARIZATION_MIN_SEGMENT_SECONDS=1.0
DIARIZATION_MAX_SPEAKERS=6
```

Set `ENABLE_SPEAKER_DIARIZATION=false` to retain the original unlabelled
single-speaker behavior. If the optional dependency or model cannot load, the
session continues without speaker labels and logs a clear error.

`Speaker 1`, `Speaker 2`, and so on are stable only during one monitoring
session. They represent voice clusters, not names or biometric identities.
Overlapping speakers and very short turns can remain unattributed because the
ASR segments do not provide source separation.

The default similarity threshold is intentionally close to SpeechBrain's
native ECAPA speaker-verification threshold. Raise it only if different people
are being merged; lower it if one person is still split into multiple IDs.
