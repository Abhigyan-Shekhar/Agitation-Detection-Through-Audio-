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

## Uploaded audio transcription (Person 1)

The batch module accepts WAV, MP3, M4A, FLAC, OGG, or WebM uploads (up to
200 MB and 120 minutes), validates that they contain decodable audio, and
resamples them in memory to mono 16 kHz float PCM for local faster-whisper.
It deliberately retains silence so every timestamp remains aligned with the
original audio used by the final playback UI.

Run the standalone upload interface:

```bash
streamlit run person1_app.py
```

Its downloaded JSON is the Person 1 → Person 2 integration contract:

```json
[
  {
    "start": 10.2,
    "end": 15.4,
    "text": "Where is my daughter?",
    "confidence": 0.91
  }
]
```

`batch_transcription.transcribe_upload` exposes the same contract directly to
Python callers. Processing is local, so there is no network upload to compress;
the in-memory mono 16 kHz representation already reduces the model input while
avoiding lossy re-encoding.

## Person 2 transcript evidence layer

Person 2 starts from the Person 1 timestamped transcript contract above. It
does not upload audio, preprocess audio, run ASR, generate timestamps, or call
Qwen/Groq. The batch entry point is:

```python
from person2_module import analyze_person1_transcript

person2_result = analyze_person1_transcript(person1_result.transcript_contract())
person3_payload = person2_result.behaviour_contract()
```

The pipeline is:

```text
Timestamped transcript
  -> contextual chunks
  -> semantic chunk embeddings
  -> repetition/context analysis
  -> initial CMAI-aligned behaviour evidence
  -> timestamped structured results for Person 3
```

Contextual chunking groups adjacent transcript segments while preserving the
original segment list and indices. Defaults live in `config.py`:

- `PERSON2_MAX_CHUNK_DURATION_SECONDS=20`
- `PERSON2_CHUNK_MAX_SEGMENTS=8`
- `PERSON2_CHUNK_OVERLAP_SEGMENTS=1`
- `PERSON2_REPETITION_MIN_OCCURRENCES=2`
- `PERSON2_REPETITION_SIMILARITY_THRESHOLD=0.90`
- `PERSON2_SEMANTIC_SIMILARITY_THRESHOLD=0.70`

For every chunk, `start` is the earliest included transcript timestamp and
`end` is the latest included timestamp. Behaviour evidence uses the timestamp
range of the chunk or the repeated occurrences that triggered the evidence.
The 20-second window is an initial MVP engineering baseline for local
contextual analysis; CMAI does not prescribe this chunk duration. The goal is
to keep related utterances such as immediate repeated questions together while
reducing unrelated later speech in the same behavioural context. The value is
configurable for future 15/20/25/30-second ablation comparisons.

The default embedding backend is `sentence-transformers`, using
`sentence-transformers/all-MiniLM-L6-v2`. It produces 384-dimensional semantic
embeddings and is loaded through the Person 2 embedding provider abstraction so
the model is reused rather than recreated for every segment. Hashing remains an
optional backend for experiments, but it is no longer the default. Embedding
similarity is used as supporting evidence for local question/request-like
repetition where wording changes slightly; it is not a clinical probability, a
CMAI score, or enough by itself to infer unsupported behaviours.

Person 2 emits only initial heuristic evidence using the existing canonical
audio taxonomy. The transcript-supported labels are:

- Cursing / verbal aggression
- Making verbal sexual advances
- Repetitive sentences or questions
- Making strange noises, when a transcript or audio-caption annotation names a
  non-speech vocalization such as `[moaning]`
- Complaining
- Negativism
- Constant unwarranted requests for attention/help
- Distressed/urgent verbalization

`Screaming` remains in the audio taxonomy, but Person 2 does not claim it from
transcript text alone because Person 1's batch JSON does not include acoustic
intensity features. Physical or visual CMAI behaviours such as pacing, hitting,
kicking, grabbing, or trying to leave are not emitted by this audio/transcript
module.

Person 3 receives a JSON-ready list of dictionaries:

```json
[
  {
    "start": 10.2,
    "end": 27.5,
    "behaviour": "Repetitive sentences or questions",
    "internal_code": "AUDIO_REPETITIVE",
    "cmai_category": "Verbally non-aggressive: repetitive questioning",
    "score": 1.0,
    "score_type": "heuristic_repetition_score",
    "evidence": "Phrase repeated 3 times within nearby transcript segments.",
    "text": "Where is my daughter? where is my daughter Where is my daughter!",
    "chunk_id": "chunk-0000",
    "repetition": {
      "repeated_phrase": "Where is my daughter?",
      "count": 3,
      "occurrences": []
    },
    "modality": "audio",
    "mapping_status": "mapped"
  }
]
```

Scores are heuristic linguistic or repetition evidence scores in the range
0–1. They are not clinical probabilities and are not final CMAI severity
ratings. Person 3 can use this payload, plus the timestamped text evidence, as
structured input to the later Qwen/Groq interpretation stage.
