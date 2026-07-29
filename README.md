# Audio + Linguistic Agitation Dashboard

Local Streamlit dashboard for real-time microphone audio, WhisperLiveKit
transcription, linguistic feature extraction, acoustic feature extraction, and
CMAI-inspired audio behaviour labels.

This is decision-support only. It is not a clinical diagnosis.

## Tested Environment

- Python 3.11, 3.12, or 3.14
- macOS with the built-in microphone tested locally
- WhisperLiveKit auto-launched by the dashboard on `127.0.0.1:8000`
- Streamlit served on `127.0.0.1:8501`
- Low-latency WLK defaults: diff WebSocket output, `tiny` model, English
  language, 0.1 second minimum chunking, and 3 second buffer trimming.

Python 3.10+ should work if the dependency resolver can install compatible
PyTorch and audio wheels, but Python 3.11 or 3.12 is the safest choice on a new
machine.

## First-Time Setup

Do not copy another machine's `venv` folder. Always create the virtualenv on the
machine that will run the dashboard.

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the venv with:

```powershell
.\venv\Scripts\Activate.ps1
```

If `sounddevice` fails on Linux, install PortAudio first:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
python -m pip install -r requirements.txt
```

## Run the Dashboard

Use the virtualenv Python, not the system Python:

```bash
source venv/bin/activate
python -m streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1
```

Open:

```text
http://127.0.0.1:8501
```

Click `Start mic`.

Expected state after startup:

- `System Status` changes to `Running`
- `System Status` panel shows `Microphone active`
- `Start mic` becomes disabled
- `Stop mic` becomes enabled
- `Current Recording` shows speech once you talk
- `Current Behaviour` updates from stable live speech first, then finalizes
  after a completed utterance

The first `Start mic` can take 30-120 seconds because WhisperLiveKit may import
large audio libraries and download the first ASR model. The default model is
`tiny` for faster setup. For better transcription quality after setup works:

```bash
WLK_MODEL=small python -m streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1
```

## Useful Run Options

Force English transcription:

```bash
WLK_LANGUAGE=en python -m streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1
```

Use automatic language detection if needed. This can be slower than a fixed
language:

```bash
WLK_LANGUAGE=auto python -m streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1
```

Lower latency further on a fast machine:

```bash
PARTIAL_ANALYSIS_STABLE_SEC=0.25 PARTIAL_ANALYSIS_INTERVAL_SEC=0.5 python -m streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1
```

Prefer more conservative ASR finalization:

```bash
WLK_BUFFER_TRIMMING_SEC=8 PARTIAL_ANALYSIS_STABLE_SEC=1.0 python -m streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1
```

Give WhisperLiveKit more startup time on a slow machine:

```bash
WLK_STARTUP_TIMEOUT_SEC=300 python -m streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1
```

Use another WLK port if `8000` is busy:

```bash
WLK_PORT=8765 python -m streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1
```

## Quick Health Checks

In another terminal:

```bash
curl http://127.0.0.1:8501/_stcore/health
```

Expected:

```text
ok
```

Check that Streamlit and WhisperLiveKit are listening:

```bash
lsof -nP -iTCP:8501 -iTCP:8000 -sTCP:LISTEN
```

Expected:

```text
Python ... TCP 127.0.0.1:8501 (LISTEN)
Python ... TCP 127.0.0.1:8000 (LISTEN)
```

On Windows PowerShell:

```powershell
netstat -ano | findstr ":8501"
netstat -ano | findstr ":8000"
```

## Troubleshooting

### `Start mic` does nothing or times out

Use the latest branch and restart Streamlit:

```bash
git pull
source venv/bin/activate
python -m pip install -r requirements.txt
WLK_STARTUP_TIMEOUT_SEC=300 python -m streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1
```

Then reload the browser page before clicking `Start mic`.

### Port already in use

If port `8000` or `8501` is already occupied, stop the old process or use other
ports:

```bash
WLK_PORT=8765 python -m streamlit run dashboard.py --server.port 8502 --server.address 127.0.0.1
```

Open:

```text
http://127.0.0.1:8502
```

### Microphone permission

On macOS:

1. Open `System Settings`.
2. Go to `Privacy & Security`.
3. Open `Microphone`.
4. Enable microphone access for the terminal app being used.
5. Restart Streamlit.

On Windows:

1. Open `Settings`.
2. Go to `Privacy & security`.
3. Open `Microphone`.
4. Allow desktop apps to access the microphone.
5. Restart Streamlit.

### Dependency install fails

Confirm Python and pip:

```bash
python --version
python -m pip --version
```

Recommended: Python 3.11 or 3.12 on a new machine.

If Python or Streamlit appears to hang before printing anything, delete and
recreate `venv`. This can happen when a virtualenv was copied from another
machine or when cloud sync/offload tools leave files in the virtualenv as
placeholders.

Then reinstall cleanly:

```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Windows PowerShell equivalent:

```powershell
Remove-Item -Recurse -Force venv
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Dashboard starts but no transcript appears

Speak clearly for 3-5 seconds, then pause for 1-2 seconds. WhisperLiveKit emits
completed utterances after it detects a speech boundary.

For debugging, expand the `Debug` panel in the sidebar. Useful fields:

- `Pipeline running`
- `WLK auto-launch`
- `WLK backend`
- `WLK model`
- `WLK language`
- `WLK output mode`
- `WLK min chunk sec`
- `WLK VAC chunk sec`
- `WLK confidence validation`
- `Latest result provisional`
- `Audio dropped frames`
- `WLK queue size`
- `WLK client stats`
- `Utterances emitted`

If `bytes_sent` increases but `messages_received` stays at 0, WLK is not
responding. Restart Streamlit and try `WLK_MODEL=tiny`.

If `messages_received` increases but `committed_count` stays at 0, speak longer
and pause after the phrase.

## Local Test Command

```bash
source venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

On Windows PowerShell:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
python -m pytest -q
```

The local verification for this branch passed:

```text
115 passed
```
