from __future__ import annotations

import queue
import subprocess

from baseline_manager import BaselineManager
from dashboard_manager import DashboardManager


def _manager() -> DashboardManager:
    return DashboardManager(
        partial_queue=queue.Queue(maxsize=20),
        committed_queue=queue.Queue(maxsize=100),
        committed_display_queue=queue.Queue(maxsize=100),
        utterance_queue=queue.Queue(maxsize=20),
        baseline_manager=BaselineManager(),
    )


def test_wlk_command_uses_unbuffered_python_for_startup_logs():
    manager = _manager()

    assert manager.wlk_command[1:3] == ["-u", "-m"]


def test_wlk_command_uses_low_latency_transcription_flags():
    manager = _manager()

    command = manager.wlk_command

    assert "--min-chunk-size" in command
    assert "--vac-chunk-size" in command
    assert "--buffer_trimming_sec" in command
    assert "--confidence-validation" in command


def test_wlk_import_warmup_timeout_does_not_abort_launch(monkeypatch):
    manager = _manager()
    run_calls = []

    def fake_run(*args, **kwargs):
        run_calls.append((args, kwargs))
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr("dashboard_manager.subprocess.run", fake_run)

    class FakeLog:
        def __init__(self):
            self.messages: list[bytes] = []

        def write(self, data: bytes) -> None:
            self.messages.append(data)

        def flush(self) -> None:
            return None

    fake_log = FakeLog()
    manager._wlk_log_file = fake_log
    manager._warm_wlk_import_cache({"PYTHONUNBUFFERED": "1"})

    assert run_calls
    assert any(b"timed out" in message for message in fake_log.messages)
