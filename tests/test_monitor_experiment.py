"""Offline tests for local supervision; these never exercise Tinker APIs."""

from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/monitor_experiment.py"
SPEC = importlib.util.spec_from_file_location("monitor_experiment", SCRIPT)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


@pytest.fixture(autouse=True)
def fixed_cpu(monkeypatch):
    monkeypatch.setattr(monitor, "process_cpu_seconds", lambda _: 0.0)


def run(code, output, **kwargs):
    return monitor.monitor_command(
        [sys.executable, "-c", code],
        output,
        heartbeat=0.03,
        stall_timeout=0.4,
        timeout=3,
        grace=0.1,
        **kwargs,
    )


def test_success_preserves_proxy_policy_private_log_and_omits_command_arguments(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ALL_PROXY", "private-test-marker")
    monkeypatch.setenv("all_proxy", "private-test-marker")
    monkeypatch.setenv("HTTPS_PROXY", "https://offline.invalid")
    result = run(
        "import os; print('private-test-marker'); assert 'ALL_PROXY' not in os.environ; assert 'all_proxy' not in os.environ; assert os.environ['HTTPS_PROXY']=='https://offline.invalid'; assert os.environ['PYTHONUNBUFFERED']=='1'",
        tmp_path,
    )
    assert result["state"] == "completed"
    assert result["monitor_exit_code"] == 0
    assert result["child_pid"] == result["process_group_id"]
    assert "private-test-marker" in (tmp_path / "stdout.log").read_text()
    for name in ("monitor-status.json", "monitor-heartbeats.jsonl", "monitor.html"):
        assert "private-test-marker" not in (tmp_path / name).read_text()
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "stdout.log").stat().st_mode & 0o777 == 0o600
    assert tmp_path.stat().st_mode & 0o777 == 0o700


def test_nonzero_exit_is_preserved_with_output(tmp_path):
    result = run("import sys; print('controlled failure'); sys.exit(7)", tmp_path)
    assert result["state"] == "failed"
    assert result["process_exit_code"] == result["monitor_exit_code"] == 7
    assert "controlled failure" in (tmp_path / "stdout.log").read_text()


def test_heartbeat_writes_do_not_extend_stall_or_kill_unrelated_process(tmp_path):
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        result = monitor.monitor_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            heartbeat=0.03,
            stall_timeout=0.18,
            timeout=3,
            grace=0.1,
        )
        assert result["state"] == "stalled"
        assert result["termination_reason"] == "no_observed_progress"
        assert result["monitor_exit_code"] == 125
        assert result["signals_sent"] == ["SIGTERM"]
        assert unrelated.poll() is None
        assert len((tmp_path / "monitor-heartbeats.jsonl").read_text().splitlines()) > 2
        assert result["last_progress_age_seconds"] >= 0.18
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_overall_timeout_applies_despite_continuous_log_progress(tmp_path):
    result = monitor.monitor_command(
        [
            sys.executable,
            "-c",
            "import time\nwhile True:\n print('progress',flush=True)\n time.sleep(.02)",
        ],
        tmp_path,
        heartbeat=0.03,
        stall_timeout=0.2,
        timeout=0.3,
        grace=0.1,
    )
    assert result["state"] == "timed_out"
    assert result["monitor_exit_code"] == 124
    assert result["termination_reason"] == "overall_timeout"
    assert result["log_bytes"] > 0


def test_actual_progress_artifacts_keep_quiet_child_alive(tmp_path):
    code = f"from pathlib import Path\nimport time\np=Path({str(tmp_path)!r})\nfor i in range(8):\n (p/'progress.jsonl').write_text(str(i))\n (p/'tools.json').write_text('{{}}')\n time.sleep(.04)"
    result = monitor.monitor_command(
        [sys.executable, "-c", code],
        tmp_path,
        heartbeat=0.025,
        stall_timeout=0.15,
        timeout=3,
        grace=0.1,
    )
    assert result["state"] == "completed"
    assert "progress.jsonl" in result["artifacts"]
    assert "tools.json" in result["artifacts"]
    assert result["log_bytes"] == 0


def test_artifact_snapshot_excludes_monitor_files_and_secrets(tmp_path):
    (tmp_path / "monitor-status.json").write_text('{"status":"running"}')
    (tmp_path / "monitor-heartbeats.jsonl").write_text("{}")
    (tmp_path / ".env").write_text("private-test-marker")
    (tmp_path / "status.json").write_text(
        '{"status":"running","token":"private-test-marker"}'
    )
    (tmp_path / "manifest.json").write_text("incomplete")
    fingerprint, details = monitor.artifact_snapshot(tmp_path, tmp_path / "stdout.log")
    assert set(fingerprint) == {"status.json", "manifest.json"}
    assert details["status.json"]["reported_status"] == "running"
    assert details["manifest.json"]["json_valid"] is False
    assert "private-test-marker" not in json.dumps(details)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_intervals_never_launch(tmp_path, monkeypatch, value):
    monkeypatch.setattr(
        monitor.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch")
    )
    with pytest.raises(ValueError, match="finite and positive"):
        monitor.monitor_command([sys.executable], tmp_path, timeout=value)


def test_launch_failure_is_recorded_without_exception_text(tmp_path):
    result = monitor.monitor_command(
        ["/missing/executable", "private-test-marker"],
        tmp_path,
        heartbeat=0.03,
        stall_timeout=1,
        timeout=3,
        grace=0.1,
    )
    assert result["state"] == "launch_failed"
    assert result["monitor_exit_code"] == 127
    assert result["error_type"] == "FileNotFoundError"
    assert "private-test-marker" not in (tmp_path / "monitor-status.json").read_text()


def test_group_ownership_mismatch_never_signals(monkeypatch):
    child = SimpleNamespace(pid=12345, poll=lambda: None)
    monkeypatch.setattr(monitor.os, "getpgid", lambda _: 23456)
    monkeypatch.setattr(
        monitor.os, "killpg", lambda *a: pytest.fail("must not signal foreign group")
    )
    assert monitor.terminate_owned_group(child, 0.1) == ["ownership_unconfirmed"]


def test_timeout_kills_owned_descendant_that_ignores_sigterm(tmp_path):
    pid_path = tmp_path / "descendant.pid"
    descendant_code = f"import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(30)"
    parent_code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{descendant_code!r}]); time.sleep(30)"
    result = monitor.monitor_command(
        [sys.executable, "-c", parent_code],
        tmp_path,
        heartbeat=0.03,
        stall_timeout=0.3,
        timeout=3,
        grace=0.1,
    )
    assert result["signals_sent"] == ["SIGTERM", "SIGKILL"]
    descendant = int(pid_path.read_text())
    for _ in range(50):
        check = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(descendant)],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode or check.stdout.strip().startswith("Z"):
            break
        time.sleep(0.02)
    else:
        pytest.fail("Owned descendant survived group termination")


def test_sigterm_of_watchdog_cleans_owned_child_and_records_interruption(tmp_path):
    watchdog = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            str(tmp_path),
            "--heartbeat",
            ".03",
            "--grace",
            ".1",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while (
            not (tmp_path / "monitor-status.json").exists()
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert (tmp_path / "monitor-status.json").exists()
        watchdog.send_signal(signal.SIGTERM)
        assert watchdog.wait(timeout=5) == 130
        report = json.loads((tmp_path / "monitor-status.json").read_text())
        assert report["state"] == "interrupted"
        assert report["process_exit_code"] is not None
    finally:
        if watchdog.poll() is None:
            watchdog.terminate()
            watchdog.wait(timeout=5)
