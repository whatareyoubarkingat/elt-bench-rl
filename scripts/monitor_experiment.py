"""Supervise one owned experiment process group without making API calls.

Example: python scripts/monitor_experiment.py --run-dir runs/experiment \
    --timeout 3600 --stall-timeout 900 -- python -m elt_bench_rl.cli train \
    --output runs/experiment ...

Exit codes: child exit code, 124 overall timeout, 125 stall, 130 interruption,
127 launch failure. Monitoring completion does not establish experiment success.
"""

from __future__ import annotations

import argparse
import fcntl
import html
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ARTIFACT_NAMES = frozenset(
    {
        "manifest.json",
        "status.json",
        "run_status.json",
        "progress.json",
        "progress.jsonl",
        "metrics.jsonl",
        "training_result.json",
        "rollout_result.json",
        "checkpoints.jsonl",
        "train_rollouts.jsonl",
        "rollout.json",
        "tools.json",
        "grade.json",
        "dbt.log",
        "run_results.json",
        "warehouse.duckdb",
    }
)
KNOWN_STATES = frozenset(
    {
        "started",
        "running",
        "completed",
        "succeeded",
        "failed",
        "error",
        "verified_update",
        "update_unverified",
        "no_learning_signal",
        "stopped",
    }
)


def _private_file(path: Path, flags: int) -> int:
    descriptor = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    os.fchmod(descriptor, 0o600)
    return descriptor


def _atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def artifact_snapshot(run_dir: Path, log_path: Path) -> tuple[dict, dict]:
    """Inspect trusted artifact metadata, excluding the watchdog's own outputs."""
    fingerprint, details = {}, {}
    for path in run_dir.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if path != log_path and path.name not in ARTIFACT_NAMES:
            continue
        try:
            info = path.stat()
            relative = str(path.relative_to(run_dir))
            fingerprint[relative] = (info.st_size, info.st_mtime_ns)
            item = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
            if path.name in {
                "manifest.json",
                "status.json",
                "run_status.json",
                "training_result.json",
                "rollout_result.json",
            }:
                if info.st_size <= 1_000_000:
                    try:
                        value = json.loads(path.read_text())
                        item["json_valid"] = isinstance(value, dict)
                        if (
                            isinstance(value, dict)
                            and value.get("status") in KNOWN_STATES
                        ):
                            item["reported_status"] = value["status"]
                    except (OSError, ValueError, TypeError):
                        item["json_valid"] = False
                else:
                    item["json_valid"] = None
            details[relative] = item
        except OSError:
            continue
    return fingerprint, details


def process_cpu_seconds(pid: int) -> float | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "time=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode:
            return None
        value = result.stdout.strip()
        days = 0
        if "-" in value:
            day, value = value.split("-", 1)
            days = int(day)
        fields = [float(part) for part in value.split(":")]
        if len(fields) not in {2, 3}:
            return None
        total = fields[-1] + fields[-2] * 60 + days * 86400
        return total + (fields[0] * 3600 if len(fields) == 3 else 0)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _group_active(group_id: int) -> bool | None:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pgid=,stat="],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        if result.returncode:
            return None
        return any(
            int(fields[0]) == group_id and not fields[1].startswith("Z")
            for line in result.stdout.splitlines()
            if len(fields := line.split()) == 2
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def terminate_owned_group(child: subprocess.Popen, grace: float) -> list[str]:
    """Keep the leader unreaped until signaling ends, preventing PID reuse."""
    if child.poll() is not None:
        return []
    try:
        if os.getpgid(child.pid) != child.pid:
            return ["ownership_unconfirmed"]
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return []
    signals = ["SIGTERM"]
    deadline = time.monotonic() + grace
    # Do not poll/wait and reap the leader before the final group signal.
    while time.monotonic() < deadline:
        if _group_active(child.pid) is False:
            break
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    try:
        if _group_active(child.pid) is not False:
            os.killpg(child.pid, signal.SIGKILL)
            signals.append("SIGKILL")
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        signals.append("termination_unconfirmed")
    return signals


def _write_status(run_dir: Path, record: dict) -> None:
    _atomic_json(run_dir / "monitor-status.json", record)
    with os.fdopen(
        _private_file(
            run_dir / "monitor-heartbeats.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND
        ),
        "a",
    ) as stream:
        stream.write(json.dumps(record) + "\n")
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(record.get(key)))}</td></tr>"
        for key in (
            "state",
            "child_pid",
            "process_group_id",
            "elapsed_seconds",
            "last_progress_age_seconds",
            "log_bytes",
            "child_cpu_seconds",
            "process_exit_code",
            "termination_reason",
        )
    )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="15"><title>Experiment watchdog</title>'
        "<style>body{font:18px system-ui;margin:40px;background:#f6f8fa;color:#172b4d}"
        "table{background:white;border-collapse:collapse}td{padding:12px 20px;border-bottom:1px solid #ddd}"
        "h1{font-size:28px}small{color:#555}</style><h1>Experiment watchdog</h1>"
        f"<small>Process supervision only. Updated {html.escape(record['updated_at'])}</small>"
        f"<table>{rows}</table></html>"
    )
    with os.fdopen(
        _private_file(run_dir / "monitor.html", os.O_WRONLY | os.O_CREAT | os.O_TRUNC),
        "w",
    ) as stream:
        stream.write(document)
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "state",
                    "elapsed_seconds",
                    "last_progress_age_seconds",
                    "log_bytes",
                )
            }
        ),
        flush=True,
    )


def monitor_command(
    command: list[str],
    run_dir: Path,
    *,
    heartbeat: float = 15,
    stall_timeout: float = 900,
    timeout: float = 3600,
    grace: float = 10,
) -> dict:
    for value in (heartbeat, stall_timeout, timeout, grace):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Watchdog intervals must be finite and positive")
    if not command:
        raise ValueError("An experiment command is required")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir.chmod(0o700)
    lock_fd = _private_file(run_dir / ".monitor.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        raise RuntimeError(
            "An experiment watchdog already owns this run directory"
        ) from None
    log_path = run_dir / "stdout.log"
    environment = os.environ.copy()
    environment.pop("ALL_PROXY", None)
    environment.pop("all_proxy", None)
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    record = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "limits_seconds": {
            "heartbeat": heartbeat,
            "stall": stall_timeout,
            "overall": timeout,
            "termination_grace": grace,
        },
        "monitor_pid": os.getpid(),
        "child_pid": None,
        "process_group_id": None,
        "executable": Path(command[0]).name,
        "state": "starting",
        "termination_reason": None,
        "process_exit_code": None,
        "monitor_exit_code": None,
        "signals_sent": [],
        "validation": "process_supervision_only",
        "last_progress_age_seconds": 0.0,
        "child_cpu_seconds": None,
        "log_bytes": 0,
        "artifacts": {},
    }
    child = None
    previous_handlers = {}
    try:

        def interrupt(signum, frame):
            raise KeyboardInterrupt

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, interrupt)
        with os.fdopen(
            _private_file(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND),
            "ab",
            buffering=0,
        ) as log:
            try:
                child = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
            except OSError as error:
                record.update(
                    state="launch_failed",
                    monitor_exit_code=127,
                    error_type=type(error).__name__,
                )
            if child is not None:
                record.update(
                    child_pid=child.pid, process_group_id=child.pid, state="running"
                )
                fingerprint, _ = artifact_snapshot(run_dir, log_path)
                previous_cpu = 0.0
                last_progress = started
                next_heartbeat = started
                while True:
                    now = time.monotonic()
                    current, artifacts = artifact_snapshot(run_dir, log_path)
                    cpu = process_cpu_seconds(child.pid)
                    if current != fingerprint or (
                        cpu is not None and cpu > previous_cpu
                    ):
                        last_progress = now
                    fingerprint = current
                    if cpu is not None:
                        previous_cpu = cpu
                    exit_code = child.poll()
                    if exit_code is not None:
                        record.update(
                            state="completed" if exit_code == 0 else "failed",
                            process_exit_code=exit_code,
                            monitor_exit_code=exit_code
                            if exit_code >= 0
                            else 128 - exit_code,
                        )
                    elif now - started >= timeout:
                        record.update(
                            state="timed_out",
                            termination_reason="overall_timeout",
                            monitor_exit_code=124,
                        )
                    elif now - last_progress >= stall_timeout:
                        record.update(
                            state="stalled",
                            termination_reason="no_observed_progress",
                            monitor_exit_code=125,
                        )
                    record.update(
                        updated_at=datetime.now(UTC).isoformat(),
                        elapsed_seconds=round(now - started, 3),
                        last_progress_age_seconds=round(now - last_progress, 3),
                        child_cpu_seconds=cpu,
                        log_bytes=log_path.stat().st_size,
                        artifacts=artifacts,
                    )
                    if record["state"] != "running":
                        if exit_code is None:
                            record["signals_sent"] = terminate_owned_group(child, grace)
                            record["process_exit_code"] = child.returncode
                        break
                    if now >= next_heartbeat:
                        _write_status(run_dir, record)
                        next_heartbeat = now + heartbeat
                    time.sleep(
                        min(
                            5.0,
                            heartbeat,
                            max(0.001, timeout - (now - started)),
                            max(0.001, stall_timeout - (now - last_progress)),
                        )
                    )
    except KeyboardInterrupt:
        record.update(
            state="interrupted",
            termination_reason="monitor_interrupted",
            monitor_exit_code=130,
        )
        if child is not None:
            record["signals_sent"] = terminate_owned_group(child, grace)
            record["process_exit_code"] = child.returncode
    except Exception as error:  # noqa: BLE001 - any monitor failure must clean up its owned child.
        record.update(
            state="monitor_error",
            termination_reason="watchdog_error",
            monitor_exit_code=126,
            error_type=type(error).__name__,
        )
        if child is not None:
            record["signals_sent"] = terminate_owned_group(child, grace)
            record["process_exit_code"] = child.returncode
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        try:
            record.update(
                updated_at=datetime.now(UTC).isoformat(),
                elapsed_seconds=round(time.monotonic() - started, 3),
                log_bytes=log_path.stat().st_size if log_path.exists() else 0,
                artifacts=artifact_snapshot(run_dir, log_path)[1],
            )
            _write_status(run_dir, record)
        finally:
            os.close(lock_fd)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--heartbeat", type=float, default=15)
    parser.add_argument("--stall-timeout", type=float, default=900)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--grace", type=float, default=10)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = (
        arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    )
    try:
        result = monitor_command(
            command,
            arguments.run_dir,
            heartbeat=arguments.heartbeat,
            stall_timeout=arguments.stall_timeout,
            timeout=arguments.timeout,
            grace=arguments.grace,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {"state": "monitor_setup_failed", "error_type": type(error).__name__}
            ),
            flush=True,
        )
        return 126
    return result["monitor_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
