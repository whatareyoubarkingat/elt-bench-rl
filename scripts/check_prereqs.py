#!/usr/bin/env python3
"""Check local prerequisites without making service requests or exposing secrets.

Run with the project virtualenv. Missing credentials are reported but do not
change the exit code; a missing local prerequisite does. --skip-docker checks
the Python/data prerequisites while explicitly leaving Docker unverified.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    ("elt_bench_rl", "elt-bench-rl"),
    ("torch", "torch"),
    ("tinker", "tinker"),
    ("tinker_cookbook", "tinker-cookbook"),
    ("tinker_cookbook.rl.train", "tinker-cookbook"),
    ("duckdb", "duckdb"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("datasets", "datasets"),
    ("huggingface_hub", "huggingface-hub"),
    ("snowflake.connector", "snowflake-connector-python"),
    ("databricks.sql", "databricks-sql-connector"),
    ("databricks.sdk", "databricks-sdk"),
    ("pytest", "pytest"),
    ("requests", "requests"),
    ("yaml", "PyYAML"),
    ("dotenv", "python-dotenv"),
)
SUBMODULES = ("third_party/elt-bench", "third_party/tinker-cookbook")
DATA_FILES = {
    "sources": {"data/trains/train.csv", "data/trains/cars.csv"},
    "ground_truth": {"ground_truth/gt_databricks/trains/trains.csv"},
}


def command_environment() -> dict[str, str]:
    # Direct .venv/bin/python invocation does not activate the virtualenv PATH.
    # Homebrew keeps libpq's psql in a keg-specific bin directory.
    environment = os.environ.copy()
    paths = [str(ROOT / ".venv/bin"), "/opt/homebrew/opt/libpq/bin",
             environment.get("PATH", os.defpath)]
    environment["PATH"] = os.pathsep.join(paths)
    return environment


def executable(name: str) -> str | None:
    return shutil.which(name, path=command_environment()["PATH"])


def command(args: list[str], timeout: int = 15) -> subprocess.CompletedProcess[str]:
    # Never print stderr or arbitrary command output: it may contain credentials.
    return subprocess.run(
        args, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False,
        env=command_environment(),
    )


def check(name: str, ready: bool, *, required: bool = True, **details: Any) -> dict:
    status = "ready" if ready else "local_not_ready" if required else "optional_missing"
    return {"name": name, "status": status,
            "required": required, **details}


def python_checks() -> list[dict]:
    results = [check("python", sys.version_info >= (3, 11),
                     version=".".join(map(str, sys.version_info[:3])))]
    for module, distribution in PACKAGES:
        try:
            # Suppress third-party import warnings and messages, including in JSON mode.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                imported = importlib.import_module(module)
                version = importlib.metadata.version(distribution)
                if module == "torch":
                    assert imported.tensor([1, 2, 3], device="cpu").sum().item() == 6
                elif module == "duckdb":
                    connection = imported.connect(":memory:")
                    try:
                        assert connection.execute(
                            "SELECT sum(i) FROM range(1, 4) AS t(i)"
                        ).fetchone() == (6,)
                    finally:
                        connection.close()
            results.append(check(module, True, version=version,
                                 smoke_test=module in {"torch", "duckdb"}))
        except Exception as exc:
            results.append(check(module, False, error_type=type(exc).__name__))
    return results


def submodule_checks() -> list[dict]:
    results = []
    for path in SUBMODULES:
        try:
            pin = command(["git", "ls-files", "--stage", "--", path])
            fields = pin.stdout.strip().split()
            registered = pin.returncode == 0 and len(fields) >= 4 and fields[0] == "160000"
            expected_sha = fields[1] if registered else None
            actual = command(["git", "-C", str(ROOT / path), "rev-parse", "HEAD"])
            actual_sha = actual.stdout.strip() if actual.returncode == 0 else None
            if actual_sha and not re.fullmatch(r"[0-9a-f]{40,64}", actual_sha):
                actual_sha = None
            dirty = command(["git", "-C", str(ROOT / path), "status", "--porcelain",
                             "--untracked-files=no"])
            tracked_changes = dirty.returncode != 0 or bool(dirty.stdout.strip())
            ready = registered and actual_sha == expected_sha and not tracked_changes
            results.append(check(path, ready, registered=registered,
                                 expected_sha=expected_sha, actual_sha=actual_sha,
                                 tracked_changes=tracked_changes))
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append(check(path, False, error_type=type(exc).__name__))
    return results


def data_checks() -> list[dict]:
    manifest_path = ROOT / "data/manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("Unsupported manifest")
        if manifest.get("task_id") != "trains":
            raise ValueError("Unexpected selected task")
        results = [check("data/manifest.json", True, task_id="trains")]
        for group, required_paths in DATA_FILES.items():
            entries = manifest.get(group)
            if not isinstance(entries, list):
                raise ValueError("Missing manifest group")
            by_path = {entry.get("path"): entry for entry in entries if isinstance(entry, dict)}
            for relative in sorted(required_paths):
                entry = by_path.get(relative)
                if not isinstance(entry, dict):
                    results.append(check(relative, False, reason="missing_manifest_entry"))
                    continue
                # Paths come from a fixed allowlist; never follow manifest paths outside ROOT.
                path = ROOT / relative
                try:
                    resolved = path.resolve()
                    resolved.relative_to(ROOT.resolve())
                    expected_bytes = entry.get("bytes")
                    expected_hash = entry.get("sha256")
                    expected_rows = entry.get("rows")
                    metadata_valid = (
                        isinstance(expected_bytes, int) and expected_bytes > 0
                        and isinstance(expected_hash, str)
                        and re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None
                        and isinstance(expected_rows, int) and expected_rows > 0
                    )
                    if not metadata_valid:
                        raise ValueError("Invalid file metadata")
                    digest = hashlib.sha256()
                    with path.open("rb") as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(chunk)
                    size_matches = path.stat().st_size == expected_bytes
                    hash_matches = digest.hexdigest() == expected_hash
                    with path.open(encoding="utf-8-sig", newline="") as source:
                        reader = csv.reader(source)
                        next(reader)  # All selected files contain a header.
                        rows = sum(1 for row in reader if row)
                    rows_match = rows == expected_rows
                    results.append(check(relative, size_matches and hash_matches and rows_match,
                                         bytes=path.stat().st_size, rows=rows,
                                         sha256_matches=hash_matches,
                                         size_matches=size_matches, rows_match=rows_match))
                except (OSError, ValueError, UnicodeError, StopIteration, csv.Error) as exc:
                    results.append(check(relative, False, error_type=type(exc).__name__))
        return results
    except (OSError, ValueError, TypeError) as exc:
        return [check("data/manifest.json", False, error_type=type(exc).__name__)]


def docker_checks(context: str, skip: bool) -> list[dict]:
    if skip:
        return [{"name": "docker", "status": "skipped", "required": False,
                 "context": context, "reason": "requested_with_skip_docker"}]
    if not executable("docker"):
        return [check("docker", False, context=context, reason="command_not_found")]
    # Whitelist fields rather than returning all info (which can include proxy settings).
    template = ('{"server_version":{{json .ServerVersion}},"os_type":{{json .OSType}},'
                '"architecture":{{json .Architecture}},"cpus":{{json .NCPU}},'
                '"memory_bytes":{{json .MemTotal}}}')
    try:
        result = command(["docker", "--context", context, "info", "--format", template], 30)
        if result.returncode != 0:
            return [check("docker", False, context=context, reason="daemon_unreachable",
                          returncode=result.returncode)]
        info = json.loads(result.stdout)
        if not isinstance(info, dict):
            raise ValueError("Invalid Docker response")
        allowed = {key: info.get(key) for key in (
            "server_version", "os_type", "architecture", "cpus", "memory_bytes"
        )}
        return [check("docker", True, context=context, **allowed)]
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return [check("docker", False, context=context, error_type=type(exc).__name__)]


def abctl_checks(skip: bool) -> list[dict]:
    if skip:
        return [{"name": "abctl", "status": "skipped", "required": False,
                 "reason": "requested_with_skip_docker"}]
    abctl = executable("abctl")
    if not abctl:
        return [check("abctl", False, reason="command_not_found")]
    try:
        result = command([abctl, "version"])
        version = re.search(r"\bv?\d+\.\d+\.\d+(?:[-+][\w.-]+)?\b", result.stdout)
        ready = result.returncode == 0 and version is not None
        return [check("abctl", ready,
                      version=version.group(0) if version else None,
                      returncode=result.returncode)]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [check("abctl", False, error_type=type(exc).__name__)]


def optional_cli_checks() -> list[dict]:
    results = []
    for name, arguments in (("dbt", ["--version"]),
                            ("terraform", ["version"]),
                            ("psql", ["--version"])):
        tool_path = executable(name)
        if not tool_path:
            results.append(check(name, False, required=False, reason="command_not_found"))
            continue
        try:
            result = command([tool_path, *arguments])
            version = re.search(r"\bv?\d+\.\d+(?:\.\d+)?(?:[-+][\w.-]+)?\b", result.stdout)
            ready = result.returncode == 0 and version is not None
            results.append(check(name, ready, required=False,
                                 version=version.group(0) if version else None,
                                 returncode=result.returncode))
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append(check(name, False, required=False, error_type=type(exc).__name__))
    return results


def credential_checks() -> dict:
    # Read .env without mutating process environment, interpolation, or API calls.
    # Only booleans derived from credentials are retained in the public report.
    values: dict[str, Any] = {}
    env_file_error = None
    if (ROOT / ".env").is_file():
        try:
            from dotenv import dotenv_values
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                values.update(dotenv_values(ROOT / ".env", interpolate=False))
        except Exception as exc:
            env_file_error = type(exc).__name__
    values.update(os.environ)
    names = (
        "TINKER_API_KEY", "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_PRIVATE_KEY_PATH", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA", "SNOWFLAKE_ROLE", "AIRBYTE_CLIENT_ID", "AIRBYTE_CLIENT_SECRET",
        "AIRBYTE_WORKSPACE_ID",
        "HF_TOKEN",
        "DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_CATALOG",
        "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET",
    )
    presence = {name: bool(str(values.get(name) or "").strip()) for name in names}
    tinker_ready = presence["TINKER_API_KEY"]
    snowflake_ready = (
        presence["SNOWFLAKE_ACCOUNT"] and presence["SNOWFLAKE_USER"]
        and (presence["SNOWFLAKE_PASSWORD"] or presence["SNOWFLAKE_PRIVATE_KEY_PATH"])
        and presence["SNOWFLAKE_WAREHOUSE"]
    )
    airbyte_ready = (presence["AIRBYTE_CLIENT_ID"] and presence["AIRBYTE_CLIENT_SECRET"]
                     and presence["AIRBYTE_WORKSPACE_ID"])
    databricks_ready = all(presence[name] for name in ("DATABRICKS_HOST", "DATABRICKS_HTTP_PATH",
                         "DATABRICKS_CATALOG", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"))
    return {
        "status": "ready" if tinker_ready and databricks_ready and airbyte_ready else "missing_credential",
        "presence": presence,
        "tinker_config_present": tinker_ready,
        "snowflake_config_present": snowflake_ready,
        "databricks_config_present": databricks_ready,
        "airbyte_config_present": airbyte_ready,
        "env_file_error_type": env_file_error,
        "verified": False,
        "note": "Presence only; no account access, paid API, or RL optimization was attempted.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print one JSON report.")
    parser.add_argument("--skip-docker", action="store_true",
                        help="Leave Docker and abctl unverified; check the Python/data subset.")
    parser.add_argument("--docker-context",
                        default=os.environ.get("ELT_BENCH_RL_DOCKER_CONTEXT", "colima-elt-bench-rl"))
    args = parser.parse_args()
    checks = (python_checks() + submodule_checks() + data_checks()
              + docker_checks(args.docker_context, args.skip_docker)
              + abctl_checks(args.skip_docker) + optional_cli_checks())
    local_ready = all(item["status"] == "ready" for item in checks if item["required"])
    credentials = credential_checks()
    status = ("local_not_ready" if not local_ready else credentials["status"])
    report = {
        "schema_version": 1, "status": status, "local_ready": local_ready,
        "scope": "python_and_data" if args.skip_docker else "full_local_prerequisites",
        "checks": checks, "credentials": credentials,
        "integration_test_run": False, "official_rollout_run": False,
        "rl_optimization_run": False,
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Local prerequisites: {'READY' if local_ready else 'NOT READY'} ({report['scope']})")
        for item in checks:
            detail = item.get("version") or item.get("reason") or item.get("error_type") or ""
            if item["name"] in SUBMODULES:
                detail = f"expected={item.get('expected_sha')} actual={item.get('actual_sha')}"
            elif item["name"] == "docker" and item["status"] == "ready":
                detail = f"context={item['context']} server={item['server_version']}"
            print(f"  [{item['status']}] {item['name']}{': ' + detail if detail else ''}")
        print(f"Credentials: {credentials['status']} (presence only; no API calls)")
        for name, present in credentials["presence"].items():
            print(f"  {name}: {str(present).lower()}")
        if credentials["env_file_error_type"]:
            print(f"  .env parser: {credentials['env_file_error_type']}")
        print("This checks prerequisites only; integration, official rollout, and RL remain unverified.")
    return 0 if local_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
