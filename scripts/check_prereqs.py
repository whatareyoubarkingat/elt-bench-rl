"""Check DuckDB/CSV/HTTP/dbt/Tinker prerequisites without exposing secrets.

Run with the project virtualenv. Missing credentials are reported but do not
change the exit code; a missing local prerequisite does. Cloud packages and
Docker/Airbyte tooling are checked only with --check-cloud, remain optional,
and never trigger cloud account access or paid API calls.
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
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    ("elt_bench_rl", "elt-bench-rl"),
    ("elt_bench_rl.ingestion", "elt-bench-rl"),
    ("elt_bench_rl.factory", "elt-bench-rl"),
    ("elt_bench_rl.training", "elt-bench-rl"),
    ("torch", "torch"),
    ("tinker", "tinker"),
    ("tinker_cookbook", "tinker-cookbook"),
    ("tinker_cookbook.rl.train", "tinker-cookbook"),
    ("duckdb", "duckdb"),
    ("dbt.adapters.duckdb", "dbt-duckdb"),
    ("sqlglot", "sqlglot"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("datasets", "datasets"),
    ("huggingface_hub", "huggingface-hub"),
    ("requests", "requests"),
    ("yaml", "PyYAML"),
    ("dotenv", "python-dotenv"),
)
CLOUD_PACKAGES = (
    ("snowflake.connector", "snowflake-connector-python"),
    ("dbt.adapters.snowflake", "dbt-snowflake"),
    ("databricks.sql", "databricks-sql-connector"),
    ("databricks.sdk", "databricks-sdk"),
    ("dbt.adapters.databricks", "dbt-databricks"),
)
UPSTREAM_NAMES = ("elt-bench", "tinker-cookbook")
DATA_FILES = {
    "sources": {"data/trains/train.csv", "data/trains/cars.csv"},
    "ground_truth": {"ground_truth/gt_databricks/trains/trains.csv"},
}


def command_environment() -> dict[str, str]:
    # Direct .venv/bin/python invocation does not activate the virtualenv PATH.
    # Homebrew keeps libpq's psql in a keg-specific bin directory.
    environment = os.environ.copy()
    paths = [
        str(ROOT / ".venv/bin"),
        "/opt/homebrew/opt/libpq/bin",
        environment.get("PATH", os.defpath),
    ]
    environment["PATH"] = os.pathsep.join(paths)
    return environment


def executable(name: str) -> str | None:
    return shutil.which(name, path=command_environment()["PATH"])


def command(args: list[str], timeout: int = 15) -> subprocess.CompletedProcess[str]:
    # Never print stderr or arbitrary command output: it may contain credentials.
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=command_environment(),
    )


def check(name: str, ready: bool, *, required: bool = True, **details: Any) -> dict:
    status = "ready" if ready else "local_not_ready" if required else "optional_missing"
    return {"name": name, "status": status, "required": required, **details}


def python_checks(include_cloud: bool = False) -> list[dict]:
    results = [
        check(
            "python",
            sys.version_info >= (3, 11),
            version=".".join(map(str, sys.version_info[:3])),
        )
    ]
    packages = [(module, distribution, True) for module, distribution in PACKAGES]
    if include_cloud:
        packages.extend(
            (module, distribution, False) for module, distribution in CLOUD_PACKAGES
        )
    for module, distribution, required in packages:
        try:
            # Suppress third-party import warnings and messages, including in JSON mode.
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
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
            results.append(
                check(
                    module,
                    True,
                    required=required,
                    version=version,
                    smoke_test=module in {"torch", "duckdb"},
                )
            )
        except Exception as exc:  # noqa: BLE001 - report arbitrary third-party import failures safely.
            results.append(
                check(module, False, required=required, error_type=type(exc).__name__)
            )
    return results


def upstream_checks() -> list[dict]:
    try:
        from elt_bench_rl.upstreams import resolve_upstream

        manifest = json.loads((ROOT / "configs/upstream-revisions.json").read_text())
    except (ImportError, OSError, ValueError) as exc:
        return [check("upstream_manifest", False, error_type=type(exc).__name__)]
    results = []
    for name in UPSTREAM_NAMES:
        try:
            expected_sha = manifest[name]["revision"]
            path = resolve_upstream(ROOT, name)
            actual = command(
                ["git", "--no-optional-locks", "-C", str(path), "rev-parse", "HEAD"]
            )
            actual_sha = actual.stdout.strip() if actual.returncode == 0 else None
            if actual_sha and not re.fullmatch(r"[0-9a-f]{40,64}", actual_sha):
                actual_sha = None
            dirty = command(
                [
                    "git",
                    "--no-optional-locks",
                    "-C",
                    str(path),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ]
            )
            checkout_dirty = dirty.returncode != 0 or bool(dirty.stdout.strip())
            ready = actual_sha == expected_sha and not checkout_dirty
            results.append(
                check(
                    name,
                    ready,
                    cache_path=manifest[name]["cache_path"],
                    expected_sha=expected_sha,
                    actual_sha=actual_sha,
                    checkout_dirty=checkout_dirty,
                )
            )
        except Exception as exc:  # noqa: BLE001 - resolver failures are reported without arbitrary messages.
            results.append(check(name, False, error_type=type(exc).__name__))
    try:
        distribution = importlib.metadata.distribution("tinker-cookbook")
        origin = json.loads(distribution.read_text("direct_url.json") or "{}")
        vcs = origin.get("vcs_info", {})
        expected = manifest["tinker-cookbook"]
        ready = (
            origin.get("url") == expected["url"]
            and vcs.get("vcs") == "git"
            and vcs.get("commit_id") == expected["revision"]
            and not origin.get("dir_info", {}).get("editable", False)
        )
        results.append(
            check(
                "tinker-cookbook_installed_origin",
                ready,
                version=distribution.version,
                expected_sha=expected["revision"],
                actual_sha=vcs.get("commit_id"),
                pinned_git_source=ready,
            )
        )
    except (
        importlib.metadata.PackageNotFoundError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
    ) as exc:
        results.append(
            check(
                "tinker-cookbook_installed_origin", False, error_type=type(exc).__name__
            )
        )
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
                raise TypeError("Missing manifest group")
            by_path = {
                entry.get("path"): entry for entry in entries if isinstance(entry, dict)
            }
            for relative in sorted(required_paths):
                entry = by_path.get(relative)
                if not isinstance(entry, dict):
                    results.append(
                        check(relative, False, reason="missing_manifest_entry")
                    )
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
                        isinstance(expected_bytes, int)
                        and expected_bytes > 0
                        and isinstance(expected_hash, str)
                        and re.fullmatch(r"[0-9a-f]{64}", expected_hash) is not None
                        and isinstance(expected_rows, int)
                        and expected_rows > 0
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
                    results.append(
                        check(
                            relative,
                            size_matches and hash_matches and rows_match,
                            bytes=path.stat().st_size,
                            rows=rows,
                            sha256_matches=hash_matches,
                            size_matches=size_matches,
                            rows_match=rows_match,
                        )
                    )
                except (
                    OSError,
                    ValueError,
                    UnicodeError,
                    StopIteration,
                    csv.Error,
                ) as exc:
                    results.append(
                        check(relative, False, error_type=type(exc).__name__)
                    )
        return results
    except (OSError, ValueError, TypeError) as exc:
        return [check("data/manifest.json", False, error_type=type(exc).__name__)]


def docker_checks(context: str, skip: bool) -> list[dict]:
    if skip:
        return [
            {
                "name": "docker",
                "status": "skipped",
                "required": False,
                "context": context,
                "reason": "requested_with_skip_docker",
            }
        ]
    if not executable("docker"):
        return [
            check(
                "docker",
                False,
                required=False,
                context=context,
                reason="command_not_found",
            )
        ]
    # Whitelist fields rather than returning all info (which can include proxy settings).
    template = (
        '{"server_version":{{json .ServerVersion}},"os_type":{{json .OSType}},'
        '"architecture":{{json .Architecture}},"cpus":{{json .NCPU}},'
        '"memory_bytes":{{json .MemTotal}}}'
    )
    try:
        result = command(
            ["docker", "--context", context, "info", "--format", template], 30
        )
        if result.returncode != 0:
            return [
                check(
                    "docker",
                    False,
                    required=False,
                    context=context,
                    reason="daemon_unreachable",
                    returncode=result.returncode,
                )
            ]
        info = json.loads(result.stdout)
        if not isinstance(info, dict):
            raise TypeError("Invalid Docker response")
        allowed = {
            key: info.get(key)
            for key in (
                "server_version",
                "os_type",
                "architecture",
                "cpus",
                "memory_bytes",
            )
        }
        return [check("docker", True, required=False, context=context, **allowed)]
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
        return [
            check(
                "docker",
                False,
                required=False,
                context=context,
                error_type=type(exc).__name__,
            )
        ]


def abctl_checks(skip: bool) -> list[dict]:
    if skip:
        return [
            {
                "name": "abctl",
                "status": "skipped",
                "required": False,
                "reason": "requested_with_skip_docker",
            }
        ]
    abctl = executable("abctl")
    if not abctl:
        return [check("abctl", False, required=False, reason="command_not_found")]
    try:
        result = command([abctl, "version"])
        version = re.search(r"\bv?\d+\.\d+\.\d+(?:[-+][\w.-]+)?\b", result.stdout)
        ready = result.returncode == 0 and version is not None
        return [
            check(
                "abctl",
                ready,
                required=False,
                version=version.group(0) if version else None,
                returncode=result.returncode,
            )
        ]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [check("abctl", False, required=False, error_type=type(exc).__name__)]


def cli_checks(include_cloud: bool = False) -> list[dict]:
    results = []
    commands = [("dbt", ["--version"], True)]
    if include_cloud:
        commands.extend(
            (("terraform", ["version"], False), ("psql", ["--version"], False))
        )
    for name, arguments, required in commands:
        tool_path = executable(name)
        if not tool_path:
            results.append(
                check(name, False, required=required, reason="command_not_found")
            )
            continue
        try:
            result = command([tool_path, *arguments])
            version = re.search(
                r"\bv?\d+\.\d+(?:\.\d+)?(?:[-+][\w.-]+)?\b", result.stdout
            )
            ready = result.returncode == 0 and version is not None
            results.append(
                check(
                    name,
                    ready,
                    required=required,
                    version=version.group(0) if version else None,
                    returncode=result.returncode,
                )
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append(
                check(name, False, required=required, error_type=type(exc).__name__)
            )
    return results


def credential_checks(include_cloud: bool = False) -> dict:
    # Read .env without mutating process environment, interpolation, or API calls.
    # Only booleans derived from credentials are retained in the public report.
    values: dict[str, Any] = {}
    env_file_error = None
    if (ROOT / ".env").is_file():
        try:
            from dotenv import dotenv_values

            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                values.update(dotenv_values(ROOT / ".env", interpolate=False))
        except Exception as exc:  # noqa: BLE001 - report parser/import failures without secret messages.
            env_file_error = type(exc).__name__
    values.update(os.environ)
    names = ("TINKER_API_KEY", "HF_TOKEN")
    if include_cloud:
        names += (
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_USER",
            "SNOWFLAKE_PASSWORD",
            "SNOWFLAKE_PRIVATE_KEY_PATH",
            "SNOWFLAKE_WAREHOUSE",
            "SNOWFLAKE_DATABASE",
            "SNOWFLAKE_SCHEMA",
            "SNOWFLAKE_ROLE",
            "AIRBYTE_CLIENT_ID",
            "AIRBYTE_CLIENT_SECRET",
            "AIRBYTE_WORKSPACE_ID",
            "DATABRICKS_HOST",
            "DATABRICKS_HTTP_PATH",
            "DATABRICKS_CATALOG",
            "DATABRICKS_CLIENT_ID",
            "DATABRICKS_CLIENT_SECRET",
        )
    presence = {name: bool(str(values.get(name) or "").strip()) for name in names}
    tinker_ready = presence["TINKER_API_KEY"]
    report = {
        "status": "ready" if tinker_ready else "missing_credential",
        "presence": presence,
        "tinker_config_present": tinker_ready,
        "required_for_local_execution": False,
        "env_file_error_type": env_file_error,
        "verified": False,
        "note": "Tinker needs its API key for model rollouts/training; local execution needs none. "
        "Presence only; no account access, paid API, or RL optimization was attempted.",
    }
    if not include_cloud:
        return report
    snowflake_ready = (
        presence["SNOWFLAKE_ACCOUNT"]
        and presence["SNOWFLAKE_USER"]
        and (presence["SNOWFLAKE_PASSWORD"] or presence["SNOWFLAKE_PRIVATE_KEY_PATH"])
        and presence["SNOWFLAKE_WAREHOUSE"]
    )
    airbyte_ready = (
        presence["AIRBYTE_CLIENT_ID"]
        and presence["AIRBYTE_CLIENT_SECRET"]
        and presence["AIRBYTE_WORKSPACE_ID"]
    )
    databricks_ready = all(
        presence[name]
        for name in (
            "DATABRICKS_HOST",
            "DATABRICKS_HTTP_PATH",
            "DATABRICKS_CATALOG",
            "DATABRICKS_CLIENT_ID",
            "DATABRICKS_CLIENT_SECRET",
        )
    )
    report.update(
        {
            "snowflake_config_present": snowflake_ready,
            "databricks_config_present": databricks_ready,
            "airbyte_config_present": airbyte_ready,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print one JSON report.")
    parser.add_argument(
        "--check-cloud",
        action="store_true",
        help="Also inspect optional cloud packages, Docker/Airbyte tools and credential presence.",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="With --check-cloud, leave Docker and abctl unverified. Docker is skipped by default.",
    )
    parser.add_argument(
        "--docker-context",
        default=os.environ.get("ELT_BENCH_RL_DOCKER_CONTEXT", "colima-elt-bench-rl"),
    )
    args = parser.parse_args()
    checks = (
        python_checks(args.check_cloud)
        + upstream_checks()
        + data_checks()
        + cli_checks(args.check_cloud)
    )
    if args.check_cloud:
        checks += docker_checks(args.docker_context, args.skip_docker) + abctl_checks(
            args.skip_docker
        )
    local_ready = all(item["status"] == "ready" for item in checks if item["required"])
    credentials = credential_checks(args.check_cloud)
    status = "local_not_ready" if not local_ready else credentials["status"]
    report = {
        "schema_version": 1,
        "status": status,
        "local_ready": local_ready,
        "scope": "local_duckdb_with_optional_cloud"
        if args.check_cloud
        else "local_duckdb_prerequisites",
        "default_destination": "duckdb",
        "default_loader": "local_csv_http",
        "checks": checks,
        "credentials": credentials,
        "integration_test_run": False,
        "official_rollout_run": False,
        "rl_optimization_run": False,
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(
            f"Local prerequisites: {'READY' if local_ready else 'NOT READY'} ({report['scope']})"
        )
        for item in checks:
            detail = (
                item.get("version")
                or item.get("reason")
                or item.get("error_type")
                or ""
            )
            if (
                item["name"] in UPSTREAM_NAMES
                or item["name"] == "tinker-cookbook_installed_origin"
            ):
                detail = f"expected={item.get('expected_sha')} actual={item.get('actual_sha')}"
            elif item["name"] == "docker" and item["status"] == "ready":
                detail = f"context={item['context']} server={item['server_version']}"
            print(
                f"  [{item['status']}] {item['name']}{': ' + detail if detail else ''}"
            )
        print(f"Credentials: {credentials['status']} (presence only; no API calls)")
        print("Local execution needs no credentials; Tinker calls need TINKER_API_KEY.")
        for name, present in credentials["presence"].items():
            print(f"  {name}: {str(present).lower()}")
        if credentials["env_file_error_type"]:
            print(f"  .env parser: {credentials['env_file_error_type']}")
        print(
            "This checks prerequisites only; integration, official rollout, and RL remain unverified."
        )
    return 0 if local_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
