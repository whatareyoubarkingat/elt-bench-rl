#!/usr/bin/env python3
"""Repair one missing local Airbyte CDK-major seed; default is read-only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE = "airbyte-abctl-control-plane"
CONTEXT = "colima-elt-bench-rl"
TABLE = "public.declarative_manifest_image_version"
TAG = "6.33.4"
MAJOR = 6
DIGEST = "sha256:8fd974de2be41d6acad43a3d43576f9940a04c4f289164a2b1a5846d6f676b65"
TAG_URL = f"https://hub.docker.com/v2/repositories/airbyte/source-declarative-manifest/tags/{TAG}/"
PSQL = 'exec psql -q -X -A -t -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "$1"'
STATE_SQL = f"""SELECT json_build_object(
  'columns', (SELECT json_agg(json_build_object(
    'name',column_name,'type',data_type,'nullable',is_nullable)
    ORDER BY ordinal_position) FROM information_schema.columns
    WHERE table_schema='public' AND table_name='declarative_manifest_image_version'),
  'constraints', (SELECT json_agg(json_build_object('type',contype,
    'definition',pg_get_constraintdef(oid))) FROM pg_constraint
    WHERE conrelid='{TABLE}'::regclass),
  'rows', (SELECT json_agg(row_to_json(seed)) FROM (
    SELECT major_version,image_version,image_sha,created_at,updated_at
    FROM {TABLE} ORDER BY major_version) seed));"""
APPLY_SQL = f"""BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '10s';
LOCK TABLE {TABLE} IN SHARE ROW EXCLUSIVE MODE;
{STATE_SQL}
WITH inserted AS (
  INSERT INTO {TABLE}
    (major_version,image_version,created_at,updated_at,image_sha)
  VALUES ({MAJOR},'{TAG}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,'{DIGEST}')
  ON CONFLICT (major_version) DO NOTHING RETURNING major_version
) SELECT json_build_object('inserted',count(*)=1) FROM inserted;
{STATE_SQL}
COMMIT;"""


class RepairError(RuntimeError):
    def __init__(self, stage: str, reason: str, returncode: int | None = None):
        self.stage, self.reason, self.returncode = stage, reason, returncode


def run(command: list[str], stage: str, *, timeout: float = 25) -> str:
    env = os.environ.copy()
    # The operation is restricted to this project's existing Colima daemon.
    env["ELT_BENCH_RL_DOCKER_CONTEXT"] = CONTEXT
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RepairError(stage, type(error).__name__) from None
    if result.returncode:
        # Never print raw Docker/psql errors or a pod's environment.
        raise RepairError(stage, "command_failed", result.returncode)
    if len(result.stdout) > 1_000_000:
        raise RepairError(stage, "unexpected_output_size")
    return result.stdout.strip()


def docker(*arguments: str) -> list[str]:
    return ["bash", str(ROOT / "scripts/with_docker.sh"), "docker", *arguments]


def verify_registry() -> dict:
    output = run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--connect-timeout",
            "10",
            "--max-time",
            "20",
            TAG_URL,
        ],
        "registry",
    )
    metadata = json.loads(output)
    architectures = sorted(
        {
            image["architecture"]
            for image in metadata["images"]
            if image.get("os") == "linux" and image.get("status") == "active"
        }
    )
    if (
        metadata.get("name") != TAG
        or metadata.get("digest") != DIGEST
        or not {"amd64", "arm64"} <= set(architectures)
    ):
        raise RepairError("registry", "official_tag_identity_does_not_match_pin")
    return {
        "url": TAG_URL,
        "image_version": TAG,
        "digest": DIGEST,
        "architectures": architectures,
    }


def find_database() -> str:
    labels = json.loads(
        run(docker("inspect", NODE, "--format", "{{json .Config.Labels}}"), "node")
    )
    running = run(docker("inspect", NODE, "--format", "{{.State.Running}}"), "node")
    if labels.get("io.x-k8s.kind.cluster") != "airbyte-abctl" or running != "true":
        raise RepairError("node", "expected_project_kind_node_not_running")
    state = json.loads(
        run(docker("exec", NODE, "crictl", "ps", "-o", "json"), "database")
    )
    candidates = [
        container["id"]
        for container in state["containers"]
        if container.get("metadata", {}).get("name") == "airbyte-db-container"
        and container.get("labels", {}).get("io.kubernetes.pod.namespace")
        == "airbyte-abctl"
        and container.get("labels", {}).get("io.kubernetes.pod.name") == "airbyte-db-0"
    ]
    if len(candidates) != 1:
        raise RepairError("database", "expected_one_project_postgres_container")
    return candidates[0]


def query(container: str, sql: str) -> list[dict]:
    output = run(
        docker(
            "exec",
            NODE,
            "crictl",
            "exec",
            "--sync",
            container,
            "sh",
            "-c",
            PSQL,
            "sh",
            sql,
        ),
        "postgres",
    )
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def validate_schema(state: dict) -> None:
    expected = {
        "major_version": "integer",
        "image_version": "character varying",
        "image_sha": "character varying",
        "created_at": "timestamp with time zone",
        "updated_at": "timestamp with time zone",
    }
    columns = state.get("columns") or []
    if (
        {column["name"]: column["type"] for column in columns} != expected
        or any(column["nullable"] != "NO" for column in columns)
        or not any(
            constraint["type"] == "p"
            and constraint["definition"] == "PRIMARY KEY (major_version)"
            for constraint in state.get("constraints") or []
        )
    ):
        raise RepairError("schema", "unexpected_seed_table_schema_or_primary_key")


def mapping(state: dict) -> dict | None:
    rows = [row for row in state.get("rows") or [] if row["major_version"] == MAJOR]
    if len(rows) > 1:
        raise RepairError("schema", "ambiguous_major_mapping")
    return rows[0] if rows else None


def repair(*, apply: bool = False) -> dict:
    registry = verify_registry()
    container = find_database()
    before = query(container, STATE_SQL)[0]
    validate_schema(before)
    existing = mapping(before)
    inserted = False
    after = before
    if apply and existing is None:
        transaction = query(container, APPLY_SQL)
        if len(transaction) != 3:
            raise RepairError("postgres", "unexpected_transaction_output")
        before, insertion, after = transaction
        validate_schema(before)
        validate_schema(after)
        inserted = insertion["inserted"]
        untouched_before = [
            row for row in before["rows"] or [] if row["major_version"] != MAJOR
        ]
        untouched_after = [
            row for row in after["rows"] or [] if row["major_version"] != MAJOR
        ]
        if untouched_before != untouched_after:
            raise RepairError("verification", "unrelated_rows_changed")
        existing = mapping(after)
    expected_mapping = (
        existing is not None
        and existing["image_version"] == TAG
        and existing["image_sha"] == DIGEST
    )
    status = (
        "configured"
        if inserted and expected_mapping
        else "ready"
        if expected_mapping
        else "skipped"
        if existing
        else "missing"
    )
    return {
        "status": status,
        "mode": "apply" if apply else "check_only",
        "inserted": inserted,
        "major_version": MAJOR,
        "registry": registry,
        "mapping": existing,
        "seed_majors": [row["major_version"] for row in after["rows"] or []],
        "reason": "verified_expected_mapping"
        if expected_mapping
        else "existing_mapping_preserved"
        if existing
        else "missing_major6_mapping",
        "unrelated_rows_preserved": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Insert major 6 only if absent; never overwrite",
    )
    mode.add_argument(
        "--check-only", action="store_true", help="Read-only check (the default)"
    )
    arguments = parser.parse_args()
    try:
        result = repair(apply=arguments.apply)
    except RepairError as error:
        result = {
            "status": "error",
            "stage": error.stage,
            "reason": error.reason,
            "returncode": error.returncode,
        }
    except (KeyError, TypeError, ValueError) as error:
        result = {
            "status": "error",
            "stage": "metadata",
            "reason": type(error).__name__,
        }
    print(json.dumps(result, sort_keys=True))
    return int(result["status"] == "error")


if __name__ == "__main__":
    raise SystemExit(main())
