"""Reproduce local seed-repair safeguards without Docker or network access."""

import importlib.util
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def repair(monkeypatch):
    path = (
        Path(__file__).resolve().parents[1] / "scripts/repair_airbyte_manifest_seed.py"
    )
    spec = importlib.util.spec_from_file_location("manifest_seed_repair_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbid_process(*args, **kwargs):
        raise AssertionError("Seed-repair tests must never run external processes")

    monkeypatch.setattr(module.subprocess, "run", forbid_process)
    return module


def seed(repair, *, major=6, version=None, digest=None):
    return {
        "major_version": major,
        "image_version": version or repair.TAG,
        "image_sha": digest or repair.DIGEST,
        "created_at": "2026-10-10T01:58:07+00:00",
        "updated_at": "2026-10-10T01:58:07+00:00",
    }


def state(repair, extra_rows=()):
    types = {
        "major_version": "integer",
        "image_version": "character varying",
        "created_at": "timestamp with time zone",
        "updated_at": "timestamp with time zone",
        "image_sha": "character varying",
    }
    return {
        "columns": [
            {"name": name, "type": dtype, "nullable": "NO"}
            for name, dtype in types.items()
        ],
        "constraints": [{"type": "p", "definition": "PRIMARY KEY (major_version)"}],
        "rows": [seed(repair, major=4, version="4.3.0"), *deepcopy(extra_rows)],
    }


def mock_repair_io(monkeypatch, repair, before, transaction=None):
    monkeypatch.setattr(
        repair, "verify_registry", Mock(return_value={"digest": repair.DIGEST})
    )
    monkeypatch.setattr(repair, "find_database", Mock(return_value="project-db"))

    def answer(container, sql):
        assert container == "project-db"
        if sql == repair.STATE_SQL:
            return [deepcopy(before)]
        assert sql == repair.APPLY_SQL and transaction is not None
        return deepcopy(transaction)

    query = Mock(side_effect=answer)
    monkeypatch.setattr(repair, "query", query)
    return query


@pytest.mark.parametrize("flags", [[], ["--check-only"]])
def test_default_and_explicit_check_are_read_only(repair, monkeypatch, capsys, flags):
    before = state(repair)
    query = mock_repair_io(monkeypatch, repair, before)
    monkeypatch.setattr("sys.argv", ["repair", *flags])

    assert repair.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "check_only"
    assert result["status"] == "missing"
    assert result["inserted"] is False
    query.assert_called_once_with("project-db", repair.STATE_SQL)
    assert before == state(repair)


def test_apply_preserves_an_existing_different_major6(repair, monkeypatch):
    existing = seed(repair, version="6.99.0", digest="sha256:other-public-image")
    before = state(repair, [existing])
    query = mock_repair_io(monkeypatch, repair, before)

    result = repair.repair(apply=True)

    assert result["status"] == "skipped"
    assert result["reason"] == "existing_mapping_preserved"
    assert result["mapping"] == existing
    assert result["inserted"] is False
    query.assert_called_once_with("project-db", repair.STATE_SQL)


def test_existing_correct_mapping_is_idempotent_without_apply_sql(repair, monkeypatch):
    existing = seed(repair)
    query = mock_repair_io(monkeypatch, repair, state(repair, [existing]))

    for _ in range(2):
        result = repair.repair(apply=True)
        assert result["status"] == "ready"
        assert result["inserted"] is False
        assert result["mapping"] == existing
    assert query.call_count == 2
    assert all(call.args[1] == repair.STATE_SQL for call in query.call_args_list)


def test_apply_inserts_only_the_missing_pinned_mapping(repair, monkeypatch):
    before = state(repair)
    after = state(repair, [seed(repair)])
    query = mock_repair_io(
        monkeypatch, repair, before, [before, {"inserted": True}, after]
    )

    result = repair.repair(apply=True)

    assert result["status"] == "configured"
    assert result["inserted"] is True
    assert result["mapping"] == seed(repair)
    assert result["unrelated_rows_preserved"] is True
    assert [call.args[1] for call in query.call_args_list] == [
        repair.STATE_SQL,
        repair.APPLY_SQL,
    ]
    assert "ON CONFLICT (major_version) DO NOTHING" in repair.APPLY_SQL
    assert "LOCK TABLE public.declarative_manifest_image_version" in repair.APPLY_SQL
    assert repair.DIGEST in repair.APPLY_SQL
    assert "DELETE" not in repair.APPLY_SQL and "UPDATE " not in repair.APPLY_SQL


@pytest.mark.parametrize("change", ["version", "timestamp", "added", "removed"])
def test_changed_unrelated_rows_are_rejected(repair, monkeypatch, change):
    before = state(repair)
    after = state(repair, [seed(repair)])
    if change == "version":
        after["rows"][0]["image_version"] = "4.99.0"
    elif change == "timestamp":
        after["rows"][0]["updated_at"] = "2026-10-11T01:58:07+00:00"
    elif change == "added":
        after["rows"].insert(0, seed(repair, major=3))
    else:
        after["rows"].pop(0)
    mock_repair_io(monkeypatch, repair, before, [before, {"inserted": True}, after])

    with pytest.raises(repair.RepairError) as caught:
        repair.repair(apply=True)
    assert caught.value.stage == "verification"
    assert caught.value.reason == "unrelated_rows_changed"


def test_concurrent_existing_mapping_is_preserved_by_conflict_handling(
    repair, monkeypatch
):
    before = state(repair)
    concurrent = state(repair, [seed(repair, version="6.99.0")])
    mock_repair_io(
        monkeypatch, repair, before, [concurrent, {"inserted": False}, concurrent]
    )

    result = repair.repair(apply=True)

    assert result["status"] == "skipped"
    assert result["inserted"] is False
    assert result["mapping"]["image_version"] == "6.99.0"


def registry_metadata(repair):
    return {
        "name": repair.TAG,
        "digest": repair.DIGEST,
        "images": [
            {"os": "linux", "status": "active", "architecture": architecture}
            for architecture in ("amd64", "arm64")
        ],
    }


def test_registry_requires_the_official_https_tag_metadata(repair, monkeypatch):
    run = Mock(return_value=json.dumps(registry_metadata(repair)))
    monkeypatch.setattr(repair, "run", run)

    result = repair.verify_registry()

    command, stage = run.call_args.args
    assert stage == "registry"
    assert command[0] == "curl" and command[-1] == repair.TAG_URL
    assert "--fail" in command and "--proto" in command and "=https" in command
    assert result["digest"] == repair.DIGEST
    assert result["architectures"] == ["amd64", "arm64"]


@pytest.mark.parametrize("change", ["digest", "tag", "missing_arm64", "inactive_arm64"])
def test_changed_registry_identity_is_rejected_before_database_access(
    repair, monkeypatch, change
):
    metadata = registry_metadata(repair)
    if change == "digest":
        metadata["digest"] = "sha256:changed"
    elif change == "tag":
        metadata["name"] = "6.33.5"
    elif change == "missing_arm64":
        metadata["images"].pop()
    else:
        metadata["images"][-1]["status"] = "inactive"
    monkeypatch.setattr(repair, "run", Mock(return_value=json.dumps(metadata)))
    database = Mock(
        side_effect=AssertionError("Registry rejection must precede DB access")
    )
    monkeypatch.setattr(repair, "find_database", database)

    with pytest.raises(repair.RepairError) as caught:
        repair.repair(apply=True)
    assert caught.value.stage == "registry"
    assert caught.value.reason == "official_tag_identity_does_not_match_pin"
    database.assert_not_called()


@pytest.mark.parametrize(
    "cluster,running", [("other-project", "true"), ("airbyte-abctl", "false")]
)
def test_wrong_or_stopped_kind_node_is_rejected(repair, monkeypatch, cluster, running):
    run = Mock(side_effect=[json.dumps({"io.x-k8s.kind.cluster": cluster}), running])
    monkeypatch.setattr(repair, "run", run)

    with pytest.raises(repair.RepairError) as caught:
        repair.find_database()
    assert caught.value.stage == "node"
    assert caught.value.reason == "expected_project_kind_node_not_running"
    assert run.call_count == 2


@pytest.mark.parametrize("change", ["namespace", "pod", "ambiguous"])
def test_wrong_or_ambiguous_postgres_pod_is_rejected(repair, monkeypatch, change):
    container = {
        "id": "db",
        "metadata": {"name": "airbyte-db-container"},
        "labels": {
            "io.kubernetes.pod.namespace": "airbyte-abctl",
            "io.kubernetes.pod.name": "airbyte-db-0",
        },
    }
    containers = [container]
    if change == "namespace":
        container["labels"]["io.kubernetes.pod.namespace"] = "other-project"
    elif change == "pod":
        container["labels"]["io.kubernetes.pod.name"] = "other-db-0"
    else:
        containers.append(deepcopy(container))
    monkeypatch.setattr(
        repair,
        "run",
        Mock(
            side_effect=[
                json.dumps({"io.x-k8s.kind.cluster": "airbyte-abctl"}),
                "true",
                json.dumps({"containers": containers}),
            ]
        ),
    )

    with pytest.raises(repair.RepairError) as caught:
        repair.find_database()
    assert caught.value.stage == "database"
    assert caught.value.reason == "expected_one_project_postgres_container"


@pytest.mark.parametrize(
    "change", ["missing_column", "wrong_type", "nullable", "wrong_primary_key"]
)
def test_unexpected_schema_is_rejected_before_apply_sql(repair, monkeypatch, change):
    before = state(repair)
    if change == "missing_column":
        before["columns"].pop()
    elif change == "wrong_type":
        before["columns"][0]["type"] = "text"
    elif change == "nullable":
        before["columns"][0]["nullable"] = "YES"
    else:
        before["constraints"][0]["definition"] = "PRIMARY KEY (image_version)"
    query = mock_repair_io(monkeypatch, repair, before)

    with pytest.raises(repair.RepairError) as caught:
        repair.repair(apply=True)
    assert caught.value.stage == "schema"
    assert caught.value.reason == "unexpected_seed_table_schema_or_primary_key"
    query.assert_called_once_with("project-db", repair.STATE_SQL)


def test_process_failures_do_not_print_raw_errors_or_credentials(
    repair, monkeypatch, capsys
):
    monkeypatch.setattr(
        repair.subprocess,
        "run",
        Mock(
            return_value=subprocess.CompletedProcess(
                ["curl"], 7, "", "secret-test-token"
            )
        ),
    )
    monkeypatch.setattr("sys.argv", ["repair", "--apply"])

    assert repair.main() == 1
    output = capsys.readouterr()
    assert "secret-test-token" not in output.out + output.err
    assert json.loads(output.out) == {
        "status": "error",
        "stage": "registry",
        "reason": "command_failed",
        "returncode": 7,
    }
