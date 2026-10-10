"""Capability and error-classification regressions; no credentials/services."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from elt_bench_rl.dbt_runner import run_models
from elt_bench_rl.errors import InfrastructureError, PolicyError
from elt_bench_rl.sql_policy import render_references, validate_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv('/private/reference.csv')",
        "SELECT * FROM read_parquet('https://example.org/private.parquet')",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM other_namespace.private_table",
        "SELECT * FROM another_catalog.raw.cars",
        "SELECT * FROM raw.cars; DROP TABLE raw.cars",
        "SELECT * INTO stolen FROM raw.cars",
        "SELECT getenv('TINKER_API_KEY')",
        "SELECT * FROM query_table('raw.cars')",
    ],
)
def test_external_access_and_mutations_are_rejected(sql):
    with pytest.raises((PolicyError, ValueError)):
        validate_sql(sql, {"cars", "train"}, {"trains"})


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT {{ env_var('TINKER_API_KEY') }}",
        "SELECT {{ run_query('DROP TABLE raw.cars') }}",
        "{% set secret = modules.os.environ %} SELECT 1",
        "SELECT * FROM {{ source('private', 'cars') }}",
    ],
)
def test_arbitrary_jinja_execution_is_rejected(sql):
    with pytest.raises(PolicyError):
        render_references(sql, {"cars"}, {"trains"})


def test_nested_recursive_cte_is_rejected():
    sql = """SELECT * FROM (
        WITH RECURSIVE recursive_rows AS (
            SELECT 1 AS id
            UNION ALL SELECT id + 1 FROM recursive_rows
        )
        SELECT SUM(id) AS total FROM recursive_rows
    ) AS nested_query"""
    with pytest.raises(PolicyError):
        validate_sql(sql, {"cars"}, {"trains"})


def test_agent_sql_error_cannot_disguise_itself_as_infrastructure(
    monkeypatch, tmp_path
):
    adapter = SimpleNamespace(
        dbt_profile=lambda: {
            "type": "duckdb",
            "path": str(tmp_path / "warehouse.duckdb"),
            "schema": "transform",
        },
        close=lambda: None,
        reopen=lambda: None,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0],
            1,
            stdout="Conversion Error: Could not convert string 'authentication' to INT32",
            stderr="",
        ),
    )
    result = run_models(
        adapter,
        {"trains": "SELECT CAST('authentication' AS INT) AS id FROM raw.cars"},
        {"cars"},
        tmp_path / "dbt",
    )
    assert result["ok"] is False
    assert not (tmp_path / "dbt/private_profile/profiles.yml").exists()


def test_cleanup_failure_keeps_private_ids_without_credentials(episode, monkeypatch):
    namespace = episode.adapter.namespace
    monkeypatch.setattr(
        episode.ingestor,
        "_resources",
        {namespace: [("source", "owned-source-id")]},
        raising=False,
    )
    monkeypatch.setattr(episode.ingestor, "_jobs", {namespace: {123}}, raising=False)
    monkeypatch.setattr(
        episode.adapter,
        "config",
        SimpleNamespace(catalog="training", client_secret="never-persist-this-secret"),
        raising=False,
    )
    cleanup = episode.adapter.cleanup

    def fail():
        raise InfrastructureError("fixture cleanup failure")

    monkeypatch.setattr(episode.adapter, "cleanup", fail)
    with pytest.raises(InfrastructureError, match="private recovery IDs"):
        episode.close()
    record = episode.work_dir / "cleanup_pending.json"
    assert record.stat().st_mode & 0o777 == 0o600
    content = record.read_text()
    assert "never-persist-this-secret" not in content
    assert json.loads(content)["airbyte_actors"] == [["source", "owned-source-id"]]
    assert not episode.closed
    monkeypatch.setattr(episode.adapter, "cleanup", cleanup)
    episode.close()
    assert episode.closed and not record.exists()


@pytest.mark.parametrize("failing_method", ["close", "reopen"])
def test_private_profile_is_removed_even_when_adapter_lifecycle_fails(
    monkeypatch, tmp_path, failing_method
):
    def fail():
        raise RuntimeError("fixture adapter lifecycle failure")

    adapter = SimpleNamespace(
        dbt_profile=lambda: {
            "type": "duckdb",
            "path": str(tmp_path / "warehouse.duckdb"),
            "schema": "transform",
        },
        close=fail if failing_method == "close" else lambda: None,
        reopen=fail if failing_method == "reopen" else lambda: None,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="", stderr=""),
    )
    with pytest.raises(RuntimeError, match="fixture adapter lifecycle failure"):
        run_models(
            adapter, {"trains": "SELECT * FROM raw.cars"}, {"cars"}, tmp_path / "dbt"
        )
    assert not (tmp_path / "dbt/private_profile/profiles.yml").exists()
