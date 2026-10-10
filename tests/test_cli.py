"""Default local CLI integration, using real HTTP, DuckDB and dbt execution."""

from __future__ import annotations

import json
import shutil
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests

from elt_bench_rl import cli
from elt_bench_rl.pipeline import PipelineBundle
from elt_bench_rl.task import load_task
from elt_bench_rl.training import PaidAccessRequired

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_project(tmp_path):
    """Mirror checked-in fixtures only; no ignored downloads, Docker or keys."""
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    shutil.copyfile(
        PROJECT / "configs/upstream-revisions.json",
        root / "configs/upstream-revisions.json",
    )
    shutil.copytree(PROJECT / "tests/fixtures/sources", root / "data/trains")
    shutil.copytree(
        PROJECT / "tests/fixtures/references",
        root / "ground_truth/gt_databricks",
    )
    return root


@pytest.fixture
def local_servers(monkeypatch):
    """Observe the actual builtin source lifecycle rather than mocking it."""
    original = cli.serve_sources
    servers = []

    @contextmanager
    def tracked_sources(*args, **kwargs):
        record = {"closed": False}
        servers.append(record)
        try:
            with original(*args, **kwargs) as api:
                record["api"] = api
                yield api
        finally:
            record["closed"] = True

    monkeypatch.setattr(cli, "serve_sources", tracked_sources)
    return servers


def assert_server_closed(record):
    assert record["closed"]
    address = urlparse(record["api"])
    assert address.hostname == "127.0.0.1"
    with (
        pytest.raises(OSError),
        socket.create_connection((address.hostname, address.port), timeout=0.2),
    ):
        pytest.fail("CLI left its builtin HTTP source listening")


def test_default_replay_loads_builtin_http_and_closes_it(
    local_project, local_servers, correct_sql, monkeypatch, capsys, tmp_path
):
    task = load_task(local_project)
    bundle = PipelineBundle(
        task.task_id,
        tuple(source.name for source in task.sources),
        (("trains", correct_sql),),
        task.benchmark_revision,
    )
    bundle_path = tmp_path / "pipeline.json"
    bundle.save(bundle_path)
    output = tmp_path / "replay"
    http_requests = []
    original_request = requests.Session.request

    def record_request(session, method, url, **kwargs):
        address = urlparse(url)
        assert address.hostname == "127.0.0.1", "Local CLI attempted an external API"
        http_requests.append((method, url))
        return original_request(session, method, url, **kwargs)

    monkeypatch.setattr(requests.Session, "request", record_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "elt-rl",
            "--project-root",
            str(local_project),
            "replay",
            "--bundle",
            str(bundle_path),
            "--output",
            str(output),
        ],
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["strict"] is True
    assert result["loading"] == result["target"] == 1
    assert result["official_stage1"] is result["official_stage2"] is True
    assert len(local_servers) == 1
    api = local_servers[0]["api"]
    assert http_requests == [("GET", f"{api}/trains/cars")]
    grade_paths = list(output.glob("*/replay_*/grade.json"))
    assert len(grade_paths) == 1
    assert json.loads(grade_paths[0].read_text()) == result
    assert not list(output.rglob("*.duckdb"))
    assert not list(output.rglob("*.duckdb.wal"))
    assert_server_closed(local_servers[0])


@pytest.mark.parametrize("command", ["rollout", "train"])
def test_paid_gate_also_closes_builtin_source(
    command, local_project, local_servers, monkeypatch, tmp_path
):
    def unexpected_client(*args, **kwargs):
        pytest.fail("A paid client was created before explicit authorization")

    monkeypatch.setattr("elt_bench_rl.training.tinker.ServiceClient", unexpected_client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "elt-rl",
            "--project-root",
            str(local_project),
            command,
            "--output",
            str(tmp_path / command),
        ],
    )
    with pytest.raises(PaidAccessRequired):
        cli.main()
    assert len(local_servers) == 1
    assert_server_closed(local_servers[0])


def test_explicit_source_api_is_preserved_without_starting_server(monkeypatch):
    def unexpected_server(*args, **kwargs):
        pytest.fail("Explicit source API must not launch a builtin server")

    monkeypatch.setattr(cli, "serve_sources", unexpected_server)
    url = "http://127.0.0.1:8456"
    args = cli.parser().parse_args(["train", "--source-api", url])
    with cli.source_service(args, PROJECT) as api:
        assert api == url


def test_offline_preflight_does_not_start_source_service(
    local_project, local_servers, monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("TINKER_API_KEY", "fixture-private-key-canary")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "elt-rl",
            "--project-root",
            str(local_project),
            "preflight",
            "--output",
            str(tmp_path / "preflight"),
        ],
    )
    cli.main()
    text = capsys.readouterr().out
    result = json.loads(text)
    assert result["database"] == "duckdb" and result["cloud_calls"] is False
    assert result["credential_presence"] == {"TINKER_API_KEY": True}
    assert "fixture-private-key-canary" not in text
    assert local_servers == []
