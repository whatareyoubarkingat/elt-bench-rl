from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from elt_bench_rl.destination import DuckDBAdapter
from elt_bench_rl.environment import Episode
from elt_bench_rl.ingestion import LocalIngestor
from elt_bench_rl.source_server import serve_sources
from elt_bench_rl.task import load_task

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def episode(tmp_path):
    """Small synthetic source/GT, unlike the official credentialed acceptance."""
    sources = PROJECT / "tests/fixtures/sources"
    with serve_sources(sources) as api:
        task = load_task(PROJECT, source_api=api)
        task = replace(
            task,
            sources=tuple(
                replace(
                    s,
                    locator=str(sources / f"{s.name}.csv")
                    if s.kind == "csv"
                    else f"{api}/trains/{s.name}",
                    expected_path=sources / f"{s.name}.csv",
                )
                for s in task.sources
            ),
        )
        env = Episode(
            task,
            DuckDBAdapter,
            LocalIngestor(),
            PROJECT / "tests/fixtures/references",
            tmp_path,
        )
        yield env
        env.close()


@pytest.fixture
def correct_sql():
    return (PROJECT / "examples/trains.sql").read_text()
