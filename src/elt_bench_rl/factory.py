"""Lazy, pickle-friendly episode construction; no cloud calls on import."""

from __future__ import annotations

import functools
from pathlib import Path

from .destination import DuckDBAdapter
from .environment import Episode
from .ingestion import LocalIngestor
from .task import load_task


def build_episode(
    *,
    project_root: str,
    task_id: str = "trains",
    destination: str = "duckdb",
    run_root: str | None = None,
    source_api: str = "http://127.0.0.1:5005",
    reference_root: str | None = None,
    **kwargs,
) -> Episode:
    root = Path(project_root).resolve()
    task = load_task(root, task_id, destination="databricks", source_api=source_api)
    reference = (
        Path(reference_root) if reference_root else root / "ground_truth/gt_databricks"
    )
    if destination == "duckdb":
        adapter = functools.partial(
            DuckDBAdapter, query_timeout=kwargs.pop("query_timeout", 15)
        )
        ingestor = LocalIngestor()
        dialect = "duckdb"
    elif destination == "databricks":
        from .airbyte import AirbyteIngestor
        from .databricks import DatabricksAdapter, DatabricksConfig

        adapter = functools.partial(
            DatabricksAdapter, config=DatabricksConfig.from_env()
        )
        ingestor = AirbyteIngestor.from_env(project_root=root)
        dialect = "databricks"
    else:
        raise ValueError("Destination must be duckdb or databricks")
    return Episode(
        task,
        adapter,
        ingestor,
        reference,
        run_root or root / "runs",
        dialect=dialect,
        **kwargs,
    )
