"""Public benchmark specifications, separate from trusted reference data."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .upstreams import resolve_upstream


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", value):
        raise ValueError("Expected a lowercase SQL identifier")
    return value


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    description: str = ""
    dtype: str = "VARCHAR"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    locator: str
    columns: tuple[ColumnSpec, ...]
    keys: tuple[str, ...] = ("id",)
    expected_path: Path | None = field(default=None, repr=False)
    official_locator: str | None = None

    def public(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "columns": [
                {"name": c.name, "description": c.description, "type": c.dtype}
                for c in self.columns
            ],
            "keys": self.keys,
            "endpoint": self.official_locator if self.kind == "csv" else self.locator,
        }


@dataclass(frozen=True)
class ModelSpec:
    name: str
    columns: tuple[ColumnSpec, ...]
    keys: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    sources: tuple[SourceSpec, ...]
    models: tuple[ModelSpec, ...]
    benchmark_revision: str
    destination: str = "databricks"

    @property
    def public_prompt(self) -> str:
        spec = {
            "task_id": self.task_id,
            "sources": [s.public() for s in self.sources],
            "models": [
                {
                    "name": m.name,
                    "description": m.description,
                    "keys": m.keys,
                    "columns": [
                        {"name": c.name, "description": c.description}
                        for c in m.columns
                    ],
                }
                for m in self.models
            ],
        }
        return (
            "Build a reproducible ELT pipeline. Start with an empty warehouse. "
            "Explicitly configure and load every source, write dbt SQL models, "
            "run them, repair errors, and call submit_pipeline. "
            "Use raw source tables through {{ source('raw', 'name') }} and models "
            "through {{ ref('name') }}. Do not invent source data.\n"
            + json.dumps(spec, indent=2)
        )

    def source(self, name: str) -> SourceSpec:
        name = identifier(name)
        for source in self.sources:
            if source.name == name:
                return source
        raise ValueError("Unknown task source")


def load_task(
    project_root: Path | str,
    task_id: str = "trains",
    *,
    destination: str = "databricks",
    source_api: str = "http://127.0.0.1:5005",
) -> TaskSpec:
    """Read the pinned official format; no ground-truth file is read here.

    CSV locators use verified downloaded bytes as an offline mirror. HTTP
    sources remain real HTTP requests. Credentials in vendor config are never
    copied into the public spec. Type hints are optional and extensible.
    """
    root = Path(project_root).resolve()
    identifier(task_id)
    if destination not in {"databricks", "snowflake", "redshift"}:
        raise ValueError("Unsupported benchmark specification destination")
    vendor = resolve_upstream(root, "elt-bench")
    config = yaml.safe_load(
        (vendor / "elt-bench" / destination / task_id / "config.yaml").read_text()
    )
    models = yaml.safe_load(
        (vendor / "elt-bench" / destination / task_id / "data_model.yaml").read_text()
    )
    keys = json.loads((vendor / "evaluation/sort_key.json").read_text()).get(
        task_id, {}
    )
    if any(not keys.get(m["name"]) for m in models["models"]):
        raise ValueError("Every target model needs official primary-key metadata")
    integers = (
        {"id", "train_id", "position", "wheels", "load_num"}
        if task_id == "trains"
        else set()
    )

    def columns(table: str) -> tuple[ColumnSpec, ...]:
        identifier(table)
        with (vendor / "elt-bench/schemas" / task_id / f"{table}.csv").open() as stream:
            return tuple(
                ColumnSpec(
                    identifier(r["column_name"].lower()),
                    r["column_description"],
                    "BIGINT" if r["column_name"] in integers else "VARCHAR",
                )
                for r in csv.DictReader(stream)
            )

    sources = []
    for item in config.get("flat_files", []):
        table = identifier(item["table"])
        if item["format"].lower() != "csv":
            raise NotImplementedError("Only CSV flat-file sources are supported")
        path = root / "data" / task_id / f"{table}.csv"
        sources.append(
            SourceSpec(
                table,
                "csv",
                str(path),
                columns(table),
                expected_path=path,
                official_locator=item["path"],
            )
        )
    for table in config.get("custom_api", {}).get("config", {}).get("tables", []):
        table = identifier(table)
        path = root / "data" / task_id / f"{table}.csv"
        sources.append(
            SourceSpec(
                table,
                "http",
                f"{source_api.rstrip('/')}/{task_id}/{table}",
                columns(table),
                expected_path=path,
            )
        )
    if not sources:
        raise ValueError("Task has no supported sources")
    return TaskSpec(
        task_id,
        tuple(sources),
        tuple(
            ModelSpec(
                identifier(m["name"]),
                tuple(
                    ColumnSpec(identifier(c["name"]), c.get("description", ""))
                    for c in m["columns"]
                ),
                tuple(keys.get(m["name"], [])),
                m.get("description", ""),
            )
            for m in models["models"]
        ),
        json.loads((root / "configs/upstream-revisions.json").read_text())["elt-bench"][
            "revision"
        ],
        destination,
    )
