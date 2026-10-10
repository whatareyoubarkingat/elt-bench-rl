"""Credential-free logical submissions with content-addressed immutable snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .dependencies import validate_dependencies
from .errors import PolicyError
from .sql_policy import render_references, validate_sql
from .task import TaskSpec, identifier


@dataclass(frozen=True)
class PipelineBundle:
    task_id: str
    loads: tuple[str, ...]
    models: tuple[tuple[str, str], ...]
    benchmark_revision: str
    version: int = 1

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "task_id": self.task_id,
            "benchmark_revision": self.benchmark_revision,
            "loads": [{"source": name, "mode": "replace"} for name in self.loads],
            "dbt_project": {"name": "elt_pipeline", "materialized": "table"},
            "models": dict(self.models),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode()
        ).hexdigest()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        path.chmod(0o444)
        project = path.parent / "submission/dbt"
        model_dir = project / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        project_file = project / "dbt_project.yml"
        project_file.write_text(
            yaml.safe_dump(
                {
                    "name": "elt_pipeline",
                    "version": "1.0",
                    "config-version": 2,
                    "profile": "elt_pipeline",
                    "model-paths": ["models"],
                    "models": {"elt_pipeline": {"+materialized": "table"}},
                }
            )
        )
        project_file.chmod(0o444)
        for name, sql in self.models:
            model = model_dir / f"{name}.sql"
            model.write_text(sql)
            model.chmod(0o444)

    def validate(self, task: TaskSpec, *, dialect: str = "duckdb") -> None:
        if (
            self.version != 1
            or self.task_id != task.task_id
            or self.benchmark_revision != task.benchmark_revision
        ):
            raise PolicyError("Bundle does not match the task revision")
        sources = {s.name for s in task.sources}
        if len(set(self.loads)) != len(self.loads) or set(self.loads) != sources:
            raise PolicyError(
                "Bundle must explicitly configure every source exactly once"
            )
        models = dict(self.models)
        if (
            len(models) != len(self.models)
            or not {m.name for m in task.models} <= set(models)
            or len(models) > 16
        ):
            raise PolicyError("Bundle does not contain the requested models")
        for name, sql in models.items():
            identifier(name)
            validate_sql(
                render_references(sql, sources, set(models)),
                sources,
                set(models),
                dialect=dialect,
            )
        validate_dependencies(
            models, sources, {m.name for m in task.models}, dialect=dialect
        )

    @classmethod
    def read(cls, path: Path) -> PipelineBundle:
        if path.stat().st_size > 600000:
            raise PolicyError("Bundle exceeds size limit")
        obj = json.loads(path.read_text())
        if set(obj) != {
            "version",
            "task_id",
            "benchmark_revision",
            "loads",
            "dbt_project",
            "models",
        }:
            raise PolicyError("Unexpected bundle fields")
        if any(
            set(item) != {"source", "mode"} or item["mode"] != "replace"
            for item in obj["loads"]
        ):
            raise PolicyError("Only logical replace load configurations are allowed")
        if obj["dbt_project"] != {"name": "elt_pipeline", "materialized": "table"}:
            raise PolicyError("Only the fixed dbt project configuration is allowed")
        return cls(
            obj["task_id"],
            tuple(item["source"] for item in obj["loads"]),
            tuple(sorted(obj["models"].items())),
            obj["benchmark_revision"],
            obj["version"],
        )
