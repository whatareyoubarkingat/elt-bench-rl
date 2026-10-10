"""Stateful ELT episodes, terminal execution reward and clean replay."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from .dbt_runner import run_models
from .errors import InfrastructureError, PolicyError
from .grading import compare_frames, official_comparison, official_stage1_loading
from .ingestion import AIRBYTE_METADATA_COLUMNS, coerce_source
from .pipeline import PipelineBundle
from .sql_policy import render_references, validate_sql
from .task import TaskSpec, identifier


@dataclass(frozen=True)
class GradeResult:
    reward: float = 0
    loading: float = 0
    target: float = 0
    replay: float = 0
    official_stage1: bool = False
    official_stage2: bool = False
    details: dict = field(default_factory=dict)

    def metrics(self) -> dict[str, float]:
        return {
            "reward": self.reward,
            "loading": self.loading,
            "target": self.target,
            "replay": self.replay,
            "official_stage1": float(self.official_stage1),
            "official_stage2": float(self.official_stage2),
        }


def grade_state(
    task, adapter, receipts, reference_root: Path
) -> tuple[float, float, bool, bool, dict]:
    loading, targets, details = [], [], {"sources": {}, "targets": {}}
    fresh_sources, counts, official_targets = {}, {}, []
    for source in task.sources:
        try:
            expected = coerce_source(pd.read_csv(source.expected_path), source)
        except (OSError, ValueError) as error:
            raise InfrastructureError(
                "Verified source reference unavailable or corrupt"
            ) from error
        counts[source.name] = len(expected)
        try:
            actual = adapter.query(
                f'SELECT * FROM raw."{source.name}" LIMIT {len(expected) + 1}'
            )
            fresh_sources[source.name] = actual
            actual = actual.drop(
                columns=[
                    column
                    for column in actual.columns
                    if column.lower() in AIRBYTE_METADATA_COLUMNS
                    and column.lower() not in set(expected.columns)
                ]
            )
            if source.name not in receipts:
                raise ValueError("No trusted loading receipt")
            result = compare_frames(actual, expected, source.keys)
            loading.append(float(result.strict))
            details["sources"][source.name] = result.details
        except InfrastructureError:
            raise
        except (ValueError, OSError, RuntimeError) as error:
            # Missing trusted reference is infrastructure, never an agent mistake.
            if isinstance(error, OSError):
                raise InfrastructureError(
                    "Verified source cache unavailable"
                ) from error
            loading.append(0.0)
            details["sources"][source.name] = {"match": False}
        except duckdb.Error as error:
            # Engine missing-table/column exceptions are ordinary candidate failures.
            loading.append(0.0)
            details["sources"][source.name] = {
                "match": False,
                "error_type": type(error).__name__,
            }
    loaded = all(loading)
    strict_targets = []
    for model in task.models:
        path = reference_root / task.task_id / f"{model.name}.csv"
        if not path.is_file():
            raise InfrastructureError(
                "Trusted target reference unavailable; fetch pinned assets"
            )
        try:
            expected = pd.read_csv(path)
        except (OSError, ValueError) as error:
            raise InfrastructureError(
                "Trusted target reference unavailable or corrupt"
            ) from error
        try:
            limit = max(1000, len(expected) * 10)
            actual = adapter.query(
                f'SELECT * FROM transform."{model.name}" LIMIT {limit + 1}'
            )
            if len(actual) > limit:
                targets.append(0.0)
                strict_targets.append(False)
                official_targets.append(False)
                details["targets"][model.name] = {
                    "match": False,
                    "output_row_limit": limit,
                }
                continue
            result = compare_frames(actual, expected, model.keys)
            targets.append(result.score if loaded else 0.0)
            strict_targets.append(result.strict)
            official_targets.append(official_comparison(actual, expected, model.keys))
            details["targets"][model.name] = result.details
        except InfrastructureError:
            raise
        except (duckdb.Error, ValueError) as error:
            targets.append(0.0)
            strict_targets.append(False)
            official_targets.append(False)
            details["targets"][model.name] = {
                "match": False,
                "error_type": type(error).__name__,
            }
    details["official_stage1"] = official_stage1_loading(fresh_sources, counts)
    details["official_stage2"] = details["official_stage1"] and all(official_targets)
    return (
        sum(loading) / len(loading),
        sum(targets) / len(targets),
        loaded,
        all(strict_targets),
        details,
    )


class Episode:
    def __init__(
        self,
        task: TaskSpec,
        destination_factory: Callable,
        ingestor,
        reference_root: Path | str,
        work_root: Path | str,
        *,
        timeout: float = 720,
        dbt_timeout: float = 60,
        dialect: str = "duckdb",
        query_rows: int = 20,
    ):
        self.task, self.destination_factory, self.ingestor = (
            task,
            destination_factory,
            ingestor,
        )
        self.reference_root = Path(reference_root)
        self.work_dir = Path(work_root) / ("episode_" + uuid.uuid4().hex)
        self.work_dir.mkdir(parents=True, mode=0o700)
        self.adapter = destination_factory(self.work_dir / "warehouse.duckdb")
        try:
            self.adapter.create()
        except InfrastructureError:
            self._cleanup_namespace(self.adapter, self.work_dir)
            raise
        self.configured, self.models, self.receipts = {}, {}, {}
        self.submitted = False
        self.result = None
        self.bundle = None
        self.started = time.monotonic()
        self.timeout, self.dbt_timeout = timeout, dbt_timeout
        self.dialect, self.query_rows = dialect, query_rows
        self.run_attempted = False
        self.closed = False
        self.trace = []

    @property
    def public_prompt(self) -> str:
        return self.task.public_prompt + (
            "\nQueries can read raw.<source> and transform.<model>. "
            "No shell or arbitrary file access is available. Loads replace tables. "
            "Submit freezes the logical pipeline and ends this attempt."
        )

    def tools_schema(self) -> list[dict]:
        def schema(name, desc, properties=None, required=None):
            return {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": properties or {},
                    "required": required or [],
                    "additionalProperties": False,
                },
            }

        string = {"type": "string"}
        return [
            schema("read_task", "Read the task specification"),
            schema(
                "inspect_source",
                "Read a source's schema and up to 5 sample records",
                {"source": string},
                ["source"],
            ),
            schema(
                "configure_load",
                "Configure a named source for replacement ingestion",
                {"source": string, "mode": {"type": "string", "enum": ["replace"]}},
                ["source"],
            ),
            schema(
                "execute_load",
                "Actually extract and load a configured source",
                {"source": string},
                ["source"],
            ),
            schema("list_tables", "List tables in this episode"),
            schema(
                "query",
                "Execute a read-only SQL query with bounded feedback",
                {"sql": string},
                ["sql"],
            ),
            schema(
                "write_model",
                "Write a dbt SELECT model; only literal ref/source Jinja allowed",
                {"name": string, "sql": string},
                ["name", "sql"],
            ),
            schema("run_dbt", "Compile and execute all submitted dbt models"),
            schema(
                "submit_pipeline",
                "Freeze, clean replay and grade this pipeline; ends the episode",
            ),
        ]

    def dispatch(self, name: str, arguments: dict) -> dict:
        if self.submitted or self.closed:
            return {"ok": False, "error": "Episode has ended", "should_stop": True}
        if time.monotonic() - self.started > self.timeout:
            self.result = GradeResult(details={"termination": "timeout"})
            self._save_grade()
            return {"ok": False, "error": "Episode timeout", "should_stop": True}
        schemas = {s["name"]: s["parameters"] for s in self.tools_schema()}
        try:
            set_deadline = getattr(self.ingestor, "set_deadline", None)
            if set_deadline:
                set_deadline(self.started + self.timeout)
            if name not in schemas or not isinstance(arguments, dict):
                raise PolicyError("Unknown tool or malformed arguments")
            schema = schemas[name]
            if set(arguments) - set(schema["properties"]) or not set(
                schema["required"]
            ) <= set(arguments):
                raise PolicyError("Unexpected or missing tool arguments")
            if any(not isinstance(value, str) for value in arguments.values()):
                raise PolicyError("All tool arguments must be strings")
            result = self._dispatch(name, arguments)
        except InfrastructureError:
            self._record(
                name, arguments, {"ok": False, "error": "infrastructure_failure"}
            )
            raise
        except (duckdb.Error, ValueError) as error:
            result = {
                "ok": False,
                "error": str(error)[:1000],
                "error_type": type(error).__name__,
            }
        self._record(name, arguments, result)
        if not self.submitted and time.monotonic() - self.started > self.timeout:
            self.result = GradeResult(details={"termination": "timeout"})
            self._save_grade()
            return {"ok": False, "error": "Episode timeout", "should_stop": True}
        return result

    def _record(self, name, args, result):
        self.trace.append({"tool": name, "arguments": args, "result": result})
        (self.work_dir / "tools.json").write_text(
            json.dumps(self.trace, indent=2, default=str) + "\n"
        )

    def _dispatch(self, name, args):
        source_names = {s.name for s in self.task.sources}
        if name == "read_task":
            return {"ok": True, "task": self.public_prompt}
        if name == "inspect_source":
            source = self.task.source(args["source"])
            frame = self.ingestor.extract(source)
            return {
                "ok": True,
                "source": source.public(),
                "sample": self._records(frame.head(5)),
            }
        if name == "configure_load":
            source = self.task.source(args["source"])
            if args.get("mode", "replace") != "replace":
                raise PolicyError("Only replacement loads are supported")
            self.configured[source.name] = "replace"
            return {"ok": True, "source": source.name, "mode": "replace"}
        if name == "execute_load":
            source = self.task.source(args["source"])
            if source.name not in self.configured:
                raise ValueError("Configure the source before loading")
            receipt = self.ingestor.load(source, self.adapter)
            self.receipts[source.name] = receipt
            return {"ok": True, "receipt": asdict(receipt)}
        if name == "list_tables":
            return {"ok": True, "tables": self.adapter.list_tables()}
        if name == "query":
            sql = validate_sql(
                args["sql"], source_names, set(self.models), dialect=self.dialect
            )
            # Apply row cap in the engine, not after materializing unbounded results.
            frame = self.adapter.query(
                f"SELECT * FROM ({sql.rstrip().rstrip(';')}) AS bounded LIMIT {self.query_rows + 1}"
            )
            return {
                "ok": True,
                "rows": self._records(frame.head(self.query_rows)),
                "columns": list(frame.columns),
                "truncated": len(frame) > self.query_rows,
            }
        if name == "write_model":
            model = identifier(args["name"])
            if model not in self.models and len(self.models) >= 16:
                raise PolicyError("Model limit reached")
            sql = args["sql"]
            validate_sql(
                render_references(sql, source_names, set(self.models) | {model}),
                source_names,
                set(self.models) | {model},
                dialect=self.dialect,
            )
            self.models[model] = sql
            return {"ok": True, "model": model}
        if name == "run_dbt":
            self.run_attempted = True
            return run_models(
                self.adapter,
                self.models,
                source_names,
                self.work_dir / "dbt",
                timeout=min(
                    self.dbt_timeout,
                    max(0.1, self.timeout - (time.monotonic() - self.started)),
                ),
                dialect=self.dialect,
            )
        if name == "submit_pipeline":
            return self.submit()
        raise PolicyError("Unknown tool")

    @staticmethod
    def _records(frame):
        return json.loads(frame.to_json(orient="records"))

    def submit(self) -> dict:
        self.submitted = True
        self.bundle = PipelineBundle(
            self.task.task_id,
            tuple(sorted(self.configured)),
            tuple(sorted(self.models.items())),
            self.task.benchmark_revision,
        )
        try:
            self.bundle.validate(self.task, dialect=self.dialect)
            if (
                set(self.receipts) != {s.name for s in self.task.sources}
                or not self.run_attempted
            ):
                raise PolicyError(
                    "Submission requires actual loading and a dbt execution attempt"
                )
        except (ValueError, PolicyError) as error:
            self.result = GradeResult(details={"invalid_submission": str(error)})
            self._save_grade()
            return {"ok": False, "error": str(error), "should_stop": True, "reward": 0}
        self.bundle.save(self.work_dir / "pipeline.json")
        loading, target, _stage1, _stage2, details = grade_state(
            self.task, self.adapter, self.receipts, self.reference_root
        )
        replay = self.replay(self.bundle)
        details["replay"] = replay
        details["current_target"] = target
        # Stale transform tables cannot reward a subsequently broken bundle.
        target = min(target, replay["target"])
        self.result = GradeResult(
            0.2 * loading + 0.5 * target + 0.3 * float(replay["strict"]),
            loading,
            target,
            float(replay["strict"]),
            details["official_stage1"] and replay["official_stage1"],
            details["official_stage2"] and replay["official_stage2"],
            details,
        )
        self._save_grade()
        # Only aggregate metrics reach model feedback; trusted diagnostics remain offline.
        return {
            "ok": True,
            "should_stop": True,
            **self.result.metrics(),
            "bundle_sha256": self.bundle.sha256,
        }

    def replay(self, bundle: PipelineBundle) -> dict:
        bundle.validate(self.task, dialect=self.dialect)
        path = self.work_dir / ("replay_" + uuid.uuid4().hex)
        adapter = self.destination_factory(path / "warehouse.duckdb")
        receipts = {}
        replay_deadline = time.monotonic() + self.timeout
        try:
            adapter.create()
            set_deadline = getattr(self.ingestor, "set_deadline", None)
            if set_deadline:
                set_deadline(replay_deadline)
            for source_name in bundle.loads:
                source = self.task.source(source_name)
                receipts[source_name] = self.ingestor.load(source, adapter)
            run = run_models(
                adapter,
                dict(bundle.models),
                {s.name for s in self.task.sources},
                path / "dbt",
                timeout=min(
                    self.dbt_timeout, max(0.1, replay_deadline - time.monotonic())
                ),
                dialect=self.dialect,
            )
            loading, target, s1, s2, details = grade_state(
                self.task, adapter, receipts, self.reference_root
            )
            result = {
                "strict": bool(run["ok"] and s1 and s2),
                "loading": loading,
                "target": target,
                "official_stage1": details["official_stage1"],
                "official_stage2": details["official_stage2"],
                "details": details,
            }
            (path / "grade.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        finally:
            self._cleanup_namespace(adapter, path)

    def _cleanup_namespace(self, adapter, path: Path) -> None:
        """Stop writers before dropping schemas, retaining private recovery IDs."""
        evidence_path = path / "cleanup_pending.json"
        try:
            cleanup_ingestor = getattr(self.ingestor, "cleanup", None)
            if cleanup_ingestor:
                cleanup_ingestor(adapter.namespace)
            adapter.cleanup()
        except InfrastructureError:
            namespace = adapter.namespace
            ownership = getattr(self.ingestor, "cleanup_ownership", None)
            evidence = {
                "namespace": namespace,
                "destination_adapter": type(adapter).__name__,
                "warehouse_file": str(adapter.path)
                if hasattr(adapter, "path")
                else None,
                "catalog": getattr(getattr(adapter, "config", None), "catalog", None),
                "owned_schemas": sorted(getattr(adapter, "_owned", ())),
                "airbyte_actors": getattr(self.ingestor, "_resources", {}).get(
                    namespace, []
                ),
                "airbyte_jobs": sorted(
                    getattr(self.ingestor, "_jobs", {}).get(namespace, ())
                ),
                "pending_sync_connections": list(
                    getattr(self.ingestor, "_pending_sync", {}).get(namespace, {})
                ),
            }
            if ownership:
                # The loader's documented recovery view includes names for
                # ambiguous actor creation, without connection configurations.
                evidence["airbyte_recovery"] = ownership(namespace)
            # Explicit fields only: never serialize provider configurations or secrets.
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(
                evidence_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w") as handle:
                json.dump(evidence, handle, indent=2)
                handle.write("\n")
            raise InfrastructureError(
                f"Cleanup incomplete; private recovery IDs: {evidence_path}"
            ) from None
        evidence_path.unlink(missing_ok=True)

    def _save_grade(self):
        (self.work_dir / "grade.json").write_text(
            json.dumps(asdict(self.result), indent=2) + "\n"
        )

    def close(self) -> None:
        if not self.closed:
            if self.result is None:
                self.result = GradeResult(
                    details={"termination": "no_valid_submission"}
                )
                self._save_grade()
            self._cleanup_namespace(self.adapter, self.work_dir)
            self.closed = True
