"""Controlled real dbt execution; the model never receives a shell capability."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import sqlglot
import yaml
from sqlglot import exp

from .errors import InfrastructureError
from .sql_policy import render_references, validate_sql


def run_models(
    adapter,
    models: dict[str, str],
    sources: set[str],
    project_dir: Path,
    *,
    timeout: float = 60,
    dialect: str = "duckdb",
) -> dict:
    if not models:
        raise ValueError("Write at least one model first")
    project_dir.mkdir(parents=True, exist_ok=True)
    model_dir = project_dir / "models"
    model_dir.mkdir(exist_ok=True)
    for stale in model_dir.glob("*.sql"):
        stale.unlink()
    for name, sql in models.items():
        rendered = render_references(sql, sources, set(models))
        validate_sql(rendered, sources, set(models), dialect=dialect)
        # Resolve every logical reference through dbt's trusted namespace.
        # Literal raw.foo must not reach a cloud server as another schema.
        tree = sqlglot.parse_one(rendered, read=dialect)
        references = {}
        for table in tree.find_all(exp.Table):
            logical_schema, table_name = table.db.lower(), table.name.lower()
            if logical_schema not in {"raw", "transform"}:
                continue  # A query-local CTE.
            marker = "elt_ref_" + uuid.uuid4().hex
            references[marker] = (
                "{{ source('raw', '" + table_name + "') }}"
                if logical_schema == "raw"
                else "{{ ref('" + table_name + "') }}"
            )
            table.set("this", exp.to_identifier(marker))
            table.set("db", None)
            table.set("catalog", None)
        model_sql = tree.sql(dialect=dialect)
        for marker, reference in references.items():
            model_sql = model_sql.replace(marker, reference)
        (model_dir / f"{name}.sql").write_text(model_sql)
    (project_dir / "dbt_project.yml").write_text(
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
    profile = adapter.dbt_profile()
    raw_schema = getattr(adapter, "raw_schema", "raw")
    source = {
        "name": "raw",
        "schema": raw_schema,
        "tables": [{"name": name} for name in sorted(sources)],
    }
    if profile.get("catalog"):
        source["database"] = profile["catalog"]
    (model_dir / "sources.yml").write_text(
        yaml.safe_dump({"version": 2, "sources": [source]})
    )
    # Profiles contain credentials only in a private directory, never the bundle.
    private = project_dir / "private_profile"
    private.mkdir(mode=0o700, exist_ok=True)
    profile_file = private / "profiles.yml"
    profile_file.write_text(
        yaml.safe_dump(
            {"elt_pipeline": {"target": "episode", "outputs": {"episode": profile}}}
        )
    )
    profile_file.chmod(0o600)
    executable = Path(sys.executable).parent / "dbt"
    if not executable.exists():
        executable = Path(shutil.which("dbt") or "missing-dbt")
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            s in k.upper()
            for s in (
                "TINKER",
                "AIRBYTE",
                "SNOWFLAKE",
                "DATABRICKS",
                "TOKEN",
                "SECRET",
                "PASSWORD",
            )
        )
    }
    env.update({"DBT_SEND_ANONYMOUS_USAGE_STATS": "false", "DO_NOT_TRACK": "1"})
    try:
        adapter.close()  # DuckDB cannot be opened concurrently by another process.
        process = subprocess.run(
            [
                str(executable),
                "--no-use-colors",
                "run",
                "--project-dir",
                str(project_dir),
                "--profiles-dir",
                str(private),
            ],
            cwd=project_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = process.stdout + process.stderr
        for key in ("client_secret", "token", "secret", "password"):
            value = profile.get(key)
            if isinstance(value, str) and value:
                output = output.replace(value, "[REDACTED]")
        (project_dir / "execution.txt").write_text(output)
        result = {
            "ok": process.returncode == 0,
            "exit_code": process.returncode,
            "feedback": output[-8000:],
        }
    except subprocess.TimeoutExpired:
        result = {"ok": False, "error": "dbt execution timed out", "timeout": True}
    except OSError as error:
        raise InfrastructureError("dbt executable unavailable") from error
    finally:
        profile_file.unlink(missing_ok=True)
        # dbt writes its own log/artifacts as well as captured CLI feedback.
        # Scrub configured secret values from all retained text diagnostics.
        secrets = [
            profile.get(key)
            for key in ("client_secret", "token", "secret", "password")
            if isinstance(profile.get(key), str) and profile.get(key)
        ]
        for directory in (project_dir / "logs", project_dir / "target"):
            if not secrets or not directory.exists():
                continue
            for artifact in directory.rglob("*"):
                if not artifact.is_file() or artifact.suffix not in {
                    ".json",
                    ".log",
                    ".sql",
                    ".txt",
                }:
                    continue
                content = artifact.read_text(errors="replace")
                for secret in secrets:
                    content = content.replace(secret, "[REDACTED]")
                artifact.write_text(content)
        reopen = getattr(adapter, "reopen", None)
        if reopen:
            reopen()
    if not result["ok"]:
        health_check = getattr(adapter, "check_health", None)
        if health_check:
            health_check()
    return result
