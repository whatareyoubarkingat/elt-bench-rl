"""Executable local smoke/replay, official rollout and one-step RL recipe."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .errors import InfrastructureError
from .factory import build_episode
from .pipeline import PipelineBundle
from .source_server import serve_sources


def _revision(root):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _scripted_episode(root, api, run_root, *, wrong=False):
    episode = build_episode(
        project_root=str(root), source_api=api, run_root=str(run_root)
    )
    try:
        assert episode.dispatch("list_tables", {})["tables"] == []
        for source in episode.task.sources:
            assert episode.dispatch("configure_load", {"source": source.name})["ok"]
            assert episode.dispatch("execute_load", {"source": source.name})["ok"]
        sql = (root / "examples/trains.sql").read_text()
        if not wrong:
            assert episode.dispatch(
                "write_model",
                {"name": "trains", "sql": sql.replace("t.direction", "t.missing_col")},
            )["ok"]
            feedback = episode.dispatch("run_dbt", {})
            assert not feedback["ok"] and "missing_col" in feedback["feedback"]
        else:
            sql = sql.replace("t.direction,", "'wrong' as direction,")
        assert episode.dispatch("write_model", {"name": "trains", "sql": sql})["ok"]
        assert episode.dispatch("run_dbt", {})["ok"]
        episode.dispatch("submit_pipeline", {})
        return {
            "scripted_agent": True,
            "model_generated": False,
            "episode_path": str(episode.work_dir),
            **episode.result.metrics(),
            "bundle_sha256": episode.bundle.sha256,
        }
    finally:
        episode.close()


def local_smoke(root, output):
    """Uses official public source bytes; this is not a model/cloud rollout."""
    with serve_sources(root / "data/trains") as api:
        good = _scripted_episode(root, api, output, wrong=False)
        bad = _scripted_episode(root, api, output, wrong=True)
    result = {
        "validation": "credential_free_local_execution",
        "passed": good["reward"] == 1 and bad["reward"] < good["reward"],
        "source_revision": _revision(root),
        "date": datetime.now(UTC).isoformat(),
        "correct": good,
        "incorrect": bad,
        "credentialed_official_rollout": "not_run",
        "real_tinker_optimization": "not_run",
    }
    (output / "local-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    commands = command.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "local-smoke", help="Offline public data, actual dbt, feedback and clean replay"
    )
    smoke.add_argument("--output", type=Path)
    for action in ("replay", "rollout", "train", "preflight"):
        sub = commands.add_parser(action)
        sub.add_argument(
            "--destination", choices=["duckdb", "databricks"], default="duckdb"
        )
        sub.add_argument("--task", default="trains")
        sub.add_argument("--source-api", default="http://127.0.0.1:5005")
        sub.add_argument("--output", type=Path)
        sub.add_argument(
            "--allow-paid",
            action="store_true",
            help="Explicitly enable Tinker / cloud warehouse calls",
        )
        if action == "replay":
            sub.add_argument("--bundle", type=Path, required=True)
        if action in {"rollout", "train"}:
            sub.add_argument(
                "--config", type=Path, help="Override recipe defaults with YAML"
            )
            sub.add_argument("--tokenizer", type=Path)
        if action == "preflight":
            sub.add_argument(
                "--online",
                action="store_true",
                help="Also test cloud permissions and connectivity",
            )
    return command


def main():
    args = parser().parse_args()
    root = args.project_root.resolve()
    load_dotenv(root / ".env", override=False)
    output = (
        args.output or root / "runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "local-smoke":
        result = local_smoke(root, output)
    else:
        if (
            args.destination == "databricks"
            and not args.allow_paid
            and (args.command != "preflight" or args.online)
        ):
            raise SystemExit(
                "Cloud warehouse execution requires --allow-paid after choosing a budget"
            )
        episode_config = {
            "project_root": str(root),
            "task_id": args.task,
            "destination": args.destination,
            "run_root": str(output),
            "source_api": args.source_api,
        }
        if args.command == "replay":
            episode = build_episode(**episode_config)
            try:
                result = episode.replay(PipelineBundle.read(args.bundle))
            finally:
                episode.close()
        elif args.command == "preflight":
            names = (
                "TINKER_API_KEY",
                "DATABRICKS_HOST",
                "DATABRICKS_HTTP_PATH",
                "DATABRICKS_CATALOG",
                "DATABRICKS_CLIENT_ID",
                "DATABRICKS_CLIENT_SECRET",
                "AIRBYTE_CLIENT_ID",
                "AIRBYTE_CLIENT_SECRET",
                "AIRBYTE_WORKSPACE_ID",
            )
            result = {
                "credential_presence": {name: bool(os.getenv(name)) for name in names},
                "cloud_calls": False,
                "database": args.destination,
            }
            if args.online:
                if args.destination == "databricks" and not args.allow_paid:
                    raise SystemExit("Online warehouse preflight requires --allow-paid")
                env = build_episode(**episode_config)
                try:
                    result["online"] = {"empty_schema": env.adapter.list_tables() == []}
                    result["cloud_calls"] = args.destination == "databricks"
                finally:
                    env.close()
        else:
            import yaml

            from .training import TrainingConfig, rollout, train

            values = yaml.safe_load(args.config.read_text()) if args.config else {}
            values = values or {}
            values.update(
                {
                    "episode_config": episode_config,
                    "log_path": str(output),
                    "tokenizer_path": str(
                        args.tokenizer or root / "data/tokenizers/Qwen3.5-4B"
                    ),
                }
            )
            if args.command == "rollout" and "group_size" not in values:
                values["group_size"] = 1
            recipe = TrainingConfig(**values)
            fn = train if args.command == "train" else rollout
            result = asyncio.run(fn(recipe, allow_paid=args.allow_paid)).as_dict()
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "source_revision": _revision(root),
                        "recipe": values,
                        "result": result,
                    },
                    indent=2,
                )
                + "\n"
            )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except InfrastructureError as error:
        raise SystemExit(
            "Infrastructure failure (excluded from RL reward): " + str(error)
        ) from error
