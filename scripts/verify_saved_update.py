"""Verify recorded initial/updated checkpoints without repeating optimization."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from dataclasses import fields
from pathlib import Path

from dotenv import load_dotenv

from elt_bench_rl.budget import TokenBudget
from elt_bench_rl.training import TrainingConfig, verify_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--max-cost-usd", type=float, default=10)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=True)
    if os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"):
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)
    recorded = json.loads((args.training_run / "training_config.json").read_text())
    allowed = {field.name for field in fields(TrainingConfig)}
    values = {key: value for key, value in recorded.items() if key in allowed}
    values.update(
        episode_config={},
        log_path=str(args.output.resolve()),
        tokenizer_path=str(root / "data/tokenizers/Qwen3.5-4B"),
    )
    recipe = TrainingConfig(**values)
    manifest = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "code_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                root / "src/elt_bench_rl/training.py",
                root / "src/elt_bench_rl/budget.py",
            )
        },
        "status": "running",
        "optimizer_repeated": False,
    }
    output = args.output.resolve()
    if output == args.training_run.resolve() or output.is_relative_to(
        args.training_run.resolve()
    ):
        raise ValueError("Verification output must be separate from the original run")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path = output / "manifest.json"

    def write_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest_path.chmod(0o600)

    write_manifest()
    try:
        budget = (
            TokenBudget(
                args.budget_ledger,
                limit_usd=args.max_cost_usd,
                model_name=recipe.model_name,
            )
            if args.allow_paid
            else None
        )
        result = asyncio.run(
            verify_checkpoint(
                recipe,
                training_run=args.training_run.resolve(),
                allow_paid=args.allow_paid,
                budget=budget,
            )
        )
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        write_manifest()
        raise
    manifest.update(status="completed", verification_status=result.get("status"))
    write_manifest()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "effective_update",
                    "state_load_verified",
                    "probe_action_tokens",
                    "probe_max_abs_logprob_delta",
                    "original_effective_action_tokens",
                    "optimizer_repeated",
                    "injected_runtime",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
