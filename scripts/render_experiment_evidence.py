"""Render a local evidence page from recorded artifacts, without API calls."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import UTC, datetime
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def render(run_dir: Path, output: Path) -> None:
    manifest = read_json(run_dir / "experiment-manifest.json")
    budget = read_json(run_dir / "budget.json")
    verification = read_json(run_dir / "checkpoint-verification" / "verification.json")
    phases = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_dir() or not (path / "monitor-status.json").exists():
            continue
        monitor = read_json(path / "monitor-status.json")
        rollout = read_json(path / "rollout_result.json")
        training = read_json(path / "training_result.json")
        events_path = path / "progress.jsonl"
        events = (
            [json.loads(line) for line in events_path.read_text().splitlines() if line]
            if events_path.exists()
            else []
        )
        phases.append(
            {
                "phase": path.name,
                "process_state": monitor.get("state"),
                "pid": monitor.get("child_pid"),
                "elapsed_seconds": monitor.get("elapsed_seconds"),
                "last_progress_age_seconds": monitor.get("last_progress_age_seconds"),
                "last_event": events[-1].get("event") if events else None,
                "model_calls": sum(e.get("event") == "sample_end" for e in events),
                "generated_tokens": sum(
                    len(e.get("action_tokens", []))
                    for e in events
                    if e.get("event") == "sample_end"
                ),
                "rewards": rollout.get("rewards"),
                "stop_reasons": rollout.get("stop_reasons"),
                "metrics": rollout.get("metrics"),
                "training_status": training.get("status"),
                "training_steps": [
                    {
                        key: step.get(key)
                        for key in (
                            "step",
                            "rewards",
                            "effective_action_tokens",
                            "optimizer_completed",
                            "effective_update",
                            "state_load_verified",
                            "probe_max_abs_logprob_delta",
                        )
                    }
                    for step in training.get("steps", [])
                ],
                "progress_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest()
                if events_path.exists()
                else None,
            }
        )
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_revision": manifest.get("source_revision"),
        "experiment_status": manifest.get("status"),
        "model": manifest.get("model"),
        "destination": manifest.get("destination"),
        "budget_limit_usd": budget.get("limit_usd"),
        "token_cost_estimate_usd": sum(
            item["accounted_usd"] for item in budget.get("reservations", [])
        )
        + budget.get("prior_cost_usd", 0),
        "storage_buffer_usd": budget.get("storage_buffer_usd"),
        "remaining_after_reservations_usd": budget.get("remaining_usd"),
        "phases": phases,
        "checkpoint_verification": verification,
        "evidence_note": "Rendered from local experiment records; this is not the Tinker console or an invoice.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def show(value):
        return html.escape(
            json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        )

    def card(label, value):
        return f'<div class="card"><small>{show(label)}</small><strong>{show(value)}</strong></div>'

    def label(key):
        return key.replace("_", " ").capitalize()

    cards = "".join(
        card(label(key), summary[key])
        for key in (
            "experiment_status",
            "model",
            "destination",
            "token_cost_estimate_usd",
            "remaining_after_reservations_usd",
        )
    )
    sections = []
    for phase in phases:
        rows = "".join(
            f"<tr><th>{show(label(key))}</th><td>{show(value)}</td></tr>"
            for key, value in phase.items()
            if key != "progress_sha256"
        )
        sections.append(
            f"<section><h2>{show(phase['phase'])}</h2><table>{rows}</table>"
            f"<small>Progress SHA256: {show(phase['progress_sha256'])}</small></section>"
        )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="refresh" content="15"><title>ELT-Bench experiment evidence</title>'
        "<style>body{font:15px system-ui;background:#f5f7fa;color:#17304b;margin:28px}"
        "main{max-width:1050px;margin:auto}h1{font-size:26px}h2{font-size:20px}"
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}"
        ".card,section{background:white;padding:18px;border:1px solid #dde4ed;border-radius:10px}"
        ".card small,.card strong{display:block}.card strong{margin-top:8px;overflow-wrap:anywhere}"
        "section{margin-top:18px}table{width:100%;border-collapse:collapse}"
        "th,td{text-align:left;vertical-align:top;padding:6px;border-bottom:1px solid #e8edf3;overflow-wrap:anywhere}"
        "th{width:35%;font-weight:500}small{color:#526477;overflow-wrap:anywhere}</style>"
        "<main><h1>ELT-Bench real experiment</h1>"
        f'<p>{show(summary["evidence_note"])}</p><div class="cards">{cards}</div>'
        f"<p><small>Source revision: {show(summary['source_revision'])}<br>"
        f"Generated UTC: {show(summary['generated_at_utc'])}</small></p>"
        + "".join(sections)
        + "</main></html>"
    )
    (output / "index.html").write_text(document)
    train_phase = manifest.get("phases", {}).get("train-01", {})
    proof_values = {
        "Task": "Official trains / DuckDB adaptation",
        "Model": summary["model"],
        "Strict model successes": train_phase.get("strict_model_successes", 0),
        "Group rewards": [round(value, 3) for value in train_phase.get("rewards", [])],
        "Optimizer completed": train_phase.get("optimizer_completed", False),
        "Checkpoint verification": verification.get("status", "pending"),
        "Effective action tokens": verification.get("original_effective_action_tokens"),
        "Fixed action logprob change": verification.get("probe_max_abs_logprob_delta"),
        "State checkpoint reloaded": verification.get("state_load_verified"),
        "Token estimate (USD)": round(summary["token_cost_estimate_usd"], 6),
    }
    proof_rows = "".join(
        f"<p><small>{show(key)}</small><strong>{show(value)}</strong></p>"
        for key, value in proof_values.items()
        if value is not None
    )
    (output / "proof.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Recorded ELT-Bench experiment evidence</title>"
        "<style>body{font:12px system-ui;margin:10px;color:#17304b;background:#f5f7fa}"
        "main{max-width:178px}h1{font-size:16px}p{background:white;"
        "padding:5px;border:1px solid #dde4ed;border-radius:6px;margin:5px 0}"
        "small,strong{display:block;overflow-wrap:anywhere}small{color:#526477;"
        "font-size:10px}strong{margin-top:2px}</style><main>"
        "<h1>ELT-Bench real experiment</h1>"
        "<small>Recorded local evidence. Not the Tinker console or an invoice.</small>"
        + proof_rows
        + f"<small>Run: 2026-10-12<br>Training source: {show(summary['source_revision'])}</small>"
        "</main></html>"
    )
    update_values = {
        "Verification": verification.get("status", "pending"),
        "Model": summary["model"],
        "Optimizer completed": verification.get("original_optimizer_completed"),
        "Effective action tokens": verification.get("original_effective_action_tokens"),
        "Fixed action logprob change": verification.get("probe_max_abs_logprob_delta"),
        "State checkpoint reloaded": verification.get("state_load_verified"),
        "Optimizer repeated": verification.get("optimizer_repeated"),
    }
    update_rows = "".join(
        f"<p><small>{show(key)}</small><strong>{show(value)}</strong></p>"
        for key, value in update_values.items()
        if value is not None
    )
    (output / "update-proof.html").write_text(
        (output / "proof.html").read_text().replace(proof_rows, update_rows)
    )
    print(
        json.dumps({"evidence_page": str(output / "index.html"), "phases": len(phases)})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.run_dir.resolve(), args.output.resolve())
