# ELT-Bench RLVR

An end-to-end ELT-Bench RLVR environment built with [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook), DuckDB, and dbt. Data processing runs locally; Tinker provides remote model sampling and training.

Each attempt starts in an isolated empty database. The model loads CSV/HTTP sources, writes SQL, uses execution feedback, and submits a pipeline for fresh replay. Reference answers remain private.

Reward: `0.2 × loading + 0.5 × target correctness + 0.3 × strict replay`.

## Quick start

Requires Git, Python 3.11, uv, curl, and internet for setup. Local execution needs no credentials, Docker, or GPU.

```bash
git clone https://github.com/whatareyoubarkingat/elt-bench-rl.git
cd elt-bench-rl
python3 scripts/fetch_upstreams.py
uv sync --locked --group dev --python 3.11
uv run --locked python scripts/fetch_trains_data.py
uv run --locked elt-rl local-smoke --output runs/local-smoke
uv run --locked pytest -q tests
```

The smoke test is a scripted pipeline with real loading, dbt execution, and clean replay. It does not call Tinker.

## Model rollout and training

Copy `configs/.env.example` to `.env`, set `TINKER_API_KEY`, and follow the [run instructions](docs/running.md) for tokenizer setup, rollout, training, and spending limits.

## Validation results

- 262 automated tests passed, including credential-free local integration.
- Real Tinker rollouts on the official `trains` task adapted to DuckDB earned **[0.6286, 1, 0, 1]**; two pipelines passed strict grading and replay before the update.
- One real RL update was verified by checkpoint reload and changed fixed action-token logprobs.
- The sole post-update evaluation scored **0.6286**. General performance improvement has not been established.

Official source and reference data are unchanged; DuckDB is an adapted destination.

## Documentation

[Report](docs/report.md) · [Design](docs/design.md) · [Run instructions](docs/running.md) · [Experiment evidence](docs/evidence/2026-10-12-real-experiment)
