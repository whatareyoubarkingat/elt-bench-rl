# ELT-Bench RLVR: Implementation Design and Validation

## Objective and Current Status

We implement an ELT-Bench RLVR environment in which a model loads data, writes dbt SQL, uses execution feedback, and submits a reproducible pipeline. DuckDB runs locally; Tinker provides remote model sampling and training. The implementation reuses Tinker Cookbook and supports additional warehouse adapters.

## Environment and Tools

The official trains task is adapted to DuckDB without changing source or reference data. Each attempt starts in an isolated empty database. Structured tools support task inspection, CSV/HTTP loading, queries, SQL editing, dbt execution, and submission. State persists within an attempt; reference answers remain private to the verifier. No cloud warehouse credentials are used.

## Reward, Submission, and Replay

Reward = 0.2 L + 0.5 T + 0.3 R.

L verifies source schemas and contents using trusted load records. T compares target cells by key, penalizes missing, extra and duplicate records, and requires correct loading. R is 1 only when the submitted pipeline passes strict grading in a fresh warehouse. Fresh replay also constrains T. Invalid or missing submissions receive zero. Submissions exclude credentials and temporary identifiers; official Stage 1/2 results are reported separately.

## Reward Hacking and Efficiency

The model cannot read reference answers, run shell commands, or query external data or other namespaces. Only the loader writes source tables, and grading reads fresh database state. Constant pipelines are rejected; memorized answers remain a limitation requiring held-out tasks or data perturbations. Source/tokenizer caching and grading only at submission reduce latency. Resources are cleaned after each attempt, and infrastructure failures are handled separately from model errors.

## Training and Validation Evidence

With Qwen/Qwen3.5-4B and real Tinker API credentials, four training rollouts earned [0.6286, 1, 0, 1]. Two pipelines passed strict grading and clean replay before the update.

One execution-reward optimizer step completed. Its saved checkpoint was independently reloaded, and changed fixed action-token logprobs confirmed the parameter update.

The sole post-update evaluation scored 0.6286 (L=1, T=6/7, R=0). General performance improvement has not been established.

Credential-free local integration passed, and 262 automated tests passed. These results demonstrate the three required validation paths. Detailed evidence is retained in docs/evidence/2026-10-12-real-experiment.
