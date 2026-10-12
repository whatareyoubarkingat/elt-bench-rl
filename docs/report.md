# ELT-Bench RLVR: Implementation Design and Validation

## Objective and Current Status

Build an ELT agent that loads sources, writes dbt SQL, repairs errors, submits a pipeline and learns from execution rewards. DuckDB/CSV/HTTP/dbt run locally; Tinker samples and trains remotely. KEY_OK passed with 18 input/3 output tokens. Two pre-update model pipelines passed strict grading/replay. One real optimizer step and its saved update are verified through separate fresh-model recovery after an original HTTP 400 reload failure. The written minimum gates are demonstrated: credential-free local integration, a real Tinker-credentialed rollout on the official task adapted to DuckDB, and an execution-reward RL step. The sole post-update evaluation was partial; general quality improvement is not claimed. Final assessment belongs to the professor.

## Environment and Tools

TaskSpec reads pinned task configuration, schemas, models and keys. Source/data/reference bytes remain unchanged. References are verifier-only, never actor-visible. Each attempt owns an initially empty database with raw/transform schemas. Tools read the task, inspect sources, configure and execute loads, query tables, write SQL, run dbt and submit. CSV parsing and HTTP extraction are real; the CLI owns a temporary local source server. External upstream caches verify revision, origin and clean state. Adapter interfaces permit new destinations without changing the RL loop.

## Reward, Submission, and Replay

For valid submissions, reward = 0.2 L + 0.5 T + 0.3 R. L verifies trusted receipts and fresh source schema/content. T aligns target cells by exact keys, penalizes missing/extra data and duplicates, and requires correct loading. R requires clean loading/dbt replay and strict target correctness. Replay also bounds T, preventing stale tables from rewarding broken final SQL. Missing submissions receive zero. Frozen PipelineBundle loads/dbt/SQL exclude credentials and runtime IDs. Grading queries fresh state; pinned official Stage-1/2 metrics are reported separately.

## Reward Hacking and Efficiency

The actor has structured tools, no shell or arbitrary files/endpoints. Only the loader writes raw tables. SQL permits one SELECT and controlled source/ref, blocking external reads, cross-namespace access, recursive CTEs, arbitrary Jinja and extensions. An EXISTS/scoped-CTE compatibility fix keeps recursive capability and dependency checks intact. Candidate CSV grading caches are ignored. Constant pipelines are rejected; memorized answers with dummy dependencies still require held-out tasks or trusted perturbations. Sources/tokenizers are cached; grading occurs at submission and owned resources are cleaned. Infrastructure failures are excluded from model-error rewards.

## Training and Validation Evidence

The recipe reuses Cookbook rollouts, assistant-token masks, importance_sampling, train_step and checkpoints. Training defaults are Qwen/Qwen3.5-4B, LoRA rank 16, LR 1e-5, four serial attempts, temperature 1, 24 turns, 8192 sampled tokens per attempt and one update. Advantages are reward minus group mean. Constant groups retry at most twice; persistent zero effective advantage aborts optimization. Update acceptance requires awaited optimizer results, effective action tokens, a reloadable state checkpoint and changed fixed action-token logprobs.

| Real experiment | Observed outcome |
| --- | --- |
| Rollout 01 | Reward 0; exhausted 8192 sampled tokens without submission. |
| Rollout 02 | Reward 0; performed real dbt/error repair but exhausted 24 turns without submission. |
| Rollout 03 | Submitted at turn 27 under a 32-turn experimental cap, no-thinking prompt and temperature 0.6. L=1, T=6/7, R=0; Stage 1 passed, Stage 2 failed; reward 0.6285714286. |
| Training 01 | Rewards [0.6285714286, 1, 0, 1], two strict successes/replays, nonzero advantages and an awaited optimizer step. Original state-load verification failed HTTP 400; separate recovery verified the saved update without repeating optimization. |
| Post-update evaluation | One step-1 sampler rollout: 13 turns, 2079 generated tokens, reward 0.6285714286; L=1, T=6/7, R=0, Stage 1 passed/Stage 2 failed. No rerun or improvement claim. |

The 32-turn adjustment is an experiment override; the default remains 24. These are real Tinker trajectories. Rollout 03 was partial; two training attempts achieved strict success. The optimizer used 11981 effective action tokens and 49978 total sequence tokens across five data items. Training exited 1 after optimization; separate non-injected recovery verified optimizer-state restoration and logprob delta 0.19415616989135742 on 814 fixed action tokens (tolerance 1e-8). Pre-launch checks were 252 full-suite/43 targeted tests; final checks were 262 full-suite tests in 47.36s, 35 focused training tests in 3.91s and 39 files checked by Ruff/format. Counts overlap; offline tests make no paid calls.

A shared $10 ledger reserves requests, retains uncertain costs and includes a $0.50 storage buffer plus the prior key-check estimate. The final uncached token estimate is $0.448166396, or $0.948166396 including the buffer; this is not an invoice. Four checkpoints total 1,167,002,514 bytes without TTL, projected at $0.1167/month; the ledger is not a lifetime storage cap. A private watchdog records PID/group, durable progress, logs and timeouts; its own heartbeats do not count as progress. The old optional Airbyte/Colima VM was deleted on 2026-10-10; it is not part of the running local route.

DuckDB retains official Databricks task source/reference bytes and uses no cloud warehouse credentials. Real Tinker API credentials were used for the official-task model rollouts. The implementation and minimum evidence are complete under the written requirements; final assessment belongs to the professor. General quality gains are not claimed. Public evidence is in docs/evidence/2026-10-12-real-experiment.
