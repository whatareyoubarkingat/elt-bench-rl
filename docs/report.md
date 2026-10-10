# ELT-Bench RLVR: Implementation Design and Validation

## Objective and Current Status

Build an end-to-end loop in which a model reads the task, configures extraction/loading, writes dbt transformations, repairs execution errors, submits a pipeline, receives execution-derived rewards, and updates through Tinker. The default route uses local DuckDB, a Python CSV/HTTP loader, and dbt-duckdb. Databricks, Airbyte, and Docker are optional. Tinker provides remote model sampling and training. Local execution is validated; real model rollouts and RL updates remain unrun pending an agreed Tinker budget.

## Environment and Tools

TaskSpec reads the pinned official configuration, source schemas, target model specification, and key definitions. Only the trusted verifier can access reference answers. Each episode begins with an independent empty DuckDB database and retains load configuration, receipts, model SQL, execution feedback, and submission state. Tools read the task, inspect sources, configure and execute loads, query tables, write models, run dbt, and submit. CSV files are actually parsed and HTTP sources actually requested. The CLI automatically starts and closes a local Python source server. Fixed upstream checkouts live outside the repository and are verified by commit, origin, and clean state before use. Warehouse adapters share a common interface; Databricks/Airbyte remain optional extensions.

## Reward, Submission, and Replay

For valid submissions, reward = 0.2 L + 0.5 T + 0.3 R. L checks trusted load receipts and complete source-table content. T measures correct target cells after exact primary-key alignment, penalizing missing or extra data and duplicate keys; T is zero unless all sources are correct. R is one only when loading, dbt execution, and strict target verification all pass in a fresh database. Replay bounds T so stale tables cannot reward broken final SQL. Invalid submissions receive zero. PipelineBundle freezes logical loads, the dbt project, and SQL without credentials or runtime identities. Grading queries fresh tables; the pinned official comparator separately reports Stage-1/2 metrics with its numeric tolerance and null rules.

## Reward Hacking and Efficiency

The model has structured tools without arbitrary shell, file, or network access. SQL accepts controlled source/ref references and a single SELECT, blocking cross-schema/catalog access, external reads, recursive SQL, arbitrary Jinja, and extension installation. Only the loader writes raw tables. dbt references resolve to the isolated episode; candidate CSV evaluation caches are ignored. Dependency checks reject pure constant answers but cannot eliminate memorized public answers with dummy dependencies. Held-out tasks or trusted source perturbations remain future protections. Time, memory, query output, and token budgets are bounded. Immutable sources and tokenizers are cached; full grading occurs only at submission. Exit cleanup removes owned databases and automatic source servers. Infrastructure failures are excluded from model-error rewards.

## Training and Validation Evidence

Reuse Cookbook environment groups, rollouts, trajectory assembly, assistant-token masks, importance_sampling, train_step, and checkpoint helpers. Defaults: Qwen/Qwen3.5-4B, LoRA rank 16, learning rate 1e-5, four attempts per group, temperature 1, 24 turns, 8192 total generated tokens, serial execution, and one update. Advantages are reward minus group mean. Constant groups are resampled at most twice; persistent zero effective advantage aborts the update. Real training evidence must include awaited optimizer responses, a reloadable checkpoint, and changed fixed action-token logprobs.

Local CSV/HTTP loading, real dbt error repair, submission, and clean replay have passed. Correct and incorrect scripted pipelines received rewards of 1.0 and 0.628571. All 193 local tests passed, covering isolation, repeat loading, row order, missing columns, duplicate keys, timeouts, forged caches, reference/directory/schema access, cleanup, and offline training interfaces. Scripted trajectories and mock optimizers do not establish real model or training acceptance. The assignment requires Tinker, at least one warehouse, and a credentialed official-task rollout, without mandating Databricks or Airbyte. This route preserves official trains source/reference bytes and adapts the task to DuckDB; the warehouse is credential-free and real sampling uses a Tinker API key. Grader acceptance of that interpretation remains unconfirmed. Run records under runs/ retain source revision, traces, bundles, metrics, replay, and checkpoint information.

## Code and Run Instructions

[GitHub Repository](https://github.com/whatareyoubarkingat/elt-bench-rl)

[Implementation and Run Commands](https://github.com/whatareyoubarkingat/elt-bench-rl/blob/main/docs/running.md)

[Official ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench)

[Tinker Cookbook RL](https://tinker-docs.thinkingmachines.ai/cookbook/rl/)
