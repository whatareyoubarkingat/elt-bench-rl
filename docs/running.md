# Running and collecting evidence

Use the locked environment from the repository root. Commands load `.env`
without printing it. Keep secrets out of CLI arguments and public logs.

## Local execution

```bash
uv sync --locked --group dev --python 3.11
uv run --locked pytest -q tests
uv run --locked python scripts/fetch_trains_data.py --destination databricks
uv run --locked elt-rl local-smoke --output runs/local-smoke
uv run --locked elt-rl replay --bundle runs/local-smoke/episode_<id>/pipeline.json --output runs/replay
```

All DuckDB execution commands start and close an ephemeral Python HTTP source
server by default. No separate service or Docker deployment is needed. Set
`--source-api` only to use an existing server. Each episode retains tool trace, dbt feedback,
frozen pipeline/digest and original/replay grades. Warehouse files are removed.
`examples/trains.sql` is human-written and never injected into model prompts.

## Model-generated official-task rollout on DuckDB

Set `TINKER_API_KEY` in `.env` and choose a Tinker spending limit first:

```bash
uv run --locked python scripts/fetch_tokenizer.py
uv run --locked elt-rl preflight --destination duckdb --online
uv run --locked elt-rl rollout --destination duckdb --allow-paid --output runs/official-local
```

Sampling uses real Tinker credentials; extraction/loading, dbt execution and
clean replay run locally. Preserve model tool trajectories, load receipts,
original/replay grades and the bundle. The official `trains` inputs and target
reference are unchanged, but DuckDB is an adapted destination. The assignment
does not mandate Databricks or Airbyte; acceptance of a Tinker-credentialed
local warehouse rollout should be confirmed with the grader.

## Optional Databricks and Airbyte extension

These services are not needed for the default route. Install optional drivers
with `uv sync --locked --extra cloud` before using this extension.

Use a SQL Warehouse and Unity Catalog. The OAuth M2M principal needs catalog
use, episode schema creation/use, tables and the selected Airbyte connector's
staging volume permissions. Restrict it to a training catalog. Free Edition
compatibility is not assumed.

Fill `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_CATALOG`,
`DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`, `AIRBYTE_URL`,
`AIRBYTE_CLIENT_ID`, `AIRBYTE_CLIENT_SECRET`, `AIRBYTE_WORKSPACE_ID` in `.env`.
Destination configuration is connector-version-specific: validate against
the actual Airbyte definition/spec before cloud sync.

```bash
bash scripts/install_airbyte.sh  # only if not installed
bash scripts/start_sources.sh
uv run --locked elt-rl preflight --destination databricks
# The next commands may start billed warehouse / Tinker sessions.
uv run --locked elt-rl preflight --destination databricks --online --allow-paid
uv run --locked python scripts/fetch_tokenizer.py
uv run --locked elt-rl rollout --destination databricks --allow-paid --output runs/official
```

Credential presence is not authenticated access. This optional cloud validation requires
a model-generated trajectory, actual successful Airbyte jobs for both sources,
Databricks execution, fresh two-stage metrics, a bundle and clean replay.
Health/check/discovery alone is insufficient. Preserve job IDs in trusted
receipts, not bundles. Do not publish secrets in connector manifests.

## Real one-step RL

Choose a spending limit and set `TINKER_API_KEY`, then:

```bash
uv run --locked elt-rl train --destination duckdb \
  --config configs/train.yaml --allow-paid --output runs/rl-one-step
```

The SDK performs inference/training on remote GPUs. The Mac hosts ELT execution
and tokenizers; model weights remain remote. The budget may include twelve
attempts (three groups of four), checkpoints, two logprob probes, and one
optimizer step. Token limits bound generation, not money; enforce provider
account limits when running paid work.

Required evidence: generated trajectories/logprobs, execution rewards,
nonzero advantages, unmasked effective action tokens, awaited optimizer
responses, state/sampler checkpoints and a changed fixed action-token logprob
probe. Injected runtimes are explicitly labeled. `no_learning_signal` and
`update_unverified` do not pass effective-training acceptance.

Use the sampler checkpoint with a Tinker sampling client; load the state
checkpoint into a compatible LoRA training client to restore optimizer state.
Keep paths and `training_result.json` with the revision/recipe manifest. Verify
model availability and account access when executing with the real account.

To resume, add `load_checkpoint_path: tinker://...` to a training YAML. To
evaluate a trained policy, add `sampler_checkpoint_path: tinker://...` to a
rollout YAML and pass `--config`; state and sampler paths have different roles.

## Failures and cleanup

Infrastructure errors stop unusable samples; invalid SQL returns repairable
feedback. Episode/group cleanup runs on failure. Incomplete cloud cleanup writes
a private mode-0600 `cleanup_pending.json` containing owned namespace/schema/
actor/job identities for recovery after process exit; it excludes connection
configurations and credentials. Successful same-process retry removes the pending
record. Never delete unrelated schemas/jobs. Airbyte jobs/connections are cleaned
before warehouse schemas.
