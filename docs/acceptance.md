# Acceptance and evidence

The default implementation runs locally with DuckDB, a Python CSV/HTTP loader
and dbt-duckdb. Two acceptance gates still require authorized real Tinker calls:
a model-generated official-task rollout and a real parameter update. No paid
calls were made during local development. Databricks and Airbyte are optional
extensions and no longer part of the default development route.

| Gate | Observed result |
| --- | --- |
| Credential-free CSV and HTTP extraction into an initially empty DuckDB | Passed |
| Actual dbt execution, SQL error feedback, repair and submission | Passed |
| Frozen pipeline replay into another empty warehouse | Passed |
| Correct versus incorrect execution rewards | 1.0 versus 0.6285714285714286 |
| Official strict metrics on the scripted correct pipeline | Stage 1 and Stage 2 passed in the original execution and replay |
| Cookbook tools, trajectory masks, advantages and optimization wiring | Offline integration and injected-client tests passed |
| Official-task model-generated rollout on DuckDB with real Tinker credentials | Not run; grader acceptance of the adapted destination unconfirmed |
| Real Tinker optimizer response, checkpoint and observed parameter change | Not run |

The local demonstration uses pinned official `trains` source bytes and target
references, while automated integration tests use committed synthetic fixtures.
The demonstration and offline toy model are explicitly labeled; neither is
evidence of model-generated official success or a remote GPU update.

## Reproduce the local gate

```bash
python3 scripts/fetch_upstreams.py
uv sync --locked --group dev --python 3.11
uv run --locked pytest -q tests
uv run --locked python scripts/fetch_trains_data.py --destination databricks
uv run --locked elt-rl local-smoke --output runs/release-validation
```

`runs/release-validation/local-smoke.json` records the source revision, rewards,
official metrics, frozen bundle hashes and episode paths. Each episode retains
`tools.json`, `grade.json`, `pipeline.json`, dbt artifacts and replay evidence;
warehouse files are cleaned. Runs are ignored because credentials, provider
identifiers and task data should not be automatically published. A sanitized
release record is kept in [validation.json](validation.json).

Latest local-route validation (2026-10-10): **176 tests passed**, no failures or
skips, with cloud Python packages absent. Ruff, formatting, dependency-lock
and configured-secret checks passed. Runtime revision
`209028a6116b858b48aac90b556c08d0ee076718` passed the official-data smoke and a
separate replay CLI invocation without a manually started source service.
The sanitized record is [local-route-validation.json](local-route-validation.json).
The earlier 171-test cloud-route development record remains in
[validation.json](validation.json) as historical evidence.

Tests cover isolation, repeat loading, actual error repair, clean replay,
row-order independence, missing columns, duplicate keys, timeouts, forged old
CSV caches, stale target tables, reference access, directory/schema escape,
SQL/Jinja capability restrictions, cleanup failures, cloud adapter behavior,
constant-reward groups and checkpoint verification. Cloud and optimizer mocks
are identified as such.

## Historical optional Airbyte checks

The following records concern the previous cloud route; they are not local
DuckDB prerequisites. Existing services and volumes have been preserved.

The local Airbyte 2.4.0 database lacked the declarative connector image-version
seed required by CDK 6.33.4. The narrowly scoped
[seed repair](airbyte-seed-repair.md) verifies the official registry digest and
inserts only an absent major-version entry. It does not restart services or
replace existing version rows. Registration/check/discovery evidence is separate
from a successful warehouse sync.

After the repair, base-image lookup and manifest publication succeeded. A
temporary source was created for one real check request; that request reached
its 120-second read timeout. It was not retried, discovery was not started, and
the temporary source was successfully deleted. No destination, connection or
warehouse sync was created. Source-check readiness remains unverified.
The published definition was subsequently read back and its exact manifest
digest matched. One duplicate unpublished draft left by our failed registration
was removed; the unique published definition remains as an immutable cache.
The final source listing contained no temporary probe sources.

## Remaining model and training gates

Follow the default DuckDB commands in [running.md](running.md) after agreeing
a Tinker spending limit. Keep generated tool trajectories, trusted local load
receipts, original/replay grades and the submitted pipeline. Keep nonzero
advantages, effective assistant tokens, awaited optimizer
responses, reloaded state checkpoint and changed action-token logprobs for RL.
All-equal rewards or an unchanged verification probe fail effective-training
acceptance.

The assignment explicitly requires Tinker and at least one warehouse, but does
not name Databricks or Airbyte as requirements. This implementation adapts the
official Databricks task variant to DuckDB using unchanged source/reference
bytes. Tinker API credentials are real; the warehouse is credential-free.
Do not claim the credentialed-rollout gate passed without an actual successful
model run and confirmation of the grader's intended warehouse requirement.

The concise native [Google Doc design](https://docs.google.com/document/d/1SWoXsokaq7h7AoHn0ud66ymrli8wtcF7yN_bWqMuZUc)
records the environment, reward, replay, safety, efficiency and current results.
