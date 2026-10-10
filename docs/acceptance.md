# Acceptance and evidence

The implementation is runnable locally. Two assignment acceptance gates still
require authorized cloud execution: an official model-generated Databricks
rollout and a real Tinker parameter update. No paid calls were made during local
development.

| Gate | Observed result |
| --- | --- |
| Credential-free CSV and HTTP extraction into an initially empty DuckDB | Passed |
| Actual dbt execution, SQL error feedback, repair and submission | Passed |
| Frozen pipeline replay into another empty warehouse | Passed |
| Correct versus incorrect execution rewards | 1.0 versus 0.6285714285714286 |
| Official strict metrics on the scripted correct pipeline | Stage 1 and Stage 2 passed in the original execution and replay |
| Cookbook tools, trajectory masks, advantages and optimization wiring | Offline integration and injected-client tests passed |
| Official model-generated Databricks rollout and real Airbyte synchronization | Not run |
| Real Tinker optimizer response, checkpoint and observed parameter change | Not run |

The local demonstration uses pinned official `trains` source bytes and target
references, while automated integration tests use committed synthetic fixtures.
The demonstration and offline toy model are explicitly labeled; neither is
evidence of model-generated official success or a remote GPU update.

## Reproduce the local gate

```bash
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
release record is kept in `validation.json`.

Tests cover isolation, repeat loading, actual error repair, clean replay,
row-order independence, missing columns, duplicate keys, timeouts, forged old
CSV caches, stale target tables, reference access, directory/schema escape,
SQL/Jinja capability restrictions, cleanup failures, cloud adapter behavior,
constant-reward groups and checkpoint verification. Cloud and optimizer mocks
are identified as such.

## Airbyte local prerequisite

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

## Remaining cloud gates

Follow [running.md](running.md) after configuring Databricks OAuth M2M and
agreeing a spending limit. Keep generated tool trajectories, successful Airbyte
job receipts, original/replay grades and the submitted pipeline for the official
rollout. Keep nonzero advantages, effective assistant tokens, awaited optimizer
responses, reloaded state checkpoint and changed action-token logprobs for RL.
All-equal rewards or an unchanged verification probe fail effective-training
acceptance.

The concise native [Google Doc design](https://docs.google.com/document/d/1SWoXsokaq7h7AoHn0ud66ymrli8wtcF7yN_bWqMuZUc)
records the environment, reward, replay, safety, efficiency and current results.
