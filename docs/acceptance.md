# Acceptance and evidence

The default implementation runs locally with DuckDB, a Python CSV/HTTP loader
and dbt-duckdb. Authorized real Tinker experiments started on 2026-10-12.
Three sampling rollouts completed, including one partially correct submission.
The first real training group produced two strict-successful pipelines and
completed an optimizer step, then failed during state-load verification.
Separate non-injected real-API recovery verified the saved update without
repeating optimization. The sole post-update evaluation was partially correct;
general policy improvement is not established.
Offline tests make no paid calls. Databricks and Airbyte remain optional.

| Gate | Observed result |
| --- | --- |
| Credential-free CSV and HTTP extraction into an initially empty DuckDB | Passed |
| Actual dbt execution, SQL error feedback, repair and submission | Passed |
| Frozen pipeline replay into another empty warehouse | Passed |
| Correct versus incorrect execution rewards | 1.0 versus 0.6285714285714286 |
| Official strict metrics on the scripted correct pipeline | Stage 1 and Stage 2 passed in the original execution and replay |
| Cookbook tools, trajectory masks, advantages and optimization wiring | Offline integration and injected-client tests passed |
| Real Tinker credential check | `KEY_OK`; 18 input and 3 output tokens |
| Model-generated official-task pipeline executed on DuckDB | Rollout 03 submitted: L=1, T=6/7, R=0; reward 0.6285714285714286 |
| Fully correct official-task model submission | Two Training 01 attempts earned reward 1 with fresh strict replay |
| Real Tinker optimizer response | Completed with execution-derived rewards and nonzero effective advantages |
| Reloadable checkpoint and verified model change | Passed through separate fresh-model optimizer-state restore; 814-token max logprob delta 0.19415616989135742 > 1e-8 |
| Single post-update policy evaluation | Submitted; reward 0.6285714285714286, Stage 1=true/Stage 2=false; no improvement claim |

The local demonstration uses pinned official `trains` source bytes and target
references, while automated integration tests use committed synthetic fixtures.
The demonstration and offline toy model are explicitly labeled; neither is
evidence of model-generated official success or a remote GPU update.

## Real Tinker experiments: 2026-10-12

The credential check and three rollouts used real API access. Each rollout
records `injected_runtime=false`. A zero process exit code indicates completed
execution, not a passing target or a verified update.

| Phase | Policy/outcome |
| --- | --- |
| Rollout 01 | Reward 0; reached 8192 sampled tokens without submitting. |
| Rollout 02 | Reward 0; used real dbt and repaired an execution error, then reached 24 turns without submitting. |
| Rollout 03 | Experimental 32-turn cap, no-thinking prompt and temperature 0.6; submitted at turn 27. Loading=1, target=0.8571428571428571, strict replay=0; official Stage 1=true, Stage 2=false; reward=0.6285714285714286. |
| Training 01 | Rewards [0.6285714285714286, 1, 0, 1]; two strict-successful pre-update pipelines/replays. Optimizer completed; original state-load verification failed HTTP 400. |
| Checkpoint verification | Separate real-API fresh-model recovery passed; saved update verified, no repeated optimizer. |
| Post-update evaluation | One step-1 sampler rollout at temperature 0.6/max 32 turns/8192 sampled tokens. Submitted after 13 turns/2079 generated tokens; reward 0.6285714285714286, L=1, T=6/7, R=0; Stage 1=true/Stage 2=false; fresh replay partial. |

The 32-turn setting is an experiment override; the default remains 24 turns.
A compatibility fix admits legitimate `EXISTS` predicates and correctly scoped
CTE references while preserving recursive capability/dependency checks.
The official task, public source bytes, data and target references were not
changed to improve results. Target references remain verifier-only and never
enter actor prompts/tools. No reward or grading rule was relaxed.

Latest pre-launch validation: **252 full-suite tests passed** after the SQL
compatibility fix; **43 targeted prompt/security tests passed**. These checks
overlap and are not a combined 295-test total. They use offline/injected model
clients and establish implementation behavior, not real training completion.
After the checkpoint-verification fix, **262 full-suite tests passed in 47.36
seconds**, **35 focused training tests passed in 3.91 seconds**, and Ruff/format
checked **39 files**.
These test counts also overlap. Additional offline provenance hardening
reproduced the same 6206-token probe prompt, 814 action tokens and checkpoint
pair against the actual run, with six unchanged hashes and no API call.

Private phase records under `runs/real-experiment-20261012/` retain manifests,
model trajectories, tool/dbt feedback, grades, frozen pipelines, progress and
watchdog PID/status/log evidence. Training 01 had nonzero advantages
[-0.028571426868, 0.342857122421, -0.657142877579, 0.342857122421],
11981 effective assistant/action tokens and 49978 total sequence tokens across
five training data items. The optimizer response was awaited. The process exited
1 when same-client `LoadWeights` returned `BadRequest` HTTP 400
(`LoadWeights not permitted`, sequence 7) during checkpoint verification.
This is a post-update verification failure; it does not mean the optimizer was
never executed. Recovery on source revision `329249c` created a fresh model
with optimizer-state restoration instead of loading into the advanced client.
The separate `checkpoint-verification/verification.json` reports
`status=verified_saved_update`, `injected_runtime=false`,
`original_injected_runtime=false`, `state_load_verified=true`,
`effective_update=true` and `optimizer_repeated=false`. The fixed probe contained
814 action tokens; maximum absolute logprob change was 0.19415616989135742,
above tolerance 1e-8. Fresh restore took 16.8 seconds and the recovery watchdog
completed in 40.164 seconds. The original failed manifest remained unchanged.

The single post-update evaluation used the saved step-1 sampler, unchanged
public task and no answer hints. Its watchdog completed in 105.377 seconds.
It was not rerun or optimized again. The observed partial result is reported
without a general improvement/degradation claim. All paid experiment processes
have exited; no experiment remains active.

See [sanitized public experiment evidence](evidence/2026-10-12-real-experiment).
Credentials and raw prompt/tool-result traces are not published. Sampler and
state checkpoint URIs are retained in the sanitized verification record for audit.

Through rollout 03, the token estimate was about $0.15165 before training,
excluding the storage buffer. At the Training 01 verification failure, the
conservative accounted/reserved estimate was $0.916151581 including the buffer,
about $0.416151581 for token operations across the attempts. After verification
and the sole post-update evaluation, the final uncached token estimate is
**$0.448166396**; with the fixed $0.50 buffer, the final ledger total is
**$0.948166396**, within the $10 estimate limit. This includes the key check,
three baseline rollouts, four training attempts, probes and post-update sample.
All commands share one ledger,
including a $0.50 storage buffer and $0.000008955 prior key-check estimate.
Unknown/timeout reservations remain accounted. Estimates use uncached rates;
they are not an invoice or an account-level cap. Checkpoints saved without TTL
continue to incur actual storage charges while retained. Recorded metadata
contains four files totaling 1,167,002,514 bytes without TTL, approximately
$0.1167/month at $0.10/GB/month; this projected storage is not an invoice.
The watchdog observes
real artifact/log/CPU activity and ignores its own heartbeats for stall detection.
See [running.md](running.md) for budget and retention details.

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

Historical English-only validation (2026-10-10): **193 tests passed**, no failures or
skips. All 74 current project text files and filenames have zero Han-character
matches. At that time, the English native report matched the historical
[report.md](report.md) and retained six section headings/four links. The initial
experiment-report synchronization on 2026-10-12 also verified six native headings,
four original links and zero Han characters; [that document validation](evidence/2026-10-12-real-experiment/google-doc-validation.json)
is a historical record. The current report retains five section headings after
the user-requested removal of the entire "Code and Run Instructions" section
and its four links. Current native-Doc synchronization uses this five-section
local mirror.
Fixed upstream checkouts are
outside the project repository; the same official comparator bytes and source
pins are preserved. Runtime revision `85adec55f5825c38af63d7a328cd146f2d02ee24`
passed the official-data smoke with rewards 1.0 and 0.6285714285714286 and clean
replay. See [english-only-validation.json](english-only-validation.json).

Previous local-route validation (2026-10-10): **176 tests passed**, no failures or
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

The following checks are historical records from 2026-10-10, before cleanup;
they are not current services or DuckDB prerequisites. The explicitly authorized
`elt-bench-rl` Colima profile and its runtime data were deleted later that day.
Project code, sources, tokenizer, runs, `.env` and representative Python files
were preserved; all 474 baseline hashes matched and DuckDB SQL still executed.

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
was removed; the unique published definition was retained at that check's time.
The final source listing then contained no temporary probe sources. The later
profile deletion removed that Airbyte database/cache with its runtime disk.

## Remaining model and training gates

Real sampling and strict model-generated ELT success are now demonstrated.
One real optimizer step and saved model change are verified through separate
checkpoint recovery. The original reload failure remains part of the record.
General quality improvement is not established by the single partial evaluation.
Follow the default
DuckDB commands in [running.md](running.md), preserving the shared budget.
Keep generated tool trajectories, trusted local load
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
