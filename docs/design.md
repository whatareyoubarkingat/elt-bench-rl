# Implementation design

The environment trains a model to construct an ELT pipeline through tools,
execute it, use errors, and submit an independently reproducible pipeline.
Initial task: `trains`, with CSV `train`, HTTP JSON `cars`, and a dbt `trains`
result containing seven requested columns.

## State and components

```mermaid
flowchart LR
    Model --> Tools[Cookbook tool environment]
    Tools --> Episode[Isolated stateful episode]
    Episode --> Loader[Local CSV / HTTP loader]
    Loader --> Raw[Raw tables]
    Episode --> dbt[Controlled dbt models]
    Raw --> dbt --> Output[Transform tables]
    Episode --> Bundle[Frozen logical pipeline]
    Bundle --> Replay[Fresh namespace replay]
    Output --> Verifier[Trusted fresh-query verifier]
    Replay --> Verifier
    Verifier --> Reward --> Tinker[Cookbook RL optimizer]
```

`TaskSpec` reads the pinned benchmark's `config.yaml`, source schema descriptions,
`data_model.yaml`, and sort keys. Connection examples and target references
never enter the prompt. Trusted source caches and target references are separate.
Fixed upstream repositories live in an external cache, outside this English-only
project repository; runtime code verifies their revisions before use. Cookbook
is installed from its exact Git revision through the dependency lock.

`DestinationAdapter` owns namespace creation, queries, base-table reads,
dbt profiles and cleanup. The default DuckDB path uses independent files with
`raw`/`transform` schemas. Its CLI starts an ephemeral Python HTTP source server
for the entire rollout/replay and closes it on exit. No Docker or database
credentials are required. Databricks is retained only as an optional adapter.

`LocalIngestor` parses CSV and fetches HTTP JSON, coerces specified types,
and replaces raw tables. The optional `AirbyteIngestor` shares its receipt interface but
uses source definitions, check/discovery, destination configuration, a
connection and a terminal-successful asynchronous sync. Healthy services or
successful connections do not prove successful ingestion.

State includes logical load configuration, receipts, SQL, execution attempts,
tool history, frozen submission and grades. State persists within an attempt;
attempts are isolated. Cookbook tools execute sequentially and stop processing
the batch immediately after submission. Stop, timeout, resource limits and
failures also end the attempt. Unsubmitted attempts receive zero.

## Submission and reward

`PipelineBundle` contains task/revision, source names and replacement modes,
fixed dbt project settings, and model SQL. No runtime schema, connector IDs,
paths or credentials are included. Submission snapshots the bundle and digest,
then executes extraction/loading/dbt in a new empty warehouse. Grading always
queries fresh warehouse tables and ignores candidate CSV evaluation caches.

For valid submissions: `reward = 0.2 L + 0.5 T + 0.3 R`.

- `L`: fraction of sources with trusted receipts and correct fresh schema/content.
- `T`: key-aligned correct-cell fraction over requested targets, zero unless
  all sources load correctly. Missing/extra rows/columns and duplicates are
  penalized. Final bundle replay also bounds this score, preventing stale
  tables from rewarding a subsequently broken model.
- `R`: one only when clean replay loads all sources, executes dbt and strictly
  matches every target.

The official functions provide numeric `rtol=0.01`, `atol=1e-9`, null equality
and trimmed case-insensitive strings. Official Stage-1/2 metrics are computed
on fresh results separately. Reward primary keys match exactly after format
normalization; official comparison may accept extra columns, duplicate keys
or close but wrong numeric keys. Those cannot earn strict replay reward.
Official Stage 2 is gated by official Stage 1.

Trusted cache/reference, network/authentication/permission failures are
infrastructure errors excluded from learning. Missing tables, columns and
SQL conversion failures are candidate errors. Reads have bounded retries;
ambiguous mutation outcomes are reconciled rather than blindly repeated.

## Safety and limitations

The model gets explicit structured tools, no shell/file access. It cannot
choose endpoints, alter dbt configuration, install dependencies, read secrets,
read references or modify grading code. Parsed SQL permits a single SELECT,
allowlisted functions and episode-owned tables. External/table functions,
catalog access, attach/copy commands, recursive CTEs, parameters and arbitrary
Jinja are blocked. DuckDB external access and extension installation/loading
are disabled. Trusted dbt profiles are mode 0600 and deleted on every exit.
SQL/models, query rows, turns, tokens, execution time, memory and feedback
size have explicit limits.

Dependency checks reject pure constant submissions and require reachable raw
source dependencies. They cannot prove semantic data use: memorized static
answers with dummy dependencies can still fool a fixed public dataset.
Stronger protection needs held-out tasks or trustworthy source perturbations
with corresponding references. Benchmark contamination is not claimed solved.
This is not a sandbox for executing arbitrary Python or shell code.

## Training and efficiency

`ELTDatasetBuilder`/`ELTEnvGroupBuilder` implement Cookbook interfaces. Lazy
initialization prevents waiting serial attempts from consuming their timeout.
The recipe reuses `do_group_rollout`, `compute_advantages`,
`assemble_training_data`, `train_step`, and checkpoint utilities. Only assistant
action tokens receive loss; tool outputs/observations are masked.

Defaults: Qwen/Qwen3.5-4B, LoRA 16, LR 1e-5, four attempts, temperature 1,
24 turns, 8192 total sampled tokens, serial execution, one update. Advantage
is reward minus group mean. Constant groups are resampled at most twice;
persistent zero effective advantage aborts the update. Optimizer futures are
awaited, state/sampler checkpoints saved, and fixed action-token logprobs
compared before/after. No observed change fails update verification.

Full grading/replay occurs only at submission. Immutable public assets and
tokenizers are cached, services reused, outputs capped, and all exits clean
owned namespace resources. New destinations reuse the same episode/RL loop.

Offline mocks do not prove real model sampling or training. A scripted DuckDB
pipeline does not prove a model-generated official-task rollout. The current
route adapts the official Databricks task specification and reference data to
DuckDB while retaining the pinned official comparator. Tinker remains remote
and uses real API credentials. Whether this fulfills the grader's credentialed
warehouse requirement remains an acceptance interpretation, not an observed
result. See [running.md](running.md) and [acceptance.md](acceptance.md).
