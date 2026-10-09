# Local setup on macOS

This setup prepares the code, dependencies, source data, and local services for
implementation. It does not establish that an ELT environment, official rollout,
or Tinker RL optimization has been implemented or passed its acceptance test.
Local preparation requires CPU, memory, and disk; no local GPU or model weights
are needed.

## Prepared components

- Apple Silicon (`arm64`) Python 3.11, `uv`, and the project's `.venv`.
- Python dependencies resolved by `uv.lock`, including Tinker, the cookbook,
  DuckDB, Snowflake connectivity, dbt, and development tools.
- Homebrew tools listed in `Brewfile`: Docker CLI/plugins, Colima, `abctl`,
  Terraform, and PostgreSQL's `psql` client (`libpq`).
- The two upstream repositories as submodules at fixed commits.
- The small official `trains` source files and Snowflake ground truth.
- A pinned Qwen tokenizer and tool-call renderer prerequisites, without weights.
- A dedicated Colima profile and a project Docker wrapper.

Upstream revisions are recorded in `configs/upstream-revisions.json`:

| Component | Fixed revision |
| --- | --- |
| `third_party/elt-bench` | `fcf3129df49ce5ce63ff46b2bbbbba1eaf9ec055` |
| `third_party/tinker-cookbook` | `cd008aacedbf0e6f7452b3ec106a345d29482ea5` |
| Hugging Face `tttjjj/elt_bench` | `52a48541547a4340e19e54386350740acca80146` |
| `Qwen/Qwen3.5-4B` tokenizer | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

The tokenizer selection prepares local formatting checks. A usable Tinker model
still needs to be confirmed with an authorized Tinker account.

The official task specification is already available under
`third_party/elt-bench/elt-bench/snowflake/trains/`; source schemas are under
`third_party/elt-bench/elt-bench/schemas/trains/`. The evaluation query is
`third_party/elt-bench/evaluation/sql/trains/trains.sql`; it reads the submitted
result rather than providing a reference transformation.

`configs/airbyte-trains.yaml` adapts the upstream Snowflake source manifest to
the `cars` stream and checks that stream instead of an unrelated dataset. It
retains the upstream [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
license and credits [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench).

## Restore the development environment

Run commands from the repository root. Homebrew must already be installed.

```bash
brew bundle --file Brewfile
bash scripts/bootstrap.sh
```

The bootstrap initializes pinned submodules, runs `uv sync --locked` with
Python 3.11, downloads the selected data/tokenizer, and performs local checks.
It does not deploy Airbyte, contact Snowflake, or start Tinker training.

The individual commands are:

```bash
git submodule update --init --recursive
uv sync --locked --python 3.11
.venv/bin/python scripts/fetch_trains_data.py
.venv/bin/python scripts/fetch_tokenizer.py
.venv/bin/python scripts/check_tokenizer.py
.venv/bin/python scripts/check_prereqs.py --skip-docker
```

Use `.venv/bin/python` or `uv run --locked` for project commands. The prerequisite
checker also finds `.venv/bin` tools and `/opt/homebrew/opt/libpq/bin/psql` when
the virtualenv has not been activated in the shell.

## Selected data

| Local path | Data rows | Bytes |
| --- | ---: | ---: |
| `data/trains/train.csv` | 20 | 164 |
| `data/trains/cars.csv` | 63 | 3,289 |
| `ground_truth/gt_snowflake/trains/trains.csv` | 20 | 565 |

`scripts/fetch_trains_data.py` downloads only these selected official assets.
It uses HTTP ranges to obtain `cars.csv` from the source API ZIP without
downloading the full archive. Downloads are checked against pinned SHA-256
hashes and row counts. Provenance is recorded in `data/manifest.json`.

The ground truth is stored separately from source data. It is trusted verifier
material and must remain unavailable to the model during future rollouts.

`scripts/fetch_tokenizer.py` prepares tokenizer/config/template files under
`data/tokenizers/Qwen3.5-4B` (about 23 MB), with no model weights. The offline
`scripts/check_tokenizer.py` validates these files and cookbook tool-call
formatting without making Tinker requests.

Data, tokenizers, environments, local Docker configuration, and execution state
are ignored by Git and are recreated locally.

## Dedicated Docker environment

The project uses Colima profile `elt-bench-rl` and Docker context
`colima-elt-bench-rl`, with 4 CPUs, 8 GiB RAM, a 20 GiB data disk, and a 10 GiB
root disk. The initial configuration uses Apple's VZ backend and Rosetta for
images that require x86 emulation:

```bash
PROJECT_ROOT="$(pwd)"
colima start elt-bench-rl \
  --cpu 4 --memory 8 --disk 20 --root-disk 10 \
  --vm-type vz --vz-rosetta \
  --activate=false --ssh-config=false \
  --mount "$PROJECT_ROOT:r"
.venv/bin/python scripts/configure_colima_dns.py
```

Once configured, use the saved profile:

```bash
colima start elt-bench-rl
colima status elt-bench-rl
colima stop elt-bench-rl
```

Stopping the profile preserves its containers and volumes. Starting it again
restores the VM; individual application health must still be checked.

Use `scripts/with_docker.sh` for project service commands:

```bash
bash scripts/with_docker.sh docker info
bash scripts/start_sources.sh
bash scripts/with_docker.sh docker compose -f configs/sources.compose.yml ps
bash scripts/with_docker.sh docker compose -f configs/sources.compose.yml stop
```

The wrapper selects the project daemon through `DOCKER_HOST` and uses
`.local/docker` for Docker configuration. It does not switch the user's default
Docker context. The profile's project mount is read-only; container volumes hold
writable service data. The source compose configuration exposes only source
data, never `ground_truth/` or `.env`.

`start_sources.sh` also connects the source API container to Airbyte's Kind
network, making the `elt-api` hostname reachable from its cluster while keeping
the Kubernetes node on its original network. It forwards
the Docker daemon's configured proxy to the build when available. Override with
`ELT_BENCH_RL_CONTAINER_PROXY` if needed; this variable applies to containers and
does not change the Mac's proxy environment.

## Airbyte

Airbyte is deployed through `abctl` on the same isolated daemon, with
authentication enabled and local access on port 8000:

```bash
bash scripts/install_airbyte.sh
bash scripts/with_docker.sh abctl local status
bash scripts/start_sources.sh
```

The local UI is at `http://localhost:8000`. Save generated credentials directly
into local configuration; do not paste them into logs or commit them. Stopping
the Colima profile stops the local Airbyte environment while preserving state.

The installer pins chart 2.4.0 and captures installation output in a private
`.local/airbyte-install.log`. It corrects a loopback proxy inherited by Kind's
containerd and prepares the database volume's top-level ownership for UID 70.
The DNS helper repairs only a broken resolver symlink in this dedicated VM,
using the nameserver from the VM's own network configuration. A repaired DNS
configuration requires restarting its Docker daemon and checking services.

Local HTTP checks should bypass shell proxy settings:

```bash
curl --noproxy '*' --fail http://localhost:8000/api/v1/health
curl --noproxy '*' --fail http://127.0.0.1:5005/trains/cars
```

Verified on the prepared Apple M3 Mac on 2026-10-09:

| Service | Verification status |
| --- | --- |
| Project Docker daemon | Running; Docker Engine 29.5.2, native ARM64 |
| Airbyte | Chart/app 2.4.0 deployed; UI and health HTTP 200; `available=true` |
| Airbyte authentication | Client token and workspace API HTTP 200; local `.env` configured |
| `trains` source API | Healthy; 63 rows, source IDs and columns verified |
| Airbyte-to-source network | Manifest-server Pod read all 63 rows via `http://elt-api:5005/trains/cars` |

The task-specific manifest is prepared, but connector publication and Airbyte
source/check/sync integration remain part of implementation. A trial of the
public declarative-definition creation API returned HTTP 409 without a detailed
cause; this does not establish a successful connector registration. The local
service and direct source access checks above passed independently.

Python 3.11.13, Tinker 0.33.1, Torch 2.14.1, DuckDB 1.5.6, dbt-core 1.12.5,
and Snowflake connector 4.8.0 were imported or checked successfully. NumPy is
pinned below 2 for the benchmark; PyArrow 23.0.1 avoids the NumPy 2 requirement
of newer wheels.

The bootstrap was replayed successfully. Selected cookbook tokenizer, message
environment, multi-turn weight, and tool execution tests passed: **61 tests**.
The offline Qwen renderer check verified exact Hugging Face/cookbook prompt
tokens, supervised tool-call weights, and tool argument parsing. Both asset
downloaders were also verified from empty temporary directories.

The cookbook checks can be reproduced with:

```bash
.venv/bin/pytest -q \
  third_party/tinker-cookbook/tinker_cookbook/tokenizer_utils_test.py \
  third_party/tinker-cookbook/tinker_cookbook/rl/message_env_test.py \
  third_party/tinker-cookbook/tinker_cookbook/rl/multiturn_weight_assignment_test.py \
  third_party/tinker-cookbook/tinker_cookbook/tool_use/tool_execution_test.py
```

## Credentials and checks

Create a local configuration without overwriting an existing one:

```bash
cp -n configs/.env.example .env
```

Fill in `.env` locally:

- Tinker: `TINKER_API_KEY`.
- Snowflake: `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`,
  `SNOWFLAKE_WAREHOUSE`, and the desired database/role.
- Airbyte: `AIRBYTE_CLIENT_ID`, `AIRBYTE_CLIENT_SECRET`, and
  `AIRBYTE_WORKSPACE_ID`. The example points `AIRBYTE_URL` to the local service.
- Docker context: `ELT_BENCH_RL_DOCKER_CONTEXT` defaults to
  `colima-elt-bench-rl`.

The prerequisite checker recognizes `SNOWFLAKE_PRIVATE_KEY_PATH` as an
alternative credential-presence signal, but does not validate authentication
or implement the future warehouse adapter's authentication behavior.

```bash
.venv/bin/python scripts/check_prereqs.py
.venv/bin/python scripts/check_prereqs.py --json
.venv/bin/python scripts/check_prereqs.py --skip-docker
```

The checker imports the training dependencies, executes a CPU Torch operation
and an in-memory DuckDB query, checks submodule SHAs and selected asset hashes,
and checks the dedicated Docker daemon and `abctl`. It reads `.env` and reports
credential presence as booleans only, without authenticating or using paid APIs.

- `local_not_ready`: a required local prerequisite failed; exit code 1.
- `missing_credential`: local prerequisites passed, but configuration is missing;
  exit code 0.
- `ready`: local prerequisites and credential-presence checks passed;
  account access and task execution remain unverified.
- `--skip-docker`: explicitly leaves Docker and `abctl` unverified and checks
  the Python/data subset.

Without Snowflake credentials, a real official Snowflake rollout cannot be
validated. Without Tinker credentials, a real RL optimization step cannot be
performed. Preparation checks and upstream smoke tests do not satisfy these
acceptance criteria; the environment, execution verifier, and training recipe
must still be implemented and tested.
