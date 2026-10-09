# ELT-Bench RL

An end-to-end reinforcement learning with verifiable rewards (RLVR) environment for ELT-Bench, using Tinker and reusing tinker-cookbook where practical.

## Status

Development prerequisites are prepared: pinned upstream repositories, a locked
Python environment, official `trains` data, tokenizer assets, and local service
configuration. The ELT environment, execution verifier, official-task rollout,
and Tinker training integration still need implementation.

See [local setup](docs/setup.md) for installation, service commands, credential
configuration, and verified results.

```bash
git clone --recurse-submodules https://github.com/whatareyoubarkingat/elt-bench-rl.git
cd elt-bench-rl
brew bundle --file Brewfile
bash scripts/bootstrap.sh
```

## Goal

Train a language model to inspect an ELT task, configure extraction and loading, implement transformations, use execution feedback, and submit a reproducible pipeline. Rewards must come from verification of the resulting warehouse state.

## Acceptance criteria

- [ ] Credential-free local integration test with actual data execution.
- [ ] Credentialed end-to-end rollout on at least one official ELT-Bench task.
- [ ] At least one real Tinker RL optimization step using execution-derived rewards.
- [ ] Concise implementation design in a Google Doc and reproducible repository documentation.

## Planned implementation

- A task loader for official benchmark specifications and local test fixtures.
- A stateful multi-turn environment with isolated resources for every episode.
- Destination adapters: DuckDB for local tests and Snowflake for the first official rollout.
- A trusted execution verifier with an explicit reward definition.
- A training recipe that reuses tinker-cookbook rollout and training infrastructure.

The first proposed official task is `trains`. Local fixtures and official-task results will be reported separately.

## Repository layout

```text
configs/             Credentials template, source services, pinned upstream revisions
docs/                Design and local setup guide
scripts/             Reproducible dependency/data setup and prerequisite checks
src/elt_bench_rl/    Python package
tests/               Acceptance-test plan
third_party/         Pinned ELT-Bench and tinker-cookbook submodules
pyproject.toml       Dependencies and package metadata
uv.lock              Resolved Python dependency versions
```

Python 3.11 is the verified development target. Local setup does not download
model weights or require a GPU. There is no ELT rollout or training command yet.

## References

- [Task description](https://docs.google.com/document/d/1Rr9Y7wG93Fw-vEHgGMuALsIKZMyXO5zRPhlo4NtvrzQ/edit)
- [Tinker documentation](https://tinker-docs.thinkingmachines.ai/tinker/)
- [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [ELT-Bench paper](https://arxiv.org/abs/2504.04808)
- [Official ELT-Bench repository](https://github.com/uiuc-kang-lab/ELT-Bench)
- [DeepSeekMath / GRPO reference](https://arxiv.org/abs/2402.03300)
