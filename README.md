# ELT-Bench RL

An end-to-end reinforcement learning with verifiable rewards (RLVR) environment for ELT-Bench, using Tinker and reusing tinker-cookbook where practical.

## Status

Project scaffold only. The ELT environment, execution verifier, official-task rollout, and Tinker training integration have not been implemented or validated yet.

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
configs/             Configuration examples to be added during implementation
docs/                Initial design and references
src/elt_bench_rl/    Python package
tests/               Acceptance-test plan
pyproject.toml       Package metadata and build configuration
```

Python 3.11 or later is the initial development target. Runtime dependencies and upstream commits will be pinned when the integrations are implemented. There is no rollout or training command yet.

## References

- [Task description](https://docs.google.com/document/d/1Rr9Y7wG93Fw-vEHgGMuALsIKZMyXO5zRPhlo4NtvrzQ/edit)
- [Tinker documentation](https://tinker-docs.thinkingmachines.ai/tinker/)
- [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [ELT-Bench paper](https://arxiv.org/abs/2504.04808)
- [Official ELT-Bench repository](https://github.com/uiuc-kang-lab/ELT-Bench)
- [DeepSeekMath / GRPO reference](https://arxiv.org/abs/2402.03300)
