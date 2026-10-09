# Initial implementation design

This is a proposal for implementation, not a description of completed features.

## Episode lifecycle

Task specification -> isolated workspace and destination -> tool-driven extraction/loading -> transformations and feedback -> pipeline submission -> trusted execution verification -> reward -> cleanup.

Persist files, connection configuration, synchronization status, warehouse state, and interaction history within an episode. Isolate mutable resources across episodes. Terminate on submission or an explicit resource limit, and distinguish infrastructure failures from incorrect candidate pipelines.

## Responsibilities

| Component | Responsibility |
| --- | --- |
| Task loader | Read task descriptions, source schemas, target models, and public connection configuration. |
| ELT environment | Expose tools, return feedback, and manage episode state and termination. |
| Destination adapter | Handle warehouse-specific connections, namespaces, SQL execution, result retrieval, and cleanup. |
| Verifier | Read fresh warehouse results and apply trusted correctness checks. |
| Tinker recipe | Build isolated trajectory groups and reuse cookbook sampling, data processing, training, and checkpointing. |

Start with DuckDB for credential-free local tests and Snowflake for one official rollout. Support for further warehouses should fit the destination adapter without rewriting the environment.

## Reward and verification

Use execution-derived terminal rewards first. Define loading and transformation correctness explicitly, keep strict benchmark metrics separate from training reward shaping, and avoid counting the same terminal reward twice.

The agent must not access reference answers, modify the verifier, or submit fabricated evaluation files. Use private per-episode evaluation storage and query the warehouse afresh: the current upstream Stage 2 evaluator can reuse previously materialized candidate CSVs. Clean replay of the submitted pipeline is a proposed additional check for reproducibility.

## Implementation order

1. Inspect one official task and document its inputs, outputs, tools, and verification rules.
2. Implement and test the complete local episode lifecycle.
3. Connect the official source services, Airbyte, and Snowflake; run one official task.
4. Integrate the Tinker training recipe and complete one real execution-reward optimization step.
5. Harden isolation, replay, timeouts, cleanup, and documentation; prepare the submission design document.

Pin benchmark and cookbook revisions before implementing their interfaces. Record actual test and rollout results without treating local fixtures as official benchmark passes.
