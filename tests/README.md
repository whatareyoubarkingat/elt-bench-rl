# Tests

`uv run --locked pytest -q tests` is credential-free and requires no Docker.
Fixtures contain four synthetic trains with a separate trusted target file.
Tests start a real HTTP server and execute real DuckDB/dbt pipelines.

Coverage: feedback/repair, replay, correct vs wrong rewards, replacement loading,
contamination, budgets, key/value/null comparison, duplicate keys/columns,
missing/extra rows/columns, stale CSV spoofing, path/schema restrictions,
Jinja/shell/extension denial, and Cookbook masks/advantages/optimizer/checkpoints.
Cloud/GPU clients are mocked where explicitly labeled.

Cloud acceptance is separate (`docs/running.md`). Offline tests never prove a
Databricks sync or real Tinker parameter update.
