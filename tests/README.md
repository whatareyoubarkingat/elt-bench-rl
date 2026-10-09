# Acceptance tests to implement

1. **Local integration:** execute extraction, loading, transformation, feedback, submission, and grading with a small deterministic fixture and DuckDB. Include an incorrect pipeline and verify episode isolation.
2. **Official rollout:** run at least one official ELT-Bench task against the supported warehouse and retain the submitted pipeline, tool trace, and execution results.
3. **Tinker optimization:** collect model-generated trajectories, derive rewards from execution, and retain evidence of a real optimizer step. Report zero group reward variance explicitly because an optimizer call alone does not demonstrate an effective RL learning signal.

These are planned tests, not implemented or passing tests.
