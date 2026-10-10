"""A complete real ELT episode through Cookbook, without a model service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_cookbook_env import ToyRenderer, render_tool_call
from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs

from elt_bench_rl import training
from elt_bench_rl.training import TrainingConfig, rollout


class ScriptedPipelineCompleter(TokenCompleter):
    def __init__(self, renderer, calls):
        self.renderer = renderer
        self.calls = list(calls)
        self.index = 0

    async def __call__(self, model_input, stop, *, max_tokens=None):
        name, arguments = self.calls[self.index]
        self.index += 1
        tokens = render_tool_call(self.renderer, name, arguments)
        assert max_tokens is None or len(tokens) <= max_tokens
        return TokensWithLogprobs(tokens, [-1.0] * len(tokens), "stop")


@pytest.mark.asyncio
async def test_real_load_repair_dbt_submit_replay_through_cookbook(
    episode, correct_sql, tmp_path, monkeypatch
):
    def forbid_service(*args, **kwargs):
        raise AssertionError("This integration test must never create a Tinker client")

    monkeypatch.setattr(training.tinker, "ServiceClient", forbid_service)
    credential_canary = "unit-test-private-tinker-credential-canary"
    verifier_canary = "unit-test-trusted-verifier-details-canary"
    monkeypatch.setenv("TINKER_API_KEY", credential_canary)
    save_grade = episode._save_grade

    def save_private_grade():
        # Real grading runs unchanged. Only its private diagnostics gain a
        # canary, verifying that details never become a model observation.
        episode.result.details["private_canary"] = verifier_canary
        save_grade()

    monkeypatch.setattr(episode, "_save_grade", save_private_grade)
    config = TrainingConfig(
        {"task_id": "trains", "destination": "duckdb"},
        str(tmp_path / "cookbook-rollout"),
        str(tmp_path / "deliberately-absent-tokenizer"),
        group_size=1,
    )
    renderer = ToyRenderer()
    calls = [("read_task", {}), ("inspect_source", {"source": "cars"})]
    for source in episode.task.sources:
        calls.extend(
            [
                ("configure_load", {"source": source.name}),
                ("execute_load", {"source": source.name}),
            ]
        )
    calls.extend(
        [
            (
                "write_model",
                {
                    "name": "trains",
                    "sql": correct_sql.replace("t.direction", "t.missing_col"),
                },
            ),
            ("run_dbt", {}),
            ("write_model", {"name": "trains", "sql": correct_sql}),
            ("run_dbt", {}),
            ("submit_pipeline", {}),
        ]
    )
    completer = ScriptedPipelineCompleter(renderer, calls)
    result = await rollout(
        config,
        renderer=renderer,
        completer=completer,
        episode_factory=lambda _config: episode,
    )
    assert result.rewards == [1.0]
    assert result.stop_reasons == ["tool_stopped"]
    assert result.metrics[0]["loading"] == result.metrics[0]["target"] == 1.0
    assert result.metrics[0]["replay"] == 1.0
    assert episode.submitted and episode.closed
    assert completer.index == len(calls)
    assert {receipt.loader for receipt in episode.receipts.values()} == {
        "local_csv_http"
    }
    assert (episode.work_dir / "pipeline.json").is_file()
    assert list(episode.work_dir.glob("replay_*/grade.json"))
    run_results = [
        entry["result"] for entry in episode.trace if entry["tool"] == "run_dbt"
    ]
    assert len(run_results) == 2 and not run_results[0]["ok"] and run_results[1]["ok"]
    transcript = Path(result.trace_path).read_text()
    assert "missing_col" in transcript  # execution feedback reached the agent
    assert verifier_canary in (episode.work_dir / "grade.json").read_text()
    for private in (verifier_canary, credential_canary, str(episode.reference_root)):
        assert private not in transcript
    terminal_feedback = next(
        entry["result"] for entry in episode.trace if entry["tool"] == "submit_pipeline"
    )
    assert "details" not in terminal_feedback
    assert json.loads(transcript)["rewards"] == [1.0]
