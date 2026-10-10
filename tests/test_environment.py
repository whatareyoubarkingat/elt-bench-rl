from __future__ import annotations

import json
import time

from elt_bench_rl.destination import DuckDBAdapter
from elt_bench_rl.pipeline import PipelineBundle


def load_all(env):
    for source in env.task.sources:
        assert env.dispatch("configure_load", {"source": source.name})["ok"]
        assert env.dispatch("execute_load", {"source": source.name})["ok"]


def test_complete_repair_submit_replay(episode, correct_sql):
    assert episode.adapter.list_tables() == []
    assert episode.dispatch("inspect_source", {"source": "cars"})["sample"]
    assert episode.adapter.list_tables() == []
    load_all(episode)
    assert episode.dispatch(
        "write_model",
        {"name": "trains", "sql": correct_sql.replace("t.direction", "t.missing_col")},
    )["ok"]
    failed = episode.dispatch("run_dbt", {})
    assert not failed["ok"] and "missing_col" in failed["feedback"]
    assert episode.dispatch("write_model", {"name": "trains", "sql": correct_sql})["ok"]
    assert episode.dispatch("run_dbt", {})["ok"]
    result = episode.dispatch("submit_pipeline", {})
    assert result["reward"] == 1
    assert episode.result.replay == 1
    bundle_path = episode.work_dir / "pipeline.json"
    assert PipelineBundle.read(bundle_path).sha256 == episode.bundle.sha256
    assert episode.dispatch("execute_load", {"source": "train"})["should_stop"]
    bundle_text = bundle_path.read_text()
    assert str(episode.work_dir) not in bundle_text
    assert "locator" not in bundle_text and "password" not in bundle_text
    assert len(list(episode.work_dir.glob("replay_*/grade.json"))) == 1
    assert not list(episode.work_dir.glob("replay_*/*.duckdb"))


def test_wrong_pipeline_gets_lower_reward(episode, correct_sql):
    load_all(episode)
    wrong = correct_sql.replace("t.direction,", "'wrong' as direction,")
    assert episode.dispatch("write_model", {"name": "trains", "sql": wrong})["ok"]
    assert episode.dispatch("run_dbt", {})["ok"]
    result = episode.dispatch("submit_pipeline", {})
    assert 0.2 < result["reward"] < 0.7
    assert episode.result.replay == 0


def test_invalid_or_no_submission_zero(episode, correct_sql):
    assert episode.dispatch("write_model", {"name": "trains", "sql": correct_sql})["ok"]
    assert episode.dispatch("submit_pipeline", {})["reward"] == 0


def test_repeated_load_is_replace_and_isolation(episode, tmp_path):
    load_all(episode)
    assert episode.dispatch("execute_load", {"source": "cars"})["ok"]
    assert len(episode.adapter.table("cars")) == 5
    other = DuckDBAdapter(tmp_path / "other.duckdb").create()
    try:
        assert other.list_tables() == []
    finally:
        other.cleanup()


def test_timeout_zero(episode):
    episode.started = time.monotonic() - 1000
    result = episode.dispatch("read_task", {})
    assert result["should_stop"] and episode.result.reward == 0


def test_stale_csv_cannot_spoof_reward(episode, correct_sql):
    load_all(episode)
    wrong = correct_sql.replace("t.direction,", "'wrong' as direction,")
    episode.dispatch("write_model", {"name": "trains", "sql": wrong})
    episode.dispatch("run_dbt", {})
    stale = episode.work_dir / "evaluation/trains.csv"
    stale.parent.mkdir()
    reference = episode.reference_root / "trains/trains.csv"
    stale.write_bytes(reference.read_bytes())
    episode.dispatch("submit_pipeline", {})
    assert episode.result.replay == 0 and episode.result.target < 1
    grade = json.loads((episode.work_dir / "grade.json").read_text())
    assert grade["details"]["targets"]["trains"]


def test_broken_final_model_cannot_use_previous_target(episode, correct_sql):
    load_all(episode)
    episode.dispatch("write_model", {"name": "trains", "sql": correct_sql})
    assert episode.dispatch("run_dbt", {})["ok"]
    broken = correct_sql.replace("t.direction,", "cast('bad' as integer) as direction,")
    assert episode.dispatch("write_model", {"name": "trains", "sql": broken})["ok"]
    episode.dispatch("submit_pipeline", {})
    assert episode.result.details["current_target"] == 1
    assert episode.result.target == 0 and episode.result.reward == 0.2
    assert not episode.result.official_stage2


def test_constant_submission_cannot_earn_loading_only_reward(episode):
    load_all(episode)
    episode.dispatch("write_model", {"name": "trains", "sql": "select 1 as train_id"})
    episode.dispatch("run_dbt", {})
    assert episode.dispatch("submit_pipeline", {})["reward"] == 0


def test_literal_table_references_are_compiled_to_episode_sources(episode, correct_sql):
    load_all(episode)
    sql = correct_sql.replace("{{ source('raw', 'train') }}", "raw.train").replace(
        "{{ source('raw', 'cars') }}", "raw.cars"
    )
    episode.dispatch("write_model", {"name": "trains", "sql": sql})
    assert episode.dispatch("run_dbt", {})["ok"]
    generated = (episode.work_dir / "dbt/models/trains.sql").read_text()
    assert "{{ source('raw', 'train') }}" in generated
    assert "{{ source('raw', 'cars') }}" in generated
    assert episode.dispatch("submit_pipeline", {})["reward"] == 1
