"""Paid-work reservations survive retries and concurrent process boundaries."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from elt_bench_rl.budget import BudgetExceededError, TokenBudget


def budget(path, limit=2.0):
    return TokenBudget(path, limit_usd=limit, storage_buffer_usd=0, prior_cost_usd=0)


def test_settlement_refunds_unused_generation_budget(tmp_path):
    ledger = budget(tmp_path / "budget.json")
    ticket = ledger.reserve("sample", input_tokens=1_000_000, output_tokens=1_000_000)
    with pytest.raises(BudgetExceededError):
        ledger.reserve("train", input_tokens=1_000_000, output_tokens=0)
    ticket.complete(1_000_000, 10_000)
    ledger.reserve("train", input_tokens=1_000_000, output_tokens=0)
    snapshot = ledger.snapshot()
    assert snapshot["committed_and_reserved_usd"] == pytest.approx(1.07705)


def test_ambiguous_remote_call_is_counted_after_reopening(tmp_path):
    path = tmp_path / "budget.json"
    ledger = budget(path)
    ticket = ledger.reserve("sample", input_tokens=1_000_000, output_tokens=1_000_000)
    ticket.uncertain("TimeoutError")
    reopened = budget(path)
    with pytest.raises(BudgetExceededError):
        reopened.reserve("train", input_tokens=1_000_000, output_tokens=0)
    assert reopened.snapshot()["reservations"][0]["state"] == "uncertain"


def test_process_crash_keeps_unsettled_reservation(tmp_path):
    path = tmp_path / "budget.json"
    budget(path).reserve("train", input_tokens=2_000_000, output_tokens=0)
    with pytest.raises(BudgetExceededError):
        budget(path).reserve("train", input_tokens=1_000_000, output_tokens=0)


def test_concurrent_reservations_cannot_overspend(tmp_path):
    path = tmp_path / "budget.json"
    budget(path, limit=1.0)

    def reserve():
        try:
            budget(path, limit=1.0).reserve(
                "train", input_tokens=1_000_000, output_tokens=0
            )
        except BudgetExceededError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(lambda _: reserve(), range(4)))
    assert sum(accepted) == 1
    assert budget(path, limit=1.0).snapshot()["committed_and_reserved_usd"] == 0.737


def test_existing_cap_and_malformed_ledger_fail_closed(tmp_path):
    path = tmp_path / "budget.json"
    budget(path)
    with pytest.raises(ValueError, match="cannot change"):
        budget(path, limit=20)
    path.write_text('{"schema_version":1}')
    with pytest.raises(ValueError):
        budget(path)


def test_unknown_model_or_token_counts_cannot_authorize_calls(tmp_path):
    with pytest.raises(ValueError, match="verified model"):
        TokenBudget(tmp_path / "unknown.json", model_name="unknown")
    ledger = budget(tmp_path / "budget.json")
    for count in (-1, True, 1.5):
        with pytest.raises(ValueError):
            ledger.reserve("sample", input_tokens=count, output_tokens=0)
