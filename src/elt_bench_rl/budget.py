"""Conservative, persistent token-cost reservations for authorized experiments.

This is an uncached rate-card estimate, not an invoice or account balance.
Reservations survive process failure so an uncertain remote call is never free.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar


class BudgetExceededError(RuntimeError):
    """A request would exceed the experiment's conservative dollar limit."""


class TokenBudget:
    """Share one locked ledger across rollout, training and recovery commands."""

    MODEL = "Qwen/Qwen3.5-4B"
    RATES: ClassVar[dict[str, float]] = {"input": 0.33, "output": 1.005, "train": 0.737}

    def __init__(
        self,
        path: Path | str,
        *,
        limit_usd: float = 10.0,
        storage_buffer_usd: float = 0.5,
        prior_cost_usd: float = 0.000008955,
        model_name: str = MODEL,
    ) -> None:
        for name, value in (
            ("limit_usd", limit_usd),
            ("storage_buffer_usd", storage_buffer_usd),
            ("prior_cost_usd", prior_cost_usd),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if limit_usd <= storage_buffer_usd + prior_cost_usd:
            raise ValueError("Budget must exceed its storage buffer and prior cost")
        if model_name != self.MODEL:
            raise ValueError("A verified model rate card is required for this budget")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.limit_usd = float(limit_usd)
        self.storage_buffer_usd = float(storage_buffer_usd)
        self.prior_cost_usd = float(prior_cost_usd)
        with self._locked() as ledger:
            if not ledger:
                ledger.update(
                    schema_version=1,
                    model_name=model_name,
                    limit_usd=self.limit_usd,
                    storage_buffer_usd=self.storage_buffer_usd,
                    prior_cost_usd=self.prior_cost_usd,
                    rates_per_million_tokens=self.RATES,
                    price_source="https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/",
                    estimate_not_invoice=True,
                    reservations=[],
                )
            self._validate(ledger)

    @contextmanager
    def _locked(self):
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(fd, "r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                ledger = json.loads(self.path.read_text()) if self.path.exists() else {}
                yield ledger
                ledger["updated_at_utc"] = datetime.now(UTC).isoformat()
                if ledger:
                    ledger["committed_and_reserved_usd"] = self._total(ledger)
                    ledger["remaining_usd"] = ledger["limit_usd"] - self._total(ledger)
                fd, temporary = tempfile.mkstemp(
                    prefix=self.path.name + ".", dir=self.path.parent
                )
                try:
                    with os.fdopen(fd, "w") as output:
                        json.dump(ledger, output, allow_nan=False, indent=2)
                        output.write("\n")
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, self.path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _validate(self, ledger: dict[str, Any]) -> None:
        expected = {
            "schema_version": 1,
            "model_name": self.MODEL,
            "limit_usd": self.limit_usd,
            "storage_buffer_usd": self.storage_buffer_usd,
            "prior_cost_usd": self.prior_cost_usd,
            "rates_per_million_tokens": self.RATES,
        }
        if any(ledger.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Existing budget settings cannot change during an experiment"
            )
        if not isinstance(ledger.get("reservations"), list):
            raise TypeError("Malformed experiment budget ledger")
        for entry in ledger["reservations"]:
            if entry.get("state") not in {"reserved", "completed", "uncertain"}:
                raise ValueError("Malformed budget reservation state")
            for name in ("reserved_usd", "accounted_usd"):
                value = entry.get(name)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("Malformed budget reservation cost")

    @staticmethod
    def _total(ledger: dict[str, Any]) -> float:
        return (
            ledger["storage_buffer_usd"]
            + ledger["prior_cost_usd"]
            + sum(entry["accounted_usd"] for entry in ledger["reservations"])
        )

    @classmethod
    def _cost(cls, operation: str, input_tokens: int, output_tokens: int) -> float:
        if operation not in {"sample", "probe", "train"}:
            raise ValueError("Unsupported budget operation")
        if any(
            type(value) is not int or value < 0
            for value in (input_tokens, output_tokens)
        ):
            raise ValueError("Token counts must be nonnegative integers")
        if operation == "train":
            if output_tokens:
                raise ValueError(
                    "Training reservations contain input sequence tokens only"
                )
            return input_tokens * cls.RATES["train"] / 1_000_000
        return (
            input_tokens * cls.RATES["input"] + output_tokens * cls.RATES["output"]
        ) / 1_000_000

    def reserve(
        self,
        operation: str,
        *,
        input_tokens: int,
        output_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> BudgetReservation:
        cost = self._cost(operation, input_tokens, output_tokens)
        ticket = uuid.uuid4().hex
        with self._locked() as ledger:
            self._validate(ledger)
            if self._total(ledger) + cost > self.limit_usd + 1e-12:
                raise BudgetExceededError(
                    f"Request estimate ${cost:.6f} exceeds remaining experiment budget "
                    f"${self.limit_usd - self._total(ledger):.6f}"
                )
            ledger["reservations"].append(
                {
                    "id": ticket,
                    "operation": operation,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reserved_usd": cost,
                    "accounted_usd": cost,
                    "state": "reserved",
                    "metadata": metadata or {},
                    "created_at_utc": datetime.now(UTC).isoformat(),
                }
            )
        return BudgetReservation(self, ticket)

    def snapshot(self) -> dict[str, Any]:
        with self._locked() as ledger:
            self._validate(ledger)
            return json.loads(json.dumps(ledger))

    def _settle(self, ticket: str, *, counts=None, error_type=None) -> None:
        with self._locked() as ledger:
            self._validate(ledger)
            entry = next(
                item for item in ledger["reservations"] if item["id"] == ticket
            )
            if entry["state"] != "reserved":
                raise ValueError("Budget reservation was already finalized")
            if counts is not None:
                cost = self._cost(entry["operation"], *counts)
                if (
                    counts[0] > entry["input_tokens"]
                    or counts[1] > entry["output_tokens"]
                ):
                    raise ValueError(
                        "Actual token counts exceed the authorized reservation"
                    )
                entry.update(
                    state="completed",
                    actual_input_tokens=counts[0],
                    actual_output_tokens=counts[1],
                    accounted_usd=cost,
                )
            else:
                entry.update(state="uncertain", error_type=str(error_type))
            entry["finished_at_utc"] = datetime.now(UTC).isoformat()


class BudgetReservation:
    def __init__(self, budget: TokenBudget, ticket: str) -> None:
        self.budget = budget
        self.ticket = ticket

    def complete(self, input_tokens: int, output_tokens: int) -> None:
        self.budget._settle(self.ticket, counts=(input_tokens, output_tokens))

    def uncertain(self, error_type: str) -> None:
        self.budget._settle(self.ticket, error_type=error_type)
