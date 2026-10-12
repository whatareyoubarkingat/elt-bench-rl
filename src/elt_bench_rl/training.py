"""Small execution-reward RL recipe using the pinned Tinker Cookbook.

Client creation is opt-in. Injected clients/completers support genuinely offline
tests; they must never be presented as evidence of a credentialed Tinker run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
import tinker
from tinker_cookbook.checkpoint_utils import save_checkpoint_async
from tinker_cookbook.completers import (
    StopCondition,
    TinkerTokenCompleter,
    TokenCompleter,
    TokensWithLogprobs,
)
from tinker_cookbook.model_info import (
    get_recommended_renderer_name,
    get_recommended_renderer_names,
)
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.train import train_step
from tinker_cookbook.rl.types import PARSE_ERROR_MASKED_METRIC_KEY, TrajectoryGroup

from .cookbook_env import (
    BoundedEpisodeStrategy,
    ELTDatasetBuilder,
    ELTEnvGroupBuilder,
    EpisodeFactory,
)

T = TypeVar("T")


class BudgetReservation(Protocol):
    """Caller-owned reservation; uncertain remote outcomes retain their cost."""

    def complete(self, input_tokens: int, output_tokens: int) -> None: ...

    def uncertain(self, error_type: str) -> None: ...


class TokenBudget(Protocol):
    """One persistent spending ledger shared across rollout, training and retries."""

    def reserve(
        self,
        operation: str,
        *,
        input_tokens: int,
        output_tokens: int,
        metadata: dict[str, Any],
    ) -> BudgetReservation: ...


class PaidAccessRequired(RuntimeError):
    """A real Tinker client would be created without explicit authorization."""


class NoLearningSignalError(RuntimeError):
    """Execution rewards failed to produce nonzero effective advantages."""


class NoWeightUpdateError(RuntimeError):
    """The requested step lacks observable pre/post model-change evidence."""


class RemoteCallTimeout(TimeoutError):
    """A bounded wait expired; remote execution may still have consumed tokens."""


@dataclass(frozen=True)
class TrainingConfig:
    episode_config: dict[str, Any]
    log_path: str
    tokenizer_path: str
    model_name: str = "Qwen/Qwen3.5-4B"
    group_size: int = 4
    lora_rank: int = 16
    learning_rate: float = 1e-5
    temperature: float = 1.0
    max_turns: int = 24
    max_sampled_tokens: int = 8192
    max_turn_tokens: int = 2048
    max_context_tokens: int = 32768
    max_tool_output_tokens: int = 2048
    max_steps: int = 1
    serial_episodes: int = 1
    max_constant_group_retries: int = 2
    update_tolerance: float = 1e-8
    load_checkpoint_path: str | None = None
    sampler_checkpoint_path: str | None = None
    remote_timeout_seconds: float = 300.0
    renderer_name: str | None = None

    def validate(self, *, training: bool) -> None:
        for name in (
            "group_size",
            "lora_rank",
            "max_turns",
            "max_sampled_tokens",
            "max_turn_tokens",
            "max_context_tokens",
            "max_tool_output_tokens",
            "max_steps",
            "serial_episodes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if training and self.group_size < 2:
            raise ValueError(
                "group_size >= 2 is required for reward-minus-mean advantages"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.max_constant_group_retries < 0 or self.max_constant_group_retries > 2:
            raise ValueError("max_constant_group_retries must be between 0 and 2")
        if not math.isfinite(self.update_tolerance) or self.update_tolerance < 0:
            raise ValueError("update_tolerance must be finite and nonnegative")
        if (
            not math.isfinite(self.remote_timeout_seconds)
            or self.remote_timeout_seconds <= 0
        ):
            raise ValueError("remote_timeout_seconds must be finite and positive")
        if self.renderer_name is not None and (
            not isinstance(self.renderer_name, str)
            or self.renderer_name not in get_recommended_renderer_names(self.model_name)
        ):
            raise ValueError(
                "renderer_name must be a supported renderer for model_name"
            )
        if self.load_checkpoint_path is not None and (
            not isinstance(self.load_checkpoint_path, str)
            or not self.load_checkpoint_path.startswith("tinker://")
        ):
            raise ValueError("load_checkpoint_path must be a Tinker state URI")
        if self.sampler_checkpoint_path is not None and (
            not isinstance(self.sampler_checkpoint_path, str)
            or not self.sampler_checkpoint_path.startswith("tinker://")
        ):
            raise ValueError("sampler_checkpoint_path must be a Tinker sampler URI")
        if training and self.sampler_checkpoint_path is not None:
            raise ValueError(
                "Training resumes from load_checkpoint_path, not sampler_checkpoint_path"
            )
        if not training and self.load_checkpoint_path is not None:
            raise ValueError(
                "Rollout uses sampler_checkpoint_path, not an optimizer state checkpoint"
            )


# One budget schema for standalone rollout and optimization.
RolloutConfig = TrainingConfig


@dataclass(frozen=True)
class RolloutResult:
    rewards: list[float]
    stop_reasons: list[str | None]
    metrics: list[dict[str, float]]
    trace_path: str
    injected_runtime: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    steps: list[dict[str, Any]]
    state_path: str
    sampler_path: str
    effective_update: bool
    injected_runtime: bool
    evidence_path: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config(config: TrainingConfig | Mapping[str, Any]) -> TrainingConfig:
    return (
        config if isinstance(config, TrainingConfig) else TrainingConfig(**dict(config))
    )


def load_renderer(config: TrainingConfig) -> Renderer:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    return get_renderer(_renderer_name(config), tokenizer)


def _renderer_name(config: TrainingConfig) -> str:
    return config.renderer_name or get_recommended_renderer_name(config.model_name)


def _group_builder(
    config: TrainingConfig,
    renderer: Renderer,
    episode_factory: EpisodeFactory | None,
    progress: DurableProgress | None = None,
) -> ELTEnvGroupBuilder:
    builder = ELTEnvGroupBuilder(
        dict(config.episode_config),
        renderer,
        group_size=config.group_size,
        max_turns=config.max_turns,
        max_sampled_tokens=config.max_sampled_tokens,
        max_turn_tokens=config.max_turn_tokens,
        max_context_tokens=config.max_context_tokens,
        max_tool_output_tokens=config.max_tool_output_tokens,
        episode_factory=episode_factory,
    )
    builder.progress = progress
    return builder


def _write_json(path: Path, value: Any) -> None:
    body = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".evidence-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _append_json(path: Path, value: Any) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class DurableProgress:
    """Flush public turn/phase evidence immediately, even if a later call fails."""

    def __init__(self, log_path: Path, *, injected_runtime: bool):
        self.path = log_path / "progress.jsonl"
        self.context: dict[str, Any] = {"injected_runtime": injected_runtime}
        self.last_phase = "initializing"
        self._lock = threading.Lock()
        self._sequence = 0

    def __call__(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            if "phase" in event:
                self.last_phase = event["phase"]
            _append_json(
                self.path,
                {
                    **self.context,
                    **event,
                    "sequence": self._sequence,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )


def _reserve(
    budget: TokenBudget | None,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    progress: DurableProgress,
) -> BudgetReservation | None:
    if budget is None:
        return None
    try:
        return budget.reserve(
            operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=dict(progress.context),
        )
    except Exception as error:
        progress({"event": "budget_denied", "error_type": type(error).__name__})
        raise


async def _remote(
    phase: str,
    call: Callable[[], Awaitable[T]],
    config: TrainingConfig,
    progress: DurableProgress,
    *,
    reservation: BudgetReservation | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> T:
    progress({"event": "remote_start", "phase": phase})
    started = time.monotonic()
    try:
        async with asyncio.timeout(config.remote_timeout_seconds):
            result = await call()
    except BaseException as error:
        if reservation is not None:
            reservation.uncertain(type(error).__name__)
        progress(
            {
                "event": "remote_error",
                "phase": phase,
                "error_type": type(error).__name__,
                "remote_outcome": "uncertain",
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if isinstance(error, TimeoutError):
            raise RemoteCallTimeout(
                f"Remote phase exceeded its wait limit: {phase}"
            ) from None
        raise
    if reservation is not None:
        reservation.complete(input_tokens, output_tokens)
    progress(
        {
            "event": "remote_end",
            "phase": phase,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return result


class ObservedCompleter(TokenCompleter):
    """Reserve a capped call and flush its exact sampled action before tools run."""

    def __init__(self, inner, config, renderer, progress, budget=None):
        self.inner, self.config, self.renderer = inner, config, renderer
        self.progress, self.budget = progress, budget
        self.calls = 0

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition, *, max_tokens=None
    ) -> TokensWithLogprobs:
        cap = min(
            self.config.max_turn_tokens,
            max_tokens if max_tokens is not None else self.config.max_turn_tokens,
            self.config.max_context_tokens - model_input.length,
        )
        if cap <= 0:
            raise ValueError("Sampling prompt leaves no room in the context budget")
        self.calls += 1
        fields = {
            "phase": "sample",
            "call": self.calls,
            "prompt_tokens": model_input.length,
            "output_cap": cap,
        }
        reservation = _reserve(
            self.budget, "sample", model_input.length, cap, self.progress
        )
        self.progress({"event": "sample_start", **fields})
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.config.remote_timeout_seconds):
                action = await self.inner(model_input, stop, max_tokens=cap)
        except BaseException as error:
            if reservation is not None:
                reservation.uncertain(type(error).__name__)
            self.progress(
                {
                    "event": "sample_error",
                    **fields,
                    "error_type": type(error).__name__,
                    "remote_outcome": "uncertain",
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            if isinstance(error, TimeoutError):
                raise RemoteCallTimeout(
                    "Sampling exceeded its remote wait limit"
                ) from None
            raise
        if reservation is not None:
            reservation.complete(model_input.length, len(action.tokens))
        logprobs = action.logprobs
        if (
            len(action.tokens) > cap
            or len(logprobs) != len(action.tokens)
            or not all(math.isfinite(value) for value in logprobs)
        ):
            raise ValueError("Sample violated its token/logprob contract")
        self.progress(
            {
                "event": "sample_end",
                **fields,
                "action_tokens": action.tokens,
                "action": self.renderer.tokenizer.decode(action.tokens),
                "sampled_logprobs": logprobs,
                "stop_reason": action.stop_reason,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        return action


def _sampler_retry(config: TrainingConfig):
    from tinker.lib.retry_handler import RetryConfig

    return RetryConfig(
        enable_retry_logic=False, progress_timeout=config.remote_timeout_seconds
    )


async def _close_service(service, status, config, progress):
    if service is None:
        return

    async def close():
        # close() can wait for the lazy holder's lock after a creation timeout.
        # Keep that synchronous section off the progress event loop as well.
        future = await asyncio.to_thread(service.close, status)
        await future

    try:
        await _remote(
            "session_close",
            close,
            replace(
                config, remote_timeout_seconds=min(config.remote_timeout_seconds, 30)
            ),
            progress,
        )
    except Exception as error:
        # Preserve the original outcome; this failure is also durable evidence.
        progress({"event": "session_close_failed", "error_type": type(error).__name__})
        if status == "success":
            _failure(progress.path.parent, progress, error)
            raise


def _failure(log_path, progress, error):
    _write_json(
        log_path / "failure.json",
        {
            "status": "failed",
            "error_type": type(error).__name__,
            "last_phase": progress.last_phase,
            "injected_runtime": progress.context["injected_runtime"],
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


async def _create_sampler(owner, config, progress, *, model_path=None):
    kwargs = {"retry_config": _sampler_retry(config)}
    if model_path is not None:
        kwargs["model_path"] = model_path
    else:
        kwargs["base_model"] = config.model_name
    sync = getattr(owner, "create_sampling_client", None)
    # A new ServiceClient lazily initializes its holder synchronously. Keep
    # that public sync operation off the monitor's event loop.
    call = (
        (lambda: asyncio.to_thread(sync, **kwargs))
        if callable(sync)
        else (lambda: owner.create_sampling_client_async(**kwargs))
    )
    return await _remote("sampler_create", call, config, progress)


class _ObservedFuture:
    def __init__(self, inner, phase, config, progress, reservation=None, tokens=0):
        self.inner, self.phase, self.config, self.progress = (
            inner,
            phase,
            config,
            progress,
        )
        self.reservation, self.tokens = reservation, tokens
        self.settled = False

    async def result_async(self):
        try:
            result = await _remote(
                self.phase,
                self.inner.result_async,
                self.config,
                self.progress,
                reservation=self.reservation,
                input_tokens=self.tokens,
            )
            if self.phase == "optimizer_wait":
                self.progress(
                    {
                        "event": "optimizer_response",
                        "response": {
                            "metrics": getattr(result, "metrics", None),
                            "type": getattr(result, "type", None),
                        },
                    }
                )
            return result
        finally:
            self.settled = True

    def uncertain(self, error_type):
        if not self.settled and self.reservation is not None:
            self.reservation.uncertain(error_type)
            self.settled = True


class _ObservedTrainingClient:
    """Retain Cookbook pipelining while bounding submissions and future waits."""

    def __init__(self, inner, config, progress, budget):
        self.inner, self.config, self.progress, self.budget = (
            inner,
            config,
            progress,
            budget,
        )
        self.pending: list[_ObservedFuture] = []

    async def forward_backward_async(self, data, *, loss_fn, loss_fn_config=None):
        tokens = sum(datum.model_input.length for datum in data)
        reservation = _reserve(self.budget, "train", tokens, 0, self.progress)
        try:
            future = await _remote(
                "forward_backward_submit",
                lambda: self.inner.forward_backward_async(
                    data, loss_fn=loss_fn, loss_fn_config=loss_fn_config
                ),
                self.config,
                self.progress,
            )
        except BaseException as error:
            if reservation is not None:
                reservation.uncertain(type(error).__name__)
            raise
        observed = _ObservedFuture(
            future,
            "forward_backward_wait",
            self.config,
            self.progress,
            reservation,
            tokens,
        )
        self.pending.append(observed)
        return observed

    async def optim_step_async(self, params):
        future = await _remote(
            "optimizer_submit",
            lambda: self.inner.optim_step_async(params),
            self.config,
            self.progress,
        )
        return _ObservedFuture(future, "optimizer_wait", self.config, self.progress)

    def uncertain(self, error_type):
        for future in self.pending:
            future.uncertain(error_type)


def _trace(
    group: TrajectoryGroup, renderer: Renderer, *, step: int, attempt: int
) -> dict[str, Any]:
    return {
        "step": step,
        "attempt": attempt,
        "rewards": group.get_total_rewards(),
        "trajectories": [
            {
                "stop_reason": trajectory.stop_reason,
                "final_reward": final_reward,
                "transitions": [
                    {
                        "observation": renderer.tokenizer.decode(
                            transition.ob.to_ints()
                        ),
                        "observation_tokens": transition.ob.to_ints(),
                        "action": renderer.tokenizer.decode(transition.ac.tokens),
                        "action_tokens": transition.ac.tokens,
                        "sampled_logprobs": transition.ac.logprobs,
                        "reward": transition.reward,
                        "metrics": transition.metrics,
                    }
                    for transition in trajectory.transitions
                ],
            }
            for trajectory, final_reward in zip(
                group.trajectories_G, group.final_rewards_G, strict=True
            )
        ],
    }


def _effective_data_stats(data: list[tinker.Datum]) -> dict[str, float]:
    effective_count = 0
    absolute_advantage = 0.0
    for datum in data:
        advantage = datum.loss_fn_inputs["advantages"].to_numpy()
        mask = datum.loss_fn_inputs["mask"].to_numpy()
        logprobs = datum.loss_fn_inputs["logprobs"].to_numpy()
        if not (
            np.isfinite(advantage).all()
            and np.isfinite(logprobs).all()
            and np.isfinite(mask).all()
        ):
            raise ValueError(
                "Training data contains non-finite advantages, masks or logprobs"
            )
        effective = (mask > 0) & (advantage != 0)
        effective_count += int(effective.sum())
        absolute_advantage += float(np.abs(advantage[effective]).sum())
    return {
        "effective_action_tokens": float(effective_count),
        "absolute_effective_advantage": absolute_advantage,
    }


def _probe(data: list[tinker.Datum]) -> tuple[tinker.ModelInput, np.ndarray]:
    for datum in data:
        advantage = datum.loss_fn_inputs["advantages"].to_numpy()
        mask = datum.loss_fn_inputs["mask"].to_numpy()
        effective = (mask > 0) & (advantage != 0)
        if effective.any():
            targets = datum.loss_fn_inputs["target_tokens"].to_numpy()
            # The right-shifted model input omits the final target token.
            full_sequence = datum.model_input.append_int(int(targets[-1]))
            return full_sequence, effective
    raise NoLearningSignalError("No unmasked action token has a nonzero advantage")


async def _probe_logprobs(
    sampling_client: tinker.SamplingClient,
    prompt: tinker.ModelInput,
    mask: np.ndarray,
    config: TrainingConfig,
    progress: DurableProgress,
    budget: TokenBudget | None,
) -> np.ndarray:
    reservation = _reserve(budget, "probe", prompt.length, 1, progress)
    values = await _remote(
        "logprob_probe",
        lambda: sampling_client.compute_logprobs_async(prompt),
        config,
        progress,
        reservation=reservation,
        input_tokens=prompt.length,
        output_tokens=1,
    )
    if len(values) != len(mask) + 1:
        raise NoWeightUpdateError(
            "Probe returned an unexpected number of token logprobs"
        )
    selected = [value for value, keep in zip(values[1:], mask, strict=True) if keep]
    if not selected or any(
        value is None or not math.isfinite(value) for value in selected
    ):
        raise NoWeightUpdateError("Probe did not return finite action-token logprobs")
    return np.asarray(selected, dtype=np.float64)


async def _load_state_checkpoint(
    training_client: tinker.TrainingClient, path: str
) -> dict[str, str | None]:
    future = await training_client.load_state_with_optimizer_async(path)
    response = await future.result_async()
    # Keep only the documented response identity fields; loading succeeding
    # proves server-side readability beyond a save response/path string.
    return {
        name: getattr(response, name, None) for name in ("path", "model_id", "type")
    }


async def _fresh_state_checkpoint(
    service, config, progress, path, *, previous_model_id=None
):
    """Restore optimizer state into a new model, as required by LoadWeights.

    The server permits legacy load only immediately after model creation, at
    seq_id 1. Loading back into a client that has trained/saved is rejected.
    The public create-from-state method handles both server creation protocols.
    """
    if service is None:
        raise ValueError("Injected training requires an explicit checkpoint_service")
    sync = getattr(service, "create_training_client_from_state_with_optimizer", None)
    kwargs = {"base_model": config.model_name}
    call = (
        (lambda: asyncio.to_thread(sync, path, **kwargs))
        if callable(sync)
        else (
            lambda: service.create_training_client_from_state_with_optimizer_async(
                path, **kwargs
            )
        )
    )
    loaded = await _remote("state_load_verify", call, config, progress)
    model_id = getattr(loaded, "model_id", None)
    if not model_id or model_id == previous_model_id:
        raise NoWeightUpdateError(
            "Checkpoint verification did not return a new loaded model"
        )
    return {
        "path": path,
        "model_id": str(model_id),
        "type": "fresh_client_optimizer_restore",
        "optimizer_restored": True,
    }


async def rollout(
    config: TrainingConfig | Mapping[str, Any],
    *,
    allow_paid: bool = False,
    completer: TokenCompleter | None = None,
    renderer: Renderer | None = None,
    episode_factory: EpisodeFactory | None = None,
    budget: TokenBudget | None = None,
) -> RolloutResult:
    config = _config(config)
    config.validate(training=False)
    injected = completer is not None
    if completer is None and not allow_paid:
        raise PaidAccessRequired("Real Tinker rollout requires allow_paid=True")
    renderer = renderer or load_renderer(config)
    log_path = Path(config.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    progress = DurableProgress(log_path, injected_runtime=injected)
    progress.context.update(step=0, attempt=0, renderer_name=_renderer_name(config))
    service = None
    status = "errored"
    try:
        progress({"event": "rollout_start"})
        if completer is None:
            service = tinker.ServiceClient()
            sampling_client = await _create_sampler(
                service, config, progress, model_path=config.sampler_checkpoint_path
            )
            completer = TinkerTokenCompleter(
                sampling_client,
                max_tokens=config.max_turn_tokens,
                temperature=config.temperature,
                context_window=config.max_context_tokens,
            )
        observed = ObservedCompleter(completer, config, renderer, progress, budget)
        group = await do_group_rollout(
            _group_builder(config, renderer, episode_factory, progress),
            observed,
            strategy=BoundedEpisodeStrategy(config.serial_episodes),
        )
        trace_path = log_path / "rollout.json"
        _write_json(trace_path, _trace(group, renderer, step=0, attempt=0))
        metrics = [
            dict(trajectory.transitions[-1].metrics) if trajectory.transitions else {}
            for trajectory in group.trajectories_G
        ]
        result = RolloutResult(
            group.get_total_rewards(),
            [trajectory.stop_reason for trajectory in group.trajectories_G],
            metrics,
            str(trace_path),
            injected,
        )
        _write_json(
            log_path / "rollout_result.json",
            result.as_dict() | {"renderer_name": _renderer_name(config)},
        )
        progress({"event": "rollout_end", "rewards": result.rewards})
        status = "success"
        return result
    except BaseException as error:
        if isinstance(error, asyncio.CancelledError):
            status = "interrupted"
        _failure(log_path, progress, error)
        raise
    finally:
        await _close_service(service, status, config, progress)


async def train(
    config: TrainingConfig | Mapping[str, Any],
    *,
    allow_paid: bool = False,
    training_client: tinker.TrainingClient | None = None,
    renderer: Renderer | None = None,
    episode_factory: EpisodeFactory | None = None,
    completer_factory: Callable[[tinker.SamplingClient], TokenCompleter] | None = None,
    budget: TokenBudget | None = None,
    checkpoint_service: tinker.ServiceClient | None = None,
) -> TrainingResult:
    config = _config(config)
    config.validate(training=True)
    injected = training_client is not None or checkpoint_service is not None
    if training_client is None and checkpoint_service is not None:
        raise ValueError(
            "checkpoint_service is an injection seam for offline training clients"
        )
    if training_client is None and not allow_paid:
        raise PaidAccessRequired("Real Tinker optimization requires allow_paid=True")
    renderer = renderer or load_renderer(config)
    log_path = Path(config.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    progress = DurableProgress(log_path, injected_runtime=injected)
    progress.context.update(renderer_name=_renderer_name(config))
    service = None
    status = "errored"
    try:
        progress({"event": "training_start"})
        if training_client is None:
            service = tinker.ServiceClient()
        result = await _train_impl(
            config,
            training_client,
            renderer,
            episode_factory,
            completer_factory,
            budget,
            progress,
            service,
            injected,
            checkpoint_service,
        )
        progress({"event": "training_end", "effective_update": result.effective_update})
        status = "success"
        return result
    except BaseException as error:
        if isinstance(error, asyncio.CancelledError):
            status = "interrupted"
        _failure(log_path, progress, error)
        raise
    finally:
        await _close_service(service, status, config, progress)


async def _train_impl(
    config,
    training_client,
    renderer,
    episode_factory,
    completer_factory,
    budget,
    progress,
    service,
    injected,
    checkpoint_service,
):
    dataset = ELTDatasetBuilder(
        episode_configs=[dict(config.episode_config)],
        model_name=config.model_name,
        tokenizer_path=config.tokenizer_path,
        group_size=config.group_size,
        max_steps=config.max_steps,
        max_turns=config.max_turns,
        max_sampled_tokens=config.max_sampled_tokens,
        max_turn_tokens=config.max_turn_tokens,
        max_context_tokens=config.max_context_tokens,
        max_tool_output_tokens=config.max_tool_output_tokens,
    ).build_with_renderer(renderer)
    if training_client is None:
        if config.load_checkpoint_path is not None:
            sync = getattr(
                service, "create_training_client_from_state_with_optimizer", None
            )
            kwargs = {"base_model": config.model_name}
            call = (
                (lambda: asyncio.to_thread(sync, config.load_checkpoint_path, **kwargs))
                if callable(sync)
                else (
                    lambda: (
                        service.create_training_client_from_state_with_optimizer_async(
                            config.load_checkpoint_path, **kwargs
                        )
                    )
                )
            )
        else:
            sync = getattr(service, "create_lora_training_client", None)
            kwargs = {"base_model": config.model_name, "rank": config.lora_rank}
            call = (
                (lambda: asyncio.to_thread(sync, **kwargs))
                if callable(sync)
                else (lambda: service.create_lora_training_client_async(**kwargs))
            )
        training_client = await _remote(
            "training_client_create", call, config, progress
        )
    elif config.load_checkpoint_path is not None:
        await _remote(
            "state_resume",
            lambda: _load_state_checkpoint(
                training_client, config.load_checkpoint_path
            ),
            config,
            progress,
        )
    log_path = Path(config.log_path)
    # Avoid serializing episode_config, which may contain private endpoint data.
    _write_json(
        log_path / "training_config.json",
        {key: value for key, value in asdict(config).items() if key != "episode_config"}
        | {
            "task_id": config.episode_config.get("task_id"),
            "destination": config.episode_config.get("destination"),
            "renderer_name": _renderer_name(config),
            "loss_fn": "importance_sampling",
            "advantage": "reward_minus_group_mean",
            "injected_runtime": injected,
        },
    )
    paths = await _remote(
        "checkpoint_initial",
        lambda: save_checkpoint_async(
            training_client,
            "initial",
            str(log_path),
            {"step": 0, "injected_runtime": injected},
            kind="both",
        ),
        config,
        progress,
    )
    progress({"event": "checkpoint_saved", "checkpoint": "initial", **paths})
    sampling_client = await _create_sampler(
        training_client, config, progress, model_path=paths["sampler_path"]
    )
    steps: list[dict[str, Any]] = []
    evidence_path = log_path / "training_result.json"
    for step in range(config.max_steps):
        original_builder = dataset.get_batch(step)[0]
        assert isinstance(original_builder, ELTEnvGroupBuilder)
        original_builder.episode_factory = episode_factory
        original_builder.progress = progress
        if completer_factory is not None:
            completer = completer_factory(sampling_client)
        else:
            completer = TinkerTokenCompleter(
                sampling_client,
                max_tokens=config.max_turn_tokens,
                temperature=config.temperature,
                context_window=config.max_context_tokens,
            )
        completer = ObservedCompleter(completer, config, renderer, progress, budget)
        data: list[tinker.Datum] = []
        stats: dict[str, float] = {}
        group: TrajectoryGroup | None = None
        for attempt in range(config.max_constant_group_retries + 1):
            progress.context.update(step=step, attempt=attempt)
            progress({"event": "group_start", "group_size": config.group_size})
            group = await do_group_rollout(
                original_builder if attempt == 0 else replace(original_builder),
                completer,
                strategy=BoundedEpisodeStrategy(config.serial_episodes),
            )
            rewards = group.get_total_rewards()
            if not all(math.isfinite(reward) for reward in rewards):
                raise ValueError("Rollout produced non-finite rewards")
            _append_json(
                log_path / "train_rollouts.jsonl",
                _trace(group, renderer, step=step, attempt=attempt),
            )
            advantages = compute_advantages([group])
            data, _metadata = assemble_training_data([group], advantages)
            stats = _effective_data_stats(data)
            stats.update(
                training_input_tokens=float(
                    sum(datum.model_input.length for datum in data)
                ),
                training_datums=float(len(data)),
            )
            progress({"event": "group_end", "rewards": rewards, **stats})
            _append_json(
                log_path / "metrics.jsonl",
                {
                    "step": step,
                    "attempt": attempt,
                    "rewards": rewards,
                    "reward_variance": float(np.var(rewards)),
                    "advantages": advantages[0].tolist(),
                    **stats,
                },
            )
            if stats["effective_action_tokens"] > 0:
                break
        if not data or stats.get("effective_action_tokens", 0.0) <= 0:
            _write_json(
                evidence_path,
                {
                    "status": "no_learning_signal",
                    "steps": steps,
                    "injected_runtime": injected,
                    "attempts": config.max_constant_group_retries + 1,
                },
            )
            raise NoLearningSignalError(
                f"No nonzero effective advantage after {config.max_constant_group_retries + 1} groups; "
                "no forward_backward or optim_step was submitted for this step"
            )
        probe_prompt, probe_mask = _probe(data)
        progress({"event": "probe_before"})
        before = await _probe_logprobs(
            sampling_client, probe_prompt, probe_mask, config, progress, budget
        )
        optimizer_metrics: dict[str, Any] = {}
        observed_client = _ObservedTrainingClient(
            training_client, config, progress, budget
        )
        try:
            training_logprobs = await _remote(
                "optimizer_step",
                lambda data=data, observed_client=observed_client, optimizer_metrics=optimizer_metrics: (
                    train_step(
                        data,
                        observed_client,
                        config.learning_rate,
                        num_substeps=1,
                        loss_fn="importance_sampling",
                        metrics=optimizer_metrics,
                    )
                ),
                config,
                progress,
            )
        except BaseException as error:
            observed_client.uncertain(type(error).__name__)
            raise
        if len(training_logprobs) != len(data) or any(
            tensor.numel() != datum.model_input.length or not tensor.isfinite().all()
            for tensor, datum in zip(training_logprobs, data, strict=True)
        ):
            raise NoWeightUpdateError(
                "Optimizer response lacks finite per-datum training logprobs"
            )
        progress(
            {"event": "optimizer_completed", "optimizer_metrics": optimizer_metrics}
        )
        paths = await _remote(
            "checkpoint_updated",
            lambda step=step: save_checkpoint_async(
                training_client,
                f"step-{step + 1}",
                str(log_path),
                {"step": step + 1, "injected_runtime": injected},
                kind="both",
            ),
            config,
            progress,
        )
        progress(
            {"event": "checkpoint_saved", "checkpoint": f"step-{step + 1}", **paths}
        )
        # Always refresh the client: an old sampler would hide real changes.
        sampling_client = await _create_sampler(
            training_client, config, progress, model_path=paths["sampler_path"]
        )
        progress({"event": "probe_after"})
        after = await _probe_logprobs(
            sampling_client, probe_prompt, probe_mask, config, progress, budget
        )
        delta = float(np.max(np.abs(after - before)))
        updated = delta > config.update_tolerance
        step_evidence = {
            "step": step + 1,
            "rewards": group.get_total_rewards() if group else [],
            **stats,
            "optimizer_completed": True,
            "optimizer_metrics": optimizer_metrics,
            "probe_action_tokens": len(before),
            "probe_max_abs_logprob_delta": delta,
            "probe_tolerance": config.update_tolerance,
            "effective_update": updated,
            "state_load_verified": False,
            **paths,
        }
        # Persist completed optimizer/probe facts before any additional API
        # verification; a late reload error must not erase a real update.
        _write_json(
            log_path / "optimizer_evidence.json",
            {
                "status": "checkpoint_verification_pending",
                "step": step_evidence,
                "probe_before_logprobs": before.tolist(),
                "probe_after_logprobs": after.tolist(),
                "injected_runtime": injected,
            },
        )
        progress({"event": "probe_comparison", **step_evidence})
        state_load_response = await _fresh_state_checkpoint(
            service or checkpoint_service,
            config,
            progress,
            paths["state_path"],
            previous_model_id=getattr(training_client, "model_id", None),
        )
        step_evidence.update(
            state_load_verified=True, state_load_response=state_load_response
        )
        steps.append(step_evidence)
        _append_json(log_path / "metrics.jsonl", step_evidence)
        _write_json(
            evidence_path,
            {
                "status": "verified_update" if updated else "update_unverified",
                "steps": steps,
                "injected_runtime": injected,
            },
        )
        if not updated:
            raise NoWeightUpdateError(
                "Optimizer completed, but fixed action-token logprobs did not change above tolerance; "
                "checkpoint paths alone are not proof of a useful update"
            )
    result = TrainingResult(
        steps,
        paths["state_path"],
        paths["sampler_path"],
        True,
        injected,
        str(evidence_path),
    )
    _write_json(evidence_path, result.as_dict() | {"status": "verified_update"})
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _recovery_probe(training_run: Path, config: TrainingConfig, renderer: Renderer):
    """Use recorded model tokens; never generate actions or consult references."""
    files = (
        "training_config.json",
        "checkpoints.jsonl",
        "train_rollouts.jsonl",
        "progress.jsonl",
        "metrics.jsonl",
        "manifest.json",
    )
    hashes = {
        name: hashlib.sha256((training_run / name).read_bytes()).hexdigest()
        for name in files
    }
    recorded = json.loads((training_run / "training_config.json").read_text())
    if recorded["model_name"] != config.model_name or recorded[
        "renderer_name"
    ] != _renderer_name(config):
        raise ValueError(
            "Verification model/renderer differs from the recorded training run"
        )
    checkpoints = _read_jsonl(training_run / "checkpoints.jsonl")
    by_step = {}
    for record in checkpoints:
        step = record.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Checkpoint step metadata is invalid")
        if step in by_step:
            raise ValueError("Checkpoint step metadata is ambiguous")
        by_step[step] = record
    if 0 not in by_step or max(by_step) < 1:
        raise ValueError("Verification requires initial and completed-step checkpoints")
    latest = by_step[max(by_step)]
    previous = by_step.get(latest["step"] - 1)
    if previous is None:
        raise ValueError("Latest checkpoint has no immediately preceding checkpoint")
    groups = _read_jsonl(training_run / "train_rollouts.jsonl")
    events = _read_jsonl(training_run / "progress.jsonl")
    completed = {
        (event.get("step"), event.get("attempt"))
        for event in events
        if event.get("event") == "remote_end" and event.get("phase") == "optimizer_wait"
    }
    matching = [
        group
        for group in groups
        if group.get("step") == latest["step"] - 1
        and (group.get("step"), group.get("attempt")) in completed
    ]
    if len(matching) != 1:
        raise ValueError(
            "Latest checkpoint requires one matching optimized rollout group"
        )
    group = matching[0]
    stats = [
        row
        for row in _read_jsonl(training_run / "metrics.jsonl")
        if row.get("step") == group["step"] and row.get("attempt") == group["attempt"]
    ]
    if not stats or stats[-1].get("effective_action_tokens", 0) <= 0:
        raise NoLearningSignalError(
            "Recorded run lacks nonzero effective training advantages"
        )
    mean = float(np.mean(group["rewards"]))
    transitions = [
        transition
        for trajectory, reward in zip(
            group["trajectories"], group["rewards"], strict=True
        )
        if reward != mean
        for transition in trajectory["transitions"]
        if transition["action_tokens"]
        and float(transition.get("metrics", {}).get(PARSE_ERROR_MASKED_METRIC_KEY, 0))
        < 1
    ]
    if not transitions:
        raise NoLearningSignalError(
            "Recorded rollout lacks an action with nonzero trajectory advantage"
        )
    # Fix this rule before any checkpoint probe. A longer recorded action
    # covers more actual decision tokens than a ubiquitous initial tool call.
    transition = max(transitions, key=lambda row: len(row["action_tokens"]))
    observation = transition["observation"]
    prompt_tokens = renderer.tokenizer.encode(observation, add_special_tokens=False)
    if renderer.tokenizer.decode(prompt_tokens) != observation:
        raise ValueError(
            "Recorded observation does not round-trip through the pinned tokenizer"
        )
    exact = transition.get("observation_tokens")
    if exact is not None and list(prompt_tokens) != exact:
        raise ValueError(
            "Reencoded observation differs from the recorded exact token IDs"
        )
    action_tokens = transition["action_tokens"]
    samples = [
        event
        for event in events
        if event.get("event") == "sample_end"
        and event.get("step") == group["step"]
        and event.get("attempt") == group["attempt"]
        and event.get("action_tokens") == action_tokens
        and event.get("prompt_tokens") == len(prompt_tokens)
    ]
    if not samples:
        raise ValueError(
            "Recorded sampling evidence does not match the reconstructed prompt/action"
        )
    full = list(prompt_tokens) + action_tokens
    mask = np.asarray([False] * (len(prompt_tokens) - 1) + [True] * len(action_tokens))
    original_manifest = json.loads((training_run / "manifest.json").read_text())
    provenance = {
        "original_training_run": str(training_run),
        "original_source_revision": original_manifest.get("source_revision"),
        "original_manifest_status": original_manifest.get("status"),
        "source_sha256": hashes,
        "original_injected_runtime": recorded.get("injected_runtime"),
        "original_optimizer_completed": True,
        "original_checkpoint_step": latest["step"],
        "original_group_step": group["step"],
        "original_group_attempt": group["attempt"],
        "comparison_predecessor_step": previous["step"],
        "original_rewards": group["rewards"],
        "original_effective_action_tokens": stats[-1]["effective_action_tokens"],
        "original_training_input_tokens": stats[-1].get("training_input_tokens"),
        "prompt_tokens": len(prompt_tokens),
        "action_tokens": action_tokens,
        "probe_selection_rule": "longest_unmasked_recorded_action_with_nonzero_trajectory_advantage",
        "probe_sequence_sha256": hashlib.sha256(json.dumps(full).encode()).hexdigest(),
        "token_provenance": "exact_recorded_tokens"
        if exact is not None
        else "decoded_observation_roundtrip_and_recorded_length",
    }
    return previous, latest, tinker.ModelInput.from_ints(full), mask, provenance


async def verify_checkpoint(
    config: TrainingConfig | Mapping[str, Any],
    *,
    training_run: Path | str,
    allow_paid: bool = False,
    service_client: tinker.ServiceClient | None = None,
    renderer: Renderer | None = None,
    budget: TokenBudget | None = None,
) -> dict[str, Any]:
    """Verify a completed saved update without sampling rollouts or optimizing.

    Writes new verification evidence only. The original failed/succeeded run is
    immutable input, including its manifest and checkpoint records.
    """
    # Verification selects recorded checkpoint pairs; resume/sampler overrides
    # from the original training configuration are not instructions to load here.
    config = replace(
        _config(config), load_checkpoint_path=None, sampler_checkpoint_path=None
    )
    config.validate(training=False)
    injected = service_client is not None
    if service_client is None and not allow_paid:
        raise PaidAccessRequired(
            "Real checkpoint verification requires allow_paid=True"
        )
    source = Path(training_run).resolve()
    output = Path(config.log_path).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError(
            "Verification output must be separate from the original training run"
        )
    renderer = renderer or load_renderer(config)
    previous, updated, prompt, mask, provenance = _recovery_probe(
        source, config, renderer
    )
    if not injected and provenance["original_injected_runtime"] is not False:
        raise ValueError(
            "Real verification requires an original run without injected runtime"
        )
    output.mkdir(parents=True, exist_ok=True)
    progress = DurableProgress(output, injected_runtime=injected)
    progress.context.update(
        renderer_name=_renderer_name(config), verification_only=True
    )
    service = service_client
    status = "errored"
    try:
        progress({"event": "checkpoint_verification_start", **provenance})
        if service is None:
            service = tinker.ServiceClient()
        before_client = await _create_sampler(
            service, config, progress, model_path=previous["sampler_path"]
        )
        before = await _probe_logprobs(
            before_client, prompt, mask, config, progress, budget
        )
        after_client = await _create_sampler(
            service, config, progress, model_path=updated["sampler_path"]
        )
        after = await _probe_logprobs(
            after_client, prompt, mask, config, progress, budget
        )
        delta = float(np.max(np.abs(after - before)))
        result = {
            **provenance,
            "status": "checkpoint_verification_pending",
            "injected_runtime": injected,
            "verification_only": True,
            "optimizer_repeated": False,
            "renderer_name": _renderer_name(config),
            "initial_sampler_path": previous["sampler_path"],
            "predecessor_sampler_path": previous["sampler_path"],
            "sampler_path": updated["sampler_path"],
            "state_path": updated["state_path"],
            "probe_before_logprobs": before.tolist(),
            "probe_after_logprobs": after.tolist(),
            "probe_action_tokens": len(before),
            "probe_max_abs_logprob_delta": delta,
            "probe_tolerance": config.update_tolerance,
            "effective_update": delta > config.update_tolerance,
            "state_load_verified": False,
        }
        _write_json(output / "verification.json", result)
        progress({"event": "probe_comparison", "probe_max_abs_logprob_delta": delta})
        loaded = await _fresh_state_checkpoint(
            service, config, progress, updated["state_path"]
        )
        result.update(
            status="verified_saved_update"
            if result["effective_update"]
            else "update_unverified",
            state_load_verified=True,
            state_load_response=loaded,
        )
        _write_json(output / "verification.json", result)
        if not result["effective_update"]:
            raise NoWeightUpdateError(
                "Saved sampler checkpoints lack observable action-logprob changes"
            )
        progress({"event": "checkpoint_verification_end", "effective_update": True})
        status = "success"
        return result
    except BaseException as error:
        if isinstance(error, asyncio.CancelledError):
            status = "interrupted"
        _failure(output, progress, error)
        raise
    finally:
        if not injected:
            await _close_service(service, status, config, progress)
