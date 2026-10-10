"""Small execution-reward RL recipe using the pinned Tinker Cookbook.

Client creation is opt-in. Injected clients/completers support genuinely offline
tests; they must never be presented as evidence of a credentialed Tinker run.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import tinker
from tinker_cookbook.checkpoint_utils import save_checkpoint_async
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.train import train_step
from tinker_cookbook.rl.types import TrajectoryGroup

from .cookbook_env import (
    BoundedEpisodeStrategy,
    ELTDatasetBuilder,
    ELTEnvGroupBuilder,
    EpisodeFactory,
)


class PaidAccessRequired(RuntimeError):
    """A real Tinker client would be created without explicit authorization."""


class NoLearningSignalError(RuntimeError):
    """Execution rewards failed to produce nonzero effective advantages."""


class NoWeightUpdateError(RuntimeError):
    """The requested step lacks observable pre/post model-change evidence."""


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
    return get_renderer(get_recommended_renderer_name(config.model_name), tokenizer)


def _group_builder(
    config: TrainingConfig, renderer: Renderer, episode_factory: EpisodeFactory | None
) -> ELTEnvGroupBuilder:
    return ELTEnvGroupBuilder(
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )


def _append_json(path: Path, value: Any) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


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
    sampling_client: tinker.SamplingClient, prompt: tinker.ModelInput, mask: np.ndarray
) -> np.ndarray:
    values = await sampling_client.compute_logprobs_async(prompt)
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


async def rollout(
    config: TrainingConfig | Mapping[str, Any],
    *,
    allow_paid: bool = False,
    completer: TokenCompleter | None = None,
    renderer: Renderer | None = None,
    episode_factory: EpisodeFactory | None = None,
) -> RolloutResult:
    config = _config(config)
    config.validate(training=False)
    injected = completer is not None
    if completer is None and not allow_paid:
        raise PaidAccessRequired("Real Tinker rollout requires allow_paid=True")
    renderer = renderer or load_renderer(config)
    if completer is None:
        service = tinker.ServiceClient()
        if config.sampler_checkpoint_path is not None:
            sampling_client = await service.create_sampling_client_async(
                model_path=config.sampler_checkpoint_path
            )
        else:
            sampling_client = await service.create_sampling_client_async(
                base_model=config.model_name
            )
        completer = TinkerTokenCompleter(
            sampling_client,
            max_tokens=config.max_turn_tokens,
            temperature=config.temperature,
            context_window=config.max_context_tokens,
        )
    log_path = Path(config.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    group = await do_group_rollout(
        _group_builder(config, renderer, episode_factory),
        completer,
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
    _write_json(log_path / "rollout_result.json", result.as_dict())
    return result


async def train(
    config: TrainingConfig | Mapping[str, Any],
    *,
    allow_paid: bool = False,
    training_client: tinker.TrainingClient | None = None,
    renderer: Renderer | None = None,
    episode_factory: EpisodeFactory | None = None,
    completer_factory: Callable[[tinker.SamplingClient], TokenCompleter] | None = None,
) -> TrainingResult:
    config = _config(config)
    config.validate(training=True)
    injected = training_client is not None
    if training_client is None and not allow_paid:
        raise PaidAccessRequired("Real Tinker optimization requires allow_paid=True")
    renderer = renderer or load_renderer(config)
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
        service = tinker.ServiceClient()
        if config.load_checkpoint_path is not None:
            training_client = (
                await service.create_training_client_from_state_with_optimizer_async(
                    config.load_checkpoint_path, base_model=config.model_name
                )
            )
        else:
            training_client = await service.create_lora_training_client_async(
                base_model=config.model_name,
                rank=config.lora_rank,
            )
    elif config.load_checkpoint_path is not None:
        await _load_state_checkpoint(training_client, config.load_checkpoint_path)
    log_path = Path(config.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    # Avoid serializing episode_config, which may contain private endpoint data.
    _write_json(
        log_path / "training_config.json",
        {key: value for key, value in asdict(config).items() if key != "episode_config"}
        | {
            "task_id": config.episode_config.get("task_id"),
            "destination": config.episode_config.get("destination"),
            "loss_fn": "importance_sampling",
            "advantage": "reward_minus_group_mean",
            "injected_runtime": injected,
        },
    )
    paths = await save_checkpoint_async(
        training_client,
        "initial",
        str(log_path),
        {"step": 0, "injected_runtime": injected},
        kind="both",
    )
    sampling_client = await training_client.create_sampling_client_async(
        paths["sampler_path"]
    )
    steps: list[dict[str, Any]] = []
    evidence_path = log_path / "training_result.json"
    for step in range(config.max_steps):
        original_builder = dataset.get_batch(step)[0]
        assert isinstance(original_builder, ELTEnvGroupBuilder)
        original_builder.episode_factory = episode_factory
        if completer_factory is not None:
            completer = completer_factory(sampling_client)
        else:
            completer = TinkerTokenCompleter(
                sampling_client,
                max_tokens=config.max_turn_tokens,
                temperature=config.temperature,
                context_window=config.max_context_tokens,
            )
        data: list[tinker.Datum] = []
        stats: dict[str, float] = {}
        group: TrajectoryGroup | None = None
        for attempt in range(config.max_constant_group_retries + 1):
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
        before = await _probe_logprobs(sampling_client, probe_prompt, probe_mask)
        optimizer_metrics: dict[str, Any] = {}
        training_logprobs = await train_step(
            data,
            training_client,
            config.learning_rate,
            num_substeps=1,
            loss_fn="importance_sampling",
            metrics=optimizer_metrics,
        )
        if len(training_logprobs) != len(data) or any(
            not tensor.isfinite().all() for tensor in training_logprobs
        ):
            raise NoWeightUpdateError(
                "Optimizer response lacks finite per-datum training logprobs"
            )
        paths = await save_checkpoint_async(
            training_client,
            f"step-{step + 1}",
            str(log_path),
            {"step": step + 1, "injected_runtime": injected},
            kind="both",
        )
        # Always refresh the client: an old sampler would hide real changes.
        sampling_client = await training_client.create_sampling_client_async(
            paths["sampler_path"]
        )
        after = await _probe_logprobs(sampling_client, probe_prompt, probe_mask)
        delta = float(np.max(np.abs(after - before)))
        updated = delta > config.update_tolerance
        # Round-trip the exact state just saved, including optimizer state.
        # This does not rewind: it restores the current completed step.
        state_load_response = await _load_state_checkpoint(
            training_client, paths["state_path"]
        )
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
            "state_load_verified": True,
            "state_load_response": state_load_response,
            **paths,
        }
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
