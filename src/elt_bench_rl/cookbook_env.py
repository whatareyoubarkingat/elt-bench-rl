"""Stateful ELT episodes adapted to the pinned Tinker Cookbook interfaces.

Only this module knows about Cookbook's message/token environment boundary.
Warehouse and verifier logic live in the trusted Episode implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import chz
from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.renderers.base import Message, ToolCall
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageStepResult
from tinker_cookbook.rl.rollout_limits import ParseErrorPolicy, RolloutLimits
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv
from tinker_cookbook.tool_use.tools import handle_tool_call, simple_tool_result
from tinker_cookbook.tool_use.types import ToolInput, ToolResult

logger = logging.getLogger(__name__)


class Grade(Protocol):
    reward: float

    def metrics(self) -> dict[str, float]: ...


class EpisodeLike(Protocol):
    public_prompt: str
    submitted: bool
    result: Grade | None

    def tools_schema(self) -> list[dict[str, Any]]: ...

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


EpisodeFactory = Callable[[dict[str, Any]], EpisodeLike]
ProgressCallback = Callable[[dict[str, Any]], None]


def build_episode(config: dict[str, Any]) -> EpisodeLike:
    """Late import keeps datasets pickleable and import-only checks credential-free."""
    from .factory import build_episode as factory

    return factory(**config)


@dataclass
class EpisodeTool:
    episode: EpisodeLike
    spec: Mapping[str, Any]
    renderer: Renderer
    max_output_tokens: int = 2048
    dispatch_lock: Any = field(default_factory=threading.Lock, repr=False)
    truncated_count: int = 0
    progress: ProgressCallback | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return str(self.spec["name"])

    @property
    def description(self) -> str:
        return str(self.spec["description"])

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return dict(self.spec["parameters"])

    def to_spec(self) -> dict[str, Any]:
        return dict(self.spec)

    async def run(self, input: ToolInput) -> ToolResult:
        # Do not catch infrastructure errors here: they are unusable samples,
        # rather than evidence that the sampled pipeline is incorrect.
        def dispatch() -> dict[str, Any]:
            with self.dispatch_lock:
                return self.episode.dispatch(self.name, input.arguments)

        if self.progress:
            self.progress(
                {"event": "tool_start", "tool": self.name, "arguments": input.arguments}
            )
        try:
            response = await asyncio.to_thread(dispatch)
        except Exception as error:
            if self.progress:
                self.progress(
                    {
                        "event": "tool_error",
                        "tool": self.name,
                        "error_type": type(error).__name__,
                    }
                )
            raise
        if self.progress:
            # Dispatch exposes only model-visible feedback, never grade.details.
            self.progress(
                {"event": "tool_end", "tool": self.name, "feedback": response}
            )
        content = json.dumps(response, ensure_ascii=False, allow_nan=False)
        encoded = self.renderer.tokenizer.encode(content, add_special_tokens=False)
        truncated = len(encoded) > self.max_output_tokens
        if truncated:
            suffix = "\n[tool output truncated; inspect a smaller range]"
            suffix_tokens = self.renderer.tokenizer.encode(
                suffix, add_special_tokens=False
            )
            if len(suffix_tokens) >= self.max_output_tokens:
                content = self.renderer.tokenizer.decode(
                    suffix_tokens[: self.max_output_tokens]
                )
            else:
                keep = self.max_output_tokens - len(suffix_tokens)
                content = self.renderer.tokenizer.decode(encoded[:keep]) + suffix
            self.truncated_count += 1
        return simple_tool_result(
            content,
            call_id=input.call_id or "",
            name=self.name,
            should_stop=bool(response.get("should_stop") or self.episode.submitted),
            metrics={"tool_output_truncated": float(truncated)},
        )


class ELTMessageEnv(AgentToolMessageEnv):
    """Cookbook tool environment with strict stop-after-submit ordering."""

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[Message]:
        messages: list[Message] = []
        for call in tool_calls:
            result = await handle_tool_call(self._tool_dict, call)
            self.history.extend(result.messages)
            messages.extend(result.messages)
            if result.should_stop:
                self._should_stop = True
                break
        return messages

    async def step(self, message: Message) -> MessageStepResult:
        result = await super().step(message)
        result.metrics["tool_output_truncated"] = float(
            sum(
                tool.truncated_count
                for tool in self.tools
                if isinstance(tool, EpisodeTool)
            )
        )
        return result


class ELTCookbookEnv(EnvFromMessageEnv):
    def __init__(
        self,
        episode: EpisodeLike,
        renderer: Renderer,
        *,
        max_turns: int = 24,
        max_sampled_tokens: int = 8192,
        max_turn_tokens: int = 2048,
        max_context_tokens: int = 32768,
        max_tool_output_tokens: int = 2048,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.episode = episode
        self.closed = False
        self.turn = 0
        self.progress = progress
        self.dispatch_lock = threading.Lock()
        tools = [
            EpisodeTool(
                episode,
                spec,
                renderer,
                max_tool_output_tokens,
                self.dispatch_lock,
                progress=self._progress,
            )
            for spec in episode.tools_schema()
        ]
        prefix = renderer.create_conversation_prefix_with_tools(
            [tool.to_spec() for tool in tools],
            "Use the provided tools to build and submit an ELT pipeline. "
            "A text-only answer ends the attempt without submission. "
            "Finish by calling submit_pipeline after loading the sources and running dbt.",
        )

        async def reward_fn(_messages: list[Message]) -> tuple[float, dict[str, float]]:
            if not episode.submitted:
                return 0.0, {"submitted": 0.0}
            if episode.result is None:
                raise RuntimeError(
                    "Episode reported submission without a trusted grade"
                )
            reward = float(episode.result.reward)
            if not math.isfinite(reward):
                raise ValueError("Verifier returned a non-finite reward")
            return reward, {**episode.result.metrics(), "submitted": 1.0}

        parse_policy = ParseErrorPolicy(
            max_consecutive=2,
            penalty_per_error=0.0,
            terminal_reward=0.0,
            mask_error_turns=True,
        )
        message_env = ELTMessageEnv(
            tools=tools,
            initial_messages=prefix
            + [{"role": "user", "content": episode.public_prompt}],
            max_turns=max_turns,
            reward_fn=reward_fn,
            failed_parse_reward=0.0,
            tool_execution="sequential",
            parse_error_policy=parse_policy,
        )
        super().__init__(
            renderer,
            message_env,
            failed_parse_reward=0.0,
            context_overflow_reward=0.0,
            terminate_on_length=False,
            parse_error_policy=parse_policy,
            max_trajectory_tokens=max_context_tokens,
            rollout_limits=RolloutLimits(
                max_turns=max_turns,
                max_sampled_tokens=max_sampled_tokens,
                max_turn_tokens=max_turn_tokens,
                max_trajectory_tokens=max_context_tokens,
            ),
        )

    def _progress(self, event: dict[str, Any]) -> None:
        if self.progress:
            work_dir = getattr(self.episode, "work_dir", None)
            self.progress(
                {
                    **event,
                    "episode_id": work_dir.name if work_dir is not None else None,
                    "turn": self.turn,
                }
            )

    async def step(self, action, *, extra=None):
        self.turn += 1
        result = await super().step(action, extra=extra)
        self._progress(
            {
                "event": "turn_end",
                "reward": result.reward,
                "episode_done": result.episode_done,
                "metrics": dict(result.metrics),
            }
        )
        return result

    async def close(self) -> None:
        if not self.closed:

            def close_after_dispatch() -> None:
                # Cancelling to_thread does not stop its worker. Wait for an
                # in-flight dispatch before closing its warehouse connection.
                with self.dispatch_lock:
                    self.episode.close()

            await asyncio.to_thread(close_after_dispatch)
            self.closed = True
            self._progress(
                {"event": "episode_end", "submitted": self.episode.submitted}
            )


class LazyELTCookbookEnv(Env):
    """Start an episode only when it is about to run, not when a group is made."""

    def __init__(
        self,
        factory: EpisodeFactory,
        config: dict[str, Any],
        renderer: Renderer,
        *,
        max_turns: int,
        max_sampled_tokens: int,
        max_turn_tokens: int,
        max_context_tokens: int,
        max_tool_output_tokens: int,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.factory, self.config, self.renderer = factory, config, renderer
        self.progress = progress
        self.options = {
            "max_turns": max_turns,
            "max_sampled_tokens": max_sampled_tokens,
            "max_turn_tokens": max_turn_tokens,
            "max_context_tokens": max_context_tokens,
            "max_tool_output_tokens": max_tool_output_tokens,
        }
        self.rollout_limits = RolloutLimits(
            max_turns=max_turns,
            max_sampled_tokens=max_sampled_tokens,
            max_turn_tokens=max_turn_tokens,
            max_trajectory_tokens=max_context_tokens,
        )
        self.inner: ELTCookbookEnv | None = None

    async def initial_observation(self):
        if self.inner is not None:
            raise RuntimeError("ELT environments are single-use")
        if self.progress:
            self.progress({"event": "episode_start"})
        creation = asyncio.create_task(
            asyncio.to_thread(self.factory, dict(self.config))
        )
        try:
            episode = await asyncio.shield(creation)
        except asyncio.CancelledError:
            # Account for the resource even if cancellation arrives while its
            # constructor is running in an uncancellable worker thread.
            episode = await creation
            await asyncio.to_thread(episode.close)
            raise
        try:
            self.inner = ELTCookbookEnv(
                episode, self.renderer, progress=self.progress, **self.options
            )
        except BaseException:
            await asyncio.to_thread(episode.close)
            raise
        self.inner._progress({"event": "episode_ready"})
        return await self.inner.initial_observation()

    async def step(self, action, *, extra=None):
        if self.inner is None:
            raise RuntimeError("initial_observation must run before step")
        return await self.inner.step(action, extra=extra)

    async def close(self) -> None:
        if self.inner is not None:
            await self.inner.close()


@dataclass
class ELTEnvGroupBuilder(EnvGroupBuilder):
    episode_config: dict[str, Any]
    renderer: Renderer
    group_size: int = 4
    max_turns: int = 24
    max_sampled_tokens: int = 8192
    max_turn_tokens: int = 2048
    max_context_tokens: int = 32768
    max_tool_output_tokens: int = 2048
    episode_factory: EpisodeFactory | None = field(default=None, repr=False)
    progress: ProgressCallback | None = field(default=None, repr=False)
    _envs: list[LazyELTCookbookEnv] = field(
        default_factory=list, init=False, repr=False
    )

    async def make_envs(self) -> Sequence[Env]:
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
        if self._envs:
            raise RuntimeError("ELT group builders are single-use")
        factory = self.episode_factory or build_episode
        for episode_index in range(self.group_size):
            progress = self.progress

            def scoped_progress(
                event: dict[str, Any], index=episode_index, callback=progress
            ) -> None:
                if callback:
                    callback({**event, "episode_index": index})

            self._envs.append(
                LazyELTCookbookEnv(
                    factory,
                    dict(self.episode_config),
                    self.renderer,
                    max_turns=self.max_turns,
                    max_sampled_tokens=self.max_sampled_tokens,
                    max_turn_tokens=self.max_turn_tokens,
                    max_context_tokens=self.max_context_tokens,
                    max_tool_output_tokens=self.max_tool_output_tokens,
                    progress=scoped_progress if progress else None,
                )
            )
        return self._envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, dict[str, float]]]:
        # A trusted terminal grade was already returned by message_env.step.
        # Do not count it again at the group boundary.
        return [(0.0, {}) for _ in trajectory_group]

    async def cleanup(self) -> None:
        for env in self._envs:
            try:
                await env.close()
            except Exception as error:  # noqa: BLE001 - cleanup must not mask rollout failures
                logger.warning("Episode cleanup failed (%s)", type(error).__name__)

    def logging_tags(self) -> list[str]:
        return ["elt_bench", str(self.episode_config.get("task_id", "unknown"))]


@dataclass(frozen=True)
class BoundedEpisodeStrategy(RolloutStrategy):
    """Limit expensive warehouse episodes; propagate all rollout failures."""

    serial_episodes: int = 1

    async def execute(
        self, env_group_builder: EnvGroupBuilder, policy: TokenCompleter
    ) -> RolloutResult:
        if self.serial_episodes < 1:
            raise ValueError("serial_episodes must be positive")
        envs = await env_group_builder.make_envs()
        trajectories: list[Trajectory] = []

        async def run_and_close(env: Env) -> Trajectory:
            try:
                trajectory = await do_single_rollout(policy, env)
            except BaseException:
                if isinstance(env, LazyELTCookbookEnv):
                    try:
                        await env.close()
                    except Exception as error:  # noqa: BLE001 - preserve the primary rollout error
                        logger.warning(
                            "Episode cleanup failed (%s)", type(error).__name__
                        )
                raise
            # A successful rollout with unsuccessful cleanup is not a usable
            # training sample. Group cleanup retries, but must never conceal
            # an orphaned Airbyte sync/warehouse as successful execution.
            if isinstance(env, LazyELTCookbookEnv):
                await env.close()
            return trajectory

        # Batches make failure cleanup deterministic: no background rollout is
        # allowed to continue mutating a warehouse after a peer fails.
        for start in range(0, len(envs), self.serial_episodes):
            tasks = [
                asyncio.create_task(run_and_close(env))
                for env in envs[start : start + self.serial_episodes]
            ]
            try:
                trajectories.extend(await asyncio.gather(*tasks))
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        return RolloutResult(trajectories=trajectories, envs=envs, errors=[])


@dataclass
class ELTDataset(RLDataset):
    builders: list[ELTEnvGroupBuilder]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return [self.builders[index]]

    def __len__(self) -> int:
        return len(self.builders)


@chz.chz
class ELTDatasetBuilder(RLDatasetBuilder):
    episode_configs: list[dict[str, Any]]
    model_name: str = "Qwen/Qwen3.5-4B"
    tokenizer_path: str = "data/tokenizers/Qwen3.5-4B"
    group_size: int = 4
    max_steps: int = 1
    max_turns: int = 24
    max_sampled_tokens: int = 8192
    max_turn_tokens: int = 2048
    max_context_tokens: int = 32768
    max_tool_output_tokens: int = 2048

    def build_with_renderer(self, renderer: Renderer) -> ELTDataset:
        if not self.episode_configs or self.max_steps < 1 or self.group_size < 2:
            raise ValueError(
                "Training requires tasks, positive max_steps and group_size >= 2"
            )
        return ELTDataset(
            [
                ELTEnvGroupBuilder(
                    dict(self.episode_configs[index % len(self.episode_configs)]),
                    renderer,
                    group_size=self.group_size,
                    max_turns=self.max_turns,
                    max_sampled_tokens=self.max_sampled_tokens,
                    max_turn_tokens=self.max_turn_tokens,
                    max_context_tokens=self.max_context_tokens,
                    max_tool_output_tokens=self.max_tool_output_tokens,
                )
                for index in range(self.max_steps)
            ]
        )

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        from tinker_cookbook.model_info import get_recommended_renderer_name
        from transformers import AutoTokenizer

        if not self.episode_configs or self.max_steps < 1 or self.group_size < 2:
            raise ValueError(
                "Training requires tasks, positive max_steps and group_size >= 2"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, local_files_only=True, trust_remote_code=False
        )
        renderer = get_renderer(
            get_recommended_renderer_name(self.model_name), tokenizer
        )
        return self.build_with_renderer(renderer), None
