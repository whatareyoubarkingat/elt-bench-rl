"""Offline tests of the tool adapter/runner with an in-memory test codec."""

from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path

import pytest
import tinker
from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers.base import ParseTermination, ToolCall
from tinker_cookbook.rl.rollouts import do_group_rollout, do_single_rollout

from elt_bench_rl.cookbook_env import (
    BoundedEpisodeStrategy,
    ELTCookbookEnv,
    ELTEnvGroupBuilder,
)
from elt_bench_rl.environment import GradeResult
from elt_bench_rl.errors import InfrastructureError


class ToyTokenizer:
    """Reversible character tokens; never suitable for a real model request."""

    def encode(self, text, *, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class ToyRenderer:
    """The public renderer seam used by Cookbook's actual message/token Env.

    This keeps environment tests independent of Hugging Face caches. Model
    template compatibility is checked separately by scripts/check_tokenizer.py.
    """

    def __init__(self):
        self.tokenizer = ToyTokenizer()

    def get_stop_sequences(self):
        return [3]

    def create_conversation_prefix_with_tools(self, tools, system_prompt):
        return [{"role": "system", "content": system_prompt + "\n" + json.dumps(tools)}]

    def build_generation_prompt(self, messages, **kwargs):
        text = json.dumps(
            messages, ensure_ascii=False, default=lambda value: value.model_dump()
        )
        return tinker.ModelInput.from_ints(self.tokenizer.encode(text + "\nassistant:"))

    def parse_response(self, tokens):
        text = self.tokenizer.decode(tokens)
        if not text.endswith("\x03"):
            return {"role": "assistant", "content": text}, ParseTermination.MALFORMED
        payload = json.loads(text[:-1])
        call = ToolCall(
            id="scripted-call",
            function=ToolCall.FunctionBody(
                name=payload["tool"], arguments=json.dumps(payload["arguments"])
            ),
        )
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [call],
        }, ParseTermination.STOP_SEQUENCE


def render_tool_call(renderer, name, arguments):
    return renderer.tokenizer.encode(
        json.dumps({"tool": name, "arguments": arguments}) + "\x03"
    )


@pytest.fixture(scope="module")
def renderer():
    return ToyRenderer()


class StubEpisode:
    public_prompt = "Load the source, write the target and submit the pipeline."

    def __init__(self, reward=1.0, fail=False):
        self.reward = reward
        self.fail = fail
        self.submitted = False
        self.result = None
        self.calls = []
        self.close_count = 0

    def tools_schema(self):
        return [
            {
                "name": name,
                "description": f"Perform {name}.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            }
            for name in ("write", "submit")
        ]

    def dispatch(self, name, arguments):
        if self.fail:
            raise InfrastructureError("Trusted fixture is unavailable")
        self.calls.append((name, arguments))
        if name == "submit":
            self.submitted = True
            self.result = GradeResult(
                reward=self.reward, loading=1, target=self.reward, replay=self.reward
            )
        return {"ok": True, "value": arguments.get("value", "")}

    def close(self):
        self.close_count += 1


def tool_call(name, **arguments):
    return ToolCall(
        id=f"call-{name}",
        function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)),
    )


def response_tokens(renderer, name="submit", value=""):
    return render_tool_call(renderer, name, {"value": value} if value else {})


class SubmitCompleter(TokenCompleter):
    def __init__(self, renderer):
        self.tokens = response_tokens(renderer)
        self.calls = 0
        self.max_tokens_seen = []
        self.active = 0
        self.max_active = 0

    async def __call__(self, model_input, stop, *, max_tokens=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        tokens = self.tokens[:max_tokens] if max_tokens is not None else self.tokens
        reason = "length" if len(tokens) < len(self.tokens) else "stop"
        return TokensWithLogprobs(tokens, [-1.0] * len(tokens), reason)


@pytest.mark.asyncio
async def test_submit_stops_later_calls_and_reward_is_terminal(renderer):
    episode = StubEpisode()
    env = ELTCookbookEnv(episode, renderer)
    initial = await env.message_env.initial_observation()
    assert episode.public_prompt in initial[-1]["content"]
    result = await env.message_env.step(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                tool_call("write", value="first"),
                tool_call("submit"),
                tool_call("write", value="late"),
            ],
        }
    )
    assert [call[0] for call in episode.calls] == ["write", "submit"]
    assert result.episode_done and result.reward == 1
    assert result.metrics["submitted"] == 1
    await env.close()
    await env.close()
    assert episode.close_count == 1


@pytest.mark.asyncio
async def test_free_text_finish_requires_submission(renderer):
    episode = StubEpisode()
    env = ELTCookbookEnv(episode, renderer)
    await env.message_env.initial_observation()
    result = await env.message_env.step(
        {"role": "assistant", "content": "The pipeline is complete."}
    )
    assert result.episode_done and result.reward == 0
    assert result.metrics["submitted"] == 0
    assert not episode.submitted


@pytest.mark.asyncio
async def test_trusted_grade_required_and_finite(renderer):
    episode = StubEpisode()
    episode.submitted = True
    env = ELTCookbookEnv(episode, renderer)
    with pytest.raises(RuntimeError, match="without a trusted grade"):
        await env.message_env.step({"role": "assistant", "content": "done"})
    episode.result = GradeResult(reward=float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        await env.message_env.step({"role": "assistant", "content": "done"})


@pytest.mark.asyncio
async def test_tool_output_truncation_is_visible_and_bounded(renderer):
    episode = StubEpisode()
    env = ELTCookbookEnv(episode, renderer, max_tool_output_tokens=64)
    result = await env.message_env.step(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                tool_call("write", value="long output " * 3000),
            ],
        }
    )
    content = result.next_messages[-1]["content"]
    assert "truncated" in content
    assert len(renderer.tokenizer.encode(content, add_special_tokens=False)) <= 64
    assert result.metrics["tool_output_truncated"] == 1


@pytest.mark.asyncio
async def test_cookbook_group_serial_and_reward_not_double_counted(renderer):
    episodes = []

    def factory(config):
        episode = StubEpisode(reward=len(episodes) / 3)
        episodes.append(episode)
        return episode

    builder = ELTEnvGroupBuilder(
        {"task_id": "fixture"}, renderer, episode_factory=factory
    )
    policy = SubmitCompleter(renderer)
    group = await do_group_rollout(builder, policy, strategy=BoundedEpisodeStrategy(1))
    assert group.get_total_rewards() == [0, 1 / 3, 2 / 3, 1]
    assert group.final_rewards_G == [0, 0, 0, 0]
    assert policy.max_active == 1
    assert all(episode.close_count == 1 for episode in episodes)
    await builder.cleanup()
    assert all(episode.close_count == 1 for episode in episodes)


@pytest.mark.asyncio
async def test_infrastructure_failure_propagates_and_group_is_cleaned(renderer):
    episodes = []

    def factory(config):
        episode = StubEpisode(fail=True)
        episodes.append(episode)
        return episode

    builder = ELTEnvGroupBuilder({}, renderer, episode_factory=factory)
    with pytest.raises(InfrastructureError, match="fixture"):
        await do_group_rollout(
            builder, SubmitCompleter(renderer), strategy=BoundedEpisodeStrategy()
        )
    assert all(episode.close_count == 1 for episode in episodes)


@pytest.mark.asyncio
async def test_trusted_timeout_signal_stops_without_submission(renderer):
    episode = StubEpisode()
    episode.dispatch = lambda _name, _args: {
        "should_stop": True,
        "error": "Episode timeout",
    }
    env = ELTCookbookEnv(episode, renderer)
    result = await env.message_env.step(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                tool_call("write"),
                tool_call("submit"),
            ],
        }
    )
    assert result.episode_done and result.reward == 0
    assert not episode.submitted


@pytest.mark.asyncio
async def test_group_defers_episode_construction_until_rollout_starts(renderer):
    episodes = []

    def factory(_config):
        episode = StubEpisode()
        episodes.append(episode)
        return episode

    builder = ELTEnvGroupBuilder({}, renderer, episode_factory=factory)
    envs = await builder.make_envs()
    assert len(envs) == 4 and not episodes
    await do_single_rollout(SubmitCompleter(renderer), envs[0])
    assert len(episodes) == 1
    await builder.cleanup()
    assert episodes[0].close_count == 1


@pytest.mark.asyncio
async def test_cleanup_does_not_mask_primary_infrastructure_error(renderer, caplog):
    episode = StubEpisode(fail=True)

    def close():
        episode.close_count += 1
        raise InfrastructureError("private-cleanup-error-canary")

    episode.close = close
    builder = ELTEnvGroupBuilder(
        {}, renderer, group_size=1, episode_factory=lambda _: episode
    )
    with pytest.raises(InfrastructureError, match="fixture is unavailable"):
        await do_group_rollout(
            builder, SubmitCompleter(renderer), strategy=BoundedEpisodeStrategy()
        )
    assert episode.close_count == 2
    assert "private-cleanup-error-canary" not in caplog.text


@pytest.mark.asyncio
async def test_truncated_actions_never_dispatch_tools_and_sampling_budget_holds(
    renderer,
):
    episode = StubEpisode()
    policy = SubmitCompleter(renderer)
    env = ELTCookbookEnv(episode, renderer, max_sampled_tokens=5, max_turn_tokens=2)
    trajectory = await do_single_rollout(policy, env)
    # The pinned Cookbook declines sampling when only one token remains.
    assert sum(len(transition.ac.tokens) for transition in trajectory.transitions) == 4
    assert trajectory.stop_reason == "max_sampled_tokens"
    assert not episode.calls
    assert policy.max_tokens_seen == [2, 2]
    assert (
        len(env.message_env.history) == 4
    )  # prefix/user plus each retained truncation


def test_production_group_builder_is_pickleable(renderer):
    builder = ELTEnvGroupBuilder(
        {"task_id": "fixture", "project_root": "/project"}, renderer
    )
    restored = pickle.loads(pickle.dumps(builder))
    assert restored.episode_config == builder.episode_config
    assert restored.group_size == 4


@pytest.mark.asyncio
async def test_progress_orders_episode_tool_and_turn_events_without_private_grade(
    renderer,
):
    episodes, events = [], []
    private = "trusted-grade-details-private-canary"

    def factory(_config):
        episode = StubEpisode()
        episode.work_dir = Path("/private/runs") / f"episode_{len(episodes)}"
        original = episode.dispatch

        def dispatch(name, arguments):
            feedback = original(name, arguments)
            if episode.result is not None:
                episode.result.details["private"] = private
            return feedback

        episode.dispatch = dispatch
        episodes.append(episode)
        return episode

    builder = ELTEnvGroupBuilder(
        {}, renderer, group_size=2, episode_factory=factory, progress=events.append
    )
    group = await do_group_rollout(
        builder, SubmitCompleter(renderer), strategy=BoundedEpisodeStrategy(1)
    )
    assert group.get_total_rewards() == [1, 1]
    for index in range(2):
        scoped = [event for event in events if event["episode_index"] == index]
        assert [event["event"] for event in scoped] == [
            "episode_start",
            "episode_ready",
            "tool_start",
            "tool_end",
            "turn_end",
            "episode_end",
        ]
        assert all(event["episode_id"] == f"episode_{index}" for event in scoped[1:])
        assert scoped[2]["turn"] == scoped[3]["turn"] == scoped[4]["turn"] == 1
        assert scoped[4]["episode_done"] and scoped[4]["reward"] == 1
    assert private not in json.dumps(events)
    assert all(episode.close_count == 1 for episode in episodes)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_event", ["tool_start", "tool_end"])
async def test_progress_callback_failure_propagates_and_closes_episode(
    renderer, failure_event
):
    episode = StubEpisode()

    def callback(event):
        if event["event"] == failure_event:
            raise OSError("Progress journal failed")

    builder = ELTEnvGroupBuilder(
        {}, renderer, group_size=1, episode_factory=lambda _: episode, progress=callback
    )
    with pytest.raises(OSError, match="journal failed"):
        await do_group_rollout(
            builder, SubmitCompleter(renderer), strategy=BoundedEpisodeStrategy(1)
        )
    assert episode.close_count == 1
    assert bool(episode.calls) == (failure_event == "tool_end")


@pytest.mark.asyncio
async def test_tool_error_progress_contains_type_but_no_private_exception_text(
    renderer,
):
    episode = StubEpisode()
    events = []

    def dispatch(_name, _arguments):
        raise InfrastructureError("private-infrastructure-error-canary")

    episode.dispatch = dispatch
    builder = ELTEnvGroupBuilder(
        {},
        renderer,
        group_size=1,
        episode_factory=lambda _: episode,
        progress=events.append,
    )
    with pytest.raises(InfrastructureError):
        await do_group_rollout(
            builder, SubmitCompleter(renderer), strategy=BoundedEpisodeStrategy(1)
        )
    error = next(event for event in events if event["event"] == "tool_error")
    assert error["error_type"] == "InfrastructureError"
    assert "private-infrastructure-error-canary" not in json.dumps(events)
    assert episode.close_count == 1
