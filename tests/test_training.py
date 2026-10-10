"""Exercise the real Cookbook optimizer orchestration using offline fake futures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import tinker
import torch
from test_cookbook_env import StubEpisode, SubmitCompleter, ToyRenderer, response_tokens
from tinker_cookbook.rl.types import RLDatasetBuilder

from elt_bench_rl import training
from elt_bench_rl.cookbook_env import ELTDatasetBuilder
from elt_bench_rl.errors import InfrastructureError
from elt_bench_rl.training import (
    NoLearningSignalError,
    NoWeightUpdateError,
    PaidAccessRequired,
    TrainingConfig,
    rollout,
    train,
)


@pytest.fixture(autouse=True)
def forbid_cloud_client(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline test attempted to create a cloud client")

    monkeypatch.setattr(training.tinker, "ServiceClient", forbidden)


@pytest.fixture(scope="module")
def renderer():
    return ToyRenderer()


@pytest.fixture
def config(tmp_path):
    return TrainingConfig(
        {"task_id": "fixture", "destination": "duckdb"},
        str(tmp_path / "training"),
        str(tmp_path / "deliberately-absent-tokenizer"),
    )


class Future:
    def __init__(self, events, name, result, on_result=None):
        self.events = events
        self.name = name
        self.result = result
        self.on_result = on_result

    async def result_async(self):
        self.events.append(f"await:{self.name}")
        if self.on_result is not None:
            self.on_result()
        return self.result


class SamplingClient:
    def __init__(self, parent, weight):
        self.parent = parent
        self.weight = weight

    async def compute_logprobs_async(self, prompt):
        self.parent.probes.append((self.weight, prompt.to_ints()))
        return [None] + [-1.0 + self.weight] * (prompt.length - 1)


class TrainingClient:
    def __init__(self, *, changes=True):
        self.events = []
        self.saved = {}
        self.weight = 0.0
        self.changes = changes
        self.forward_calls = []
        self.optim_calls = []
        self.probes = []
        self.load_calls = []

    async def save_state_async(self, name, ttl_seconds=None):
        self.events.append(f"submit:state:{name}")
        path = f"tinker://offline/state/{name}"
        self.saved[path] = self.weight
        return Future(self.events, f"state:{name}", SimpleNamespace(path=path))

    async def load_state_with_optimizer_async(self, path):
        self.events.append("submit:load_state")
        self.load_calls.append(path)

        def restore():
            self.weight = self.saved[path]

        return Future(
            self.events,
            "load_state",
            SimpleNamespace(path=path, model_id="offline-model", type="load_weights"),
            restore,
        )

    async def save_weights_for_sampler_async(self, name, ttl_seconds=None):
        self.events.append(f"submit:sampler:{name}")
        path = f"tinker://offline/sampler/{name}"
        self.saved[path] = self.weight
        return Future(self.events, f"sampler:{name}", SimpleNamespace(path=path))

    async def create_sampling_client_async(self, path):
        return SamplingClient(self, self.saved[path])

    async def forward_backward_async(self, data, *, loss_fn, loss_fn_config=None):
        self.events.append("submit:forward_backward")
        self.forward_calls.append((data, loss_fn))
        assert all("mask" not in datum.loss_fn_inputs for datum in data)
        assert any(
            datum.loss_fn_inputs["advantages"].to_numpy().any() for datum in data
        )
        result = SimpleNamespace(
            loss_fn_outputs=[
                {
                    "logprobs": tinker.TensorData.from_torch(
                        torch.full((datum.model_input.length,), -1.0)
                    )
                }
                for datum in data
            ]
        )
        return Future(self.events, "forward_backward", result)

    async def optim_step_async(self, params):
        self.events.append("submit:optim_step")
        self.optim_calls.append(params)

        def update():
            if self.changes:
                self.weight += 0.001

        return Future(
            self.events,
            "optim_step",
            SimpleNamespace(metrics={"grad_norm": 0.25}),
            update,
        )


def factory_with_rewards(rewards):
    episodes = []

    def factory(_config):
        episode = StubEpisode(rewards[len(episodes) % len(rewards)])
        episodes.append(episode)
        return episode

    return factory, episodes


@pytest.mark.asyncio
async def test_paid_gate_precedes_client_creation(config):
    with pytest.raises(PaidAccessRequired):
        await train(config)
    with pytest.raises(PaidAccessRequired):
        await rollout(config)


@pytest.mark.asyncio
async def test_execution_reward_reaches_real_cookbook_optimizer(config, renderer):
    client = TrainingClient()
    factory, episodes = factory_with_rewards([0, 0.25, 0.75, 1])
    result = await train(
        config,
        training_client=client,
        renderer=renderer,
        episode_factory=factory,
        completer_factory=lambda _: SubmitCompleter(renderer),
    )
    assert result.effective_update and result.injected_runtime
    assert len(client.forward_calls) == len(client.optim_calls) == 1
    assert client.forward_calls[0][1] == "importance_sampling"
    assert client.optim_calls[0].learning_rate == 1e-5
    # Cookbook submits both operations before awaiting either future.
    relevant = [
        event
        for event in client.events
        if event.endswith(("forward_backward", "optim_step"))
    ]
    assert relevant == [
        "submit:forward_backward",
        "submit:optim_step",
        "await:forward_backward",
        "await:optim_step",
    ]
    assert result.steps[0]["absolute_effective_advantage"] > 0
    assert result.steps[0]["probe_max_abs_logprob_delta"] == pytest.approx(0.001)
    assert len(client.probes) == 2 and client.probes[0][1] == client.probes[1][1]
    assert client.load_calls == [result.state_path]
    assert result.steps[0]["state_load_verified"] is True
    assert result.steps[0]["state_load_response"]["path"] == result.state_path
    assert all(episode.close_count == 1 for episode in episodes)
    evidence = json.loads(Path(result.evidence_path).read_text())
    assert evidence["injected_runtime"] is True
    records = [
        json.loads(line)
        for line in (Path(config.log_path) / "checkpoints.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [record["step"] for record in records] == [0, 1]
    assert all(record["state_path"] and record["sampler_path"] for record in records)


@pytest.mark.asyncio
async def test_constant_rewards_retry_only_twice_and_never_optimize(config, renderer):
    client = TrainingClient()
    factory, episodes = factory_with_rewards([0])
    with pytest.raises(NoLearningSignalError, match="after 3 groups"):
        await train(
            config,
            training_client=client,
            renderer=renderer,
            episode_factory=factory,
            completer_factory=lambda _: SubmitCompleter(renderer),
        )
    assert len(episodes) == 12
    assert not client.forward_calls and not client.optim_calls and not client.probes
    evidence = json.loads((Path(config.log_path) / "training_result.json").read_text())
    assert evidence["status"] == "no_learning_signal"
    assert evidence["attempts"] == 3


@pytest.mark.asyncio
async def test_a_later_nonconstant_group_can_supply_signal(config, renderer):
    client = TrainingClient()
    factory, episodes = factory_with_rewards([0, 0, 0, 0, 0, 0.25, 0.75, 1])
    result = await train(
        config,
        training_client=client,
        renderer=renderer,
        episode_factory=factory,
        completer_factory=lambda _: SubmitCompleter(renderer),
    )
    assert result.effective_update and len(episodes) == 8
    assert len(client.optim_calls) == 1


@pytest.mark.asyncio
async def test_checkpoint_and_optim_success_do_not_imply_update(config, renderer):
    client = TrainingClient(changes=False)
    factory, _episodes = factory_with_rewards([0, 0.25, 0.75, 1])
    with pytest.raises(NoWeightUpdateError, match="did not change"):
        await train(
            config,
            training_client=client,
            renderer=renderer,
            episode_factory=factory,
            completer_factory=lambda _: SubmitCompleter(renderer),
        )
    assert len(client.optim_calls) == 1
    evidence = json.loads((Path(config.log_path) / "training_result.json").read_text())
    assert evidence["status"] == "update_unverified"
    assert evidence["steps"][0]["effective_update"] is False


@pytest.mark.asyncio
async def test_observations_have_zero_advantages_while_generated_tokens_train(
    config, renderer
):
    client = TrainingClient()
    factory, _episodes = factory_with_rewards([0, 0.25, 0.75, 1])
    await train(
        config,
        training_client=client,
        renderer=renderer,
        episode_factory=factory,
        completer_factory=lambda _: SubmitCompleter(renderer),
    )
    datum = client.forward_calls[0][0][0]
    advantages = datum.loss_fn_inputs["advantages"].to_numpy()
    assert advantages[0] == 0 and advantages[-1] != 0
    assert (advantages == 0).sum() > (advantages != 0).sum()


@pytest.mark.asyncio
async def test_cleanup_failure_is_retried_but_never_trains(config, renderer):
    client = TrainingClient()
    episodes = []

    def factory(_config):
        episode = StubEpisode(reward=1)

        def close():
            episode.close_count += 1
            if episode.close_count == 1:
                raise InfrastructureError("Cleanup not yet complete")

        episode.close = close
        episodes.append(episode)
        return episode

    with pytest.raises(InfrastructureError, match="Cleanup not yet complete"):
        await train(
            config,
            training_client=client,
            renderer=renderer,
            episode_factory=factory,
            completer_factory=lambda _: SubmitCompleter(renderer),
        )
    assert len(episodes) == 1 and episodes[0].close_count == 2
    assert not client.forward_calls and not client.optim_calls


@pytest.mark.asyncio
async def test_resume_loads_optimizer_state_before_rollouts(config, renderer):
    client = TrainingClient()
    resume = "tinker://offline/state/resume"
    client.saved[resume] = 0.123
    factory, _episodes = factory_with_rewards([0, 0.25, 0.75, 1])
    result = await train(
        replace(config, load_checkpoint_path=resume),
        training_client=client,
        renderer=renderer,
        episode_factory=factory,
        completer_factory=lambda _: SubmitCompleter(renderer),
    )
    assert client.load_calls == [resume, result.state_path]
    assert client.probes[0][0] == pytest.approx(0.123)
    assert client.probes[1][0] == pytest.approx(0.124)
    assert client.events[:2] == ["submit:load_state", "await:load_state"]


@pytest.mark.asyncio
async def test_state_save_response_without_successful_load_is_not_verified(
    config, renderer
):
    client = TrainingClient()

    async def unreadable(_path):
        raise RuntimeError("Saved checkpoint cannot be loaded")

    client.load_state_with_optimizer_async = unreadable
    factory, _episodes = factory_with_rewards([0, 0.25, 0.75, 1])
    with pytest.raises(RuntimeError, match="cannot be loaded"):
        await train(
            config,
            training_client=client,
            renderer=renderer,
            episode_factory=factory,
            completer_factory=lambda _: SubmitCompleter(renderer),
        )
    assert len(client.optim_calls) == 1
    assert not (Path(config.log_path) / "training_result.json").exists()


@pytest.mark.asyncio
async def test_standalone_rollout_returns_execution_reward(config, renderer):
    factory, episodes = factory_with_rewards([1])
    result = await rollout(
        replace(config, group_size=1),
        completer=SubmitCompleter(renderer),
        renderer=renderer,
        episode_factory=factory,
    )
    assert result.rewards == [1]
    assert result.metrics[0]["submitted"] == 1
    assert result.injected_runtime and episodes[0].close_count == 1
    assert Path(result.trace_path).is_file()


@pytest.mark.asyncio
async def test_standalone_rollout_can_sample_saved_policy(
    config, renderer, monkeypatch
):
    requests = []

    class Service:
        async def create_sampling_client_async(self, **kwargs):
            requests.append(kwargs)

            class Sampler:
                async def sample_async(self, *, prompt, num_samples, sampling_params):
                    assert num_samples == 1
                    tokens = response_tokens(renderer)
                    assert len(tokens) <= sampling_params.max_tokens
                    return SimpleNamespace(
                        sequences=[
                            SimpleNamespace(
                                tokens=tokens,
                                logprobs=[-1.0] * len(tokens),
                                stop_reason="stop",
                            )
                        ]
                    )

            return Sampler()

    monkeypatch.setattr(training.tinker, "ServiceClient", Service)
    path = "tinker://offline/sampler/trained-policy"
    factory, _episodes = factory_with_rewards([1])
    result = await rollout(
        replace(config, group_size=1, sampler_checkpoint_path=path),
        allow_paid=True,
        renderer=renderer,
        episode_factory=factory,
    )
    assert result.rewards == [1]
    assert requests == [{"model_path": path}]


@pytest.mark.asyncio
async def test_checkpoint_types_are_never_silently_ignored(config, renderer):
    with pytest.raises(ValueError, match="sampler_checkpoint_path"):
        await rollout(
            replace(config, load_checkpoint_path="tinker://offline/state/saved"),
            completer=SubmitCompleter(renderer),
            renderer=renderer,
        )
    with pytest.raises(ValueError, match="load_checkpoint_path"):
        await train(
            replace(config, sampler_checkpoint_path="tinker://offline/sampler/saved"),
            training_client=TrainingClient(),
            renderer=renderer,
        )


@pytest.mark.asyncio
async def test_dataset_builder_uses_offline_tokenizer_and_fixed_groups(
    config, renderer, monkeypatch
):
    from transformers import AutoTokenizer

    from elt_bench_rl import cookbook_env

    calls = []

    def local_tokenizer(path, **kwargs):
        calls.append((path, kwargs))
        return renderer.tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", local_tokenizer)
    monkeypatch.setattr(
        cookbook_env, "get_renderer", lambda _name, _tokenizer: renderer
    )
    builder = ELTDatasetBuilder(
        episode_configs=[config.episode_config],
        tokenizer_path=config.tokenizer_path,
        max_steps=2,
    )
    assert isinstance(builder, RLDatasetBuilder)
    dataset, test = await builder()
    assert calls == [
        (config.tokenizer_path, {"local_files_only": True, "trust_remote_code": False})
    ]
    assert len(dataset) == 2 and test is None
    assert dataset.get_batch(0)[0].group_size == 4
    assert dataset.get_batch(0)[0] is not dataset.get_batch(1)[0]
    with pytest.raises(IndexError):
        dataset.get_batch(2)


@pytest.mark.parametrize(
    "changes",
    [
        {"group_size": 1},
        {"max_constant_group_retries": 3},
        {"learning_rate": 0},
        {"max_sampled_tokens": 0},
    ],
)
@pytest.mark.asyncio
async def test_invalid_training_configuration_fails_before_clients(config, changes):
    with pytest.raises(ValueError):
        await train(replace(config, **changes))
