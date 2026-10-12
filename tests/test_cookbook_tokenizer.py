"""Actual pinned Qwen tool framing and token types, with local assets only."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from transformers import AutoTokenizer

from elt_bench_rl.cookbook_env import ELTCookbookEnv


@pytest.fixture(scope="module")
def qwen_renderer():
    path = Path(__file__).resolve().parents[1] / "data/tokenizers/Qwen3.5-4B"
    if not (path / "tokenizer_config.json").is_file():
        pytest.skip(
            "Fetch pinned local tokenizer assets before this optional compatibility test"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    return get_renderer(get_recommended_renderer_name("Qwen/Qwen3.5-4B"), tokenizer)


def xml_action(renderer, name, arguments):
    parameters = "".join(
        f"<parameter={key}>\n{json.dumps(value)}\n</parameter>\n"
        for key, value in arguments.items()
    )
    response = (
        f"\n</think>\n\n<tool_call>\n<function={name}>\n{parameters}"
        "</function>\n</tool_call><|im_end|>"
    )
    return renderer.tokenizer.encode(response, add_special_tokens=False)


@pytest.mark.asyncio
async def test_actual_trains_tool_prompt_matches_hf_integer_tokens(
    qwen_renderer, episode
):
    env = ELTCookbookEnv(episode, qwen_renderer)
    observation, _stop = await env.initial_observation()
    tokens = observation.to_ints()
    assert tokens and all(type(token) is int for token in tokens)
    assert observation.length < 32768
    template_tokens = qwen_renderer.tokenizer.apply_chat_template(
        env.message_env.history,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    assert tokens == template_tokens
    # Transformers 5 can return BatchEncoding by default, rather than list[int].
    default = qwen_renderer.tokenizer.apply_chat_template(
        env.message_env.history, tokenize=True, add_generation_prompt=True
    )
    assert (default["input_ids"] if isinstance(default, Mapping) else default) == tokens
    prompt = qwen_renderer.tokenizer.decode(tokens)
    assert "<function=example_function_name>" in prompt
    for spec in episode.tools_schema():
        assert spec["name"] in prompt
    assert not re.search(r"[\u3400-\u9fff]", prompt)
    assert "ground_truth" not in prompt and str(episode.reference_root) not in prompt
    await env.close()


@pytest.mark.asyncio
async def test_actual_qwen_xml_tools_inspect_configure_and_load_empty_episode(
    qwen_renderer, episode
):
    env = ELTCookbookEnv(episode, qwen_renderer)
    await env.initial_observation()
    assert episode.adapter.list_tables() == []
    for name, arguments in [
        ("inspect_source", {"source": "cars"}),
        ("configure_load", {"source": "cars"}),
        ("execute_load", {"source": "cars"}),
        ("query", {"sql": "SELECT id FROM raw.cars ORDER BY id"}),
    ]:
        result = await env.step(xml_action(qwen_renderer, name, arguments))
        assert not result.episode_done
        assert all(type(token) is int for token in result.next_observation.to_ints())
        feedback = json.loads(env.message_env.history[-1]["content"])
        assert feedback["ok"]
    assert [entry["tool"] for entry in episode.trace] == [
        "inspect_source",
        "configure_load",
        "execute_load",
        "query",
    ]
    assert set(episode.receipts) == {"cars"}
    assert len(episode.adapter.table("cars")) == 5
    await env.close()


@pytest.mark.asyncio
async def test_actual_qwen_text_answer_cannot_replace_submit(qwen_renderer, episode):
    env = ELTCookbookEnv(episode, qwen_renderer)
    await env.initial_observation()
    action = qwen_renderer.tokenizer.encode(
        "\n</think>\n\nThe pipeline is complete.<|im_end|>", add_special_tokens=False
    )
    result = await env.step(action)
    assert result.episode_done and result.reward == 0
    assert not episode.submitted and not episode.receipts
    await env.close()
