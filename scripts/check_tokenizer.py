"""Check the pinned local tokenizer and cookbook tool renderer without API calls."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from fetch_tokenizer import MODEL_ID, REVISION, TOKENIZER_DIRECTORY, verify_files


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    verify_files(project_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import AutoTokenizer
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.renderers.base import ToolCall

    tokenizer = AutoTokenizer.from_pretrained(
        project_dir / TOKENIZER_DIRECTORY, local_files_only=True, trust_remote_code=False,
    )
    renderer_name = get_recommended_renderer_name(MODEL_ID)
    renderer = get_renderer(renderer_name, tokenizer)
    tools = [{
        "name": "run_sql",
        "description": "Execute SQL in the isolated task warehouse.",
        "parameters": {
            "type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"],
        },
    }]
    user_messages = [{"role": "user", "content": "Inspect the loaded source tables."}]
    conversation = renderer.create_conversation_prefix_with_tools(tools, "") + user_messages
    prompt = renderer.build_generation_prompt(conversation)
    hf_tokens = tokenizer.apply_chat_template(
        user_messages, tools=tools, tokenize=True, add_generation_prompt=True,
        enable_thinking="disable_thinking" not in renderer_name,
    )
    if isinstance(hf_tokens, Mapping):
        hf_tokens = hf_tokens["input_ids"]
    if hasattr(hf_tokens, "tolist"):
        hf_tokens = hf_tokens.tolist()
    _require(prompt.to_ints() == hf_tokens, "Cookbook and Hugging Face prompt templates differ")

    call = ToolCall(
        id="smoke-call",
        function=ToolCall.FunctionBody(name="run_sql", arguments='{"sql": "SELECT 1"}'),
    )
    rendered, weights = renderer.build_supervised_example(
        conversation + [{"role": "assistant", "content": "", "tool_calls": [call]}]
    )
    rendered_text = tokenizer.decode(rendered.to_ints())
    _require("<function=run_sql>" in rendered_text and "SELECT 1" in rendered_text,
             "The rendered tool call does not contain the expected function and SQL")
    response = (
        "\n</think>\n\n<tool_call>\n<function=run_sql>\n"
        "<parameter=sql>\nSELECT 1\n</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    parsed, termination = renderer.parse_response(tokenizer.encode(response, add_special_tokens=False))
    _require(termination.is_clean, "Tool-call parsing did not terminate cleanly")
    parsed_call = parsed["tool_calls"][0]
    _require(parsed_call.function.name == "run_sql", "Parsed tool name differs")
    _require(json.loads(parsed_call.function.arguments) == {"sql": "SELECT 1"}, "Parsed SQL differs")
    _require(float(weights.sum()) > 0, "The rendered assistant tokens have no training weight")
    print(json.dumps({
        "status": "passed", "model_id": MODEL_ID, "revision": REVISION,
        "renderer": renderer_name, "prompt_tokens": prompt.length,
        "rendered_tool_call_tokens": rendered.length,
        "offline": True, "tinker_api_called": False,
    }, indent=2))


if __name__ == "__main__":
    main()
