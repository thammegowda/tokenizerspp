"""Compare local model chat templates with Hugging Face's Jinja environment."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from jinja2 import TemplateError
from transformers.utils.chat_template_utils import _compile_jinja_template


def conversation_cases():
    function = {
        "name": "weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "options": {"type": "object"}},
            "required": ["city"],
        },
    }
    tools = [{"type": "function", "function": function}]
    call = {
        "id": "c1",
        "type": "function",
        "function": {
            "name": "weather",
            "arguments": {"city": "Paris", "options": {"z": 1, "a": "x,:y", "A": True}},
        },
    }
    user = {"role": "user", "content": "What is the weather?"}
    assistant = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Check Paris.",
        "tool_calls": [call],
    }
    result = {"role": "tool", "tool_call_id": "c1", "name": "weather", "content": "20 C"}
    second_call = {
        **call, "id": "c2",
        "function": {**call["function"], "arguments": {"city": "Tokyo", "days": [1, 2]}},
    }
    string_call = {
        **call, "function": {**call["function"], "arguments": json.dumps(call["function"]["arguments"])},
    }
    return [
        ("plain", [user], {}),
        ("system", [{"role": "system", "content": "Be brief."}, user], {}),
        ("tools", [user], {"tools": tools}),
        ("roundtrip", [user, assistant, result], {"tools": tools}),
        ("parallel", [user, {**assistant, "tool_calls": [call, second_call]},
                      result, {**result, "tool_call_id": "c2", "content": "25 C"}], {"tools": tools}),
        ("string-arguments", [user, {**assistant, "tool_calls": [string_call]}, result], {"tools": tools}),
        ("final", [user, assistant, result, {"role": "assistant", "content": "20 C."}], {"tools": tools}),
        ("reasoning", [user], {"reasoning_effort": "low", "enable_thinking": False}),
        ("numeric-effort", [user], {"reasoning_effort": 0.123456789}),
        ("invalid-effort", [user], {"reasoning_effort": "invalid"}),
        ("multiturn", [user, {"role": "assistant", "content": "20 C."}, user], {}),
        ("empty-content", [{**user, "content": ""}], {}),
    ]


def load_template(path):
    if path.suffix != ".json":
        return path.read_text(), {}
    config = json.loads(path.read_text())
    template = config.get("chat_template")
    if template is None:
        template = path.with_name("chat_template.jinja").read_text()
    if not isinstance(template, str):
        raise ValueError(f"{path}: select a single template and pass it as a .jinja file")
    tokens = {}
    for key in ("bos_token", "eos_token"):
        token = config.get(key)
        if token is not None:
            tokens[key] = token if isinstance(token, str) else token["content"]
    return template, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--template", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    requests, expected, labels = [], [], []
    for path in args.template:
        template, tokens = load_template(path)
        compiled = _compile_jinja_template(template)
        for name, messages, variables in conversation_cases():
            for prompt in (False, True):
                labels.append({
                    "template": str(path),
                    "sha256": hashlib.sha256(template.encode()).hexdigest(),
                    "case": name,
                    "add_generation_prompt": prompt,
                })
                requests.append({
                    "template": template, "messages": messages,
                    "template_args": variables, "add_generation_prompt": prompt, **tokens,
                })
                try:
                    expected.append({"rendered": compiled.render(
                        messages=messages, add_generation_prompt=prompt, **tokens, **variables,
                    )})
                except (TemplateError, TypeError) as error:
                    expected.append({"error": str(error)})
    process = subprocess.run(
        [str(args.renderer.resolve())],
        input="".join(json.dumps(request) + "\n" for request in requests),
        text=True, capture_output=True, check=True, timeout=120,
    )
    actual = [json.loads(line) for line in process.stdout.splitlines()]
    if len(actual) != len(expected):
        raise RuntimeError(f"Renderer returned {len(actual)} responses for {len(expected)} requests")
    records = []
    for label, result, reference in zip(labels, actual, expected):
        if "rendered" in reference:
            passed = result == reference
        else:
            passed = "error" in result and result["error"].endswith(reference["error"])
        records.append({**label, "passed": passed, "actual": result, "expected": reference})
        if not passed:
            print(f"FAIL {label['template']} {label['case']} prompt={label['add_generation_prompt']}")
            print(f"  actual: {result!r}\n  expected: {reference!r}")
    if args.output:
        args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    passed = sum(record["passed"] for record in records)
    reference_errors = sum("error" in record["expected"] for record in records)
    print(f"Parity: {passed}/{len(records)} ({reference_errors} reference-error cases)")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
