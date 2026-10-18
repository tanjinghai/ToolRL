"""Parse ToolRL completions without using the training Reward parser."""
from dataclasses import dataclass, field
import re

from .io import loads

MAX_CALLS = 64


@dataclass
class Parsed:
    calls: list = field(default_factory=list)
    protocol_valid: bool = False
    json_valid: bool = False
    shape_valid: bool = False
    response: str = ""
    think: str = ""
    errors: list = field(default_factory=list)


def parse_completion(text):
    result = Parsed()
    if not isinstance(text, str) or not text.strip():
        result.errors.append("empty_output")
        return result
    text = text.strip()
    # Delimit thought first: literal tool/response tags inside it are not actions.
    match = re.fullmatch(r"<think>(.*?)</think>\n(.*)", text, re.S)
    if not match:
        result.errors.append("invalid_protocol")
        return result
    result.think, body = match.groups()
    if "</think>" in body or "<think>" in body:
        result.errors.append("ambiguous_think_delimiter")
        return result
    response_only = re.fullmatch(r"<response>(.*?)</response>", body, re.S)
    if response_only:
        if any(tag in response_only[1] for tag in ('<response>', '</response>', '<tool_call>', '</tool_call>')):
            result.errors.append('multiple_or_nested_response_blocks')
            return result
        result.response = response_only[1]
        result.protocol_valid = result.json_valid = result.shape_valid = True
        return result
    tool_match = re.fullmatch(r"<tool_call>\n(.*?)\n</tool_call>(?:\n<response>(.*?)</response>)?", body, re.S)
    if not tool_match:
        result.errors.append("invalid_protocol")
        return result
    payload, response = tool_match.groups()
    if response is not None and any(tag in response for tag in ('<response>', '</response>', '<tool_call>', '</tool_call>')):
        result.errors.append('multiple_or_nested_response_blocks')
        return result
    if any(tag in payload for tag in ("<tool_call>", "</tool_call>", "<response>", "</response>")):
        result.errors.append("multiple_or_nested_call_blocks")
        return result
    result.protocol_valid = True
    result.response = response or ""
    lines = [line for line in payload.splitlines() if line.strip()]
    if not lines or len(lines) > MAX_CALLS:
        result.errors.append("empty_or_excessive_call_block")
        return result
    result.json_valid = True
    result.shape_valid = True
    for i, line in enumerate(lines):
        try:
            call = loads(line)
        except (ValueError, TypeError) as error:
            result.json_valid = False
            result.shape_valid = False
            result.errors.append(f"invalid_json_line:{i}:{type(error).__name__}")
            continue
        if (not isinstance(call, dict) or set(call) != {"name", "parameters"}
                or not isinstance(call["name"], str) or not call["name"].strip()
                or not isinstance(call["parameters"], dict)):
            result.shape_valid = False
            result.errors.append(f"invalid_call_shape:{i}")
            continue
        result.calls.append(call)
    return result


def render_completion(calls, response="", think="Mock reasoning."):
    """Fixture/mock helper. Not a repair function for real predictions."""
    from .io import canonical
    text = f"<think>{think}</think>"
    if calls:
        text += "\n<tool_call>\n" + "\n".join(canonical(c) for c in calls) + "\n</tool_call>"
        if response:
            text += f"\n<response>{response}</response>"
    else:
        text += f"\n<response>{response or 'No tool is needed.'}</response>"
    return text
