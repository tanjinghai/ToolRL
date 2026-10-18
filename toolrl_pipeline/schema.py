"""Conservative schema extraction and JSON type validation; never coerce labels."""
import ast
import re

from .io import canonical, loads

ALIASES = {"str": "string", "string": "string", "int": "integer", "integer": "integer",
           "float": "number", "number": "number", "bool": "boolean", "boolean": "boolean",
           "dict": "object", "Dict": "object", "object": "object", "list": "array", "List": "array",
           "array": "array", "null": "null", "None": "null", "NoneType": "null"}


def annotation_schema(value):
    if isinstance(value, list):
        return {"anyOf": [annotation_schema(v) for v in value]}
    if not isinstance(value, str):
        return {"x-unsupported-type": str(value)}
    value = re.split(r",\s*(?:optional\b|default\s*=)", value, maxsplit=1)[0].strip()
    if value in ALIASES:
        return {"type": ALIASES[value]}
    try:
        node = ast.parse(value, mode="eval").body
    except SyntaxError:
        return {"x-unsupported-type": value}

    def convert(node):
        if isinstance(node, ast.Name):
            return {"type": ALIASES[node.id]} if node.id in ALIASES else {"x-unsupported-type": node.id}
        if isinstance(node, ast.Constant) and node.value is None:
            return {"type": "null"}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'list' and len(node.args) == 1 and not node.keywords:
            return {'type': 'array', 'items': convert(node.args[0])}
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            return {"x-unsupported-type": value}
        name, arg = node.value.id, node.slice
        args = arg.elts if isinstance(arg, ast.Tuple) else [arg]
        if name in ("List", "list"):
            return {"type": "array", "items": convert(args[0])}
        if name in ("Tuple", "tuple"):
            return {"type": "array", "prefixItems": [convert(a) for a in args], "minItems": len(args), "maxItems": len(args)}
        if name in ("Union", "Optional"):
            options = [convert(a) for a in args]
            if name == "Optional":
                options.append({"type": "null"})
            return {"anyOf": options}
        if name in ("Dict", "dict") and len(args) == 2 and isinstance(args[0], ast.Name) and args[0].id in ("str", "string"):
            return {"type": "object", "additionalProperties": convert(args[1])}
        return {"x-unsupported-type": value}
    return convert(node)


def normalize_schema(raw):
    if not isinstance(raw, dict):
        return {"x-unsupported-type": "non-object parameter declaration"}
    out = annotation_schema(raw.get("type", "unknown"))
    out["x-original-type"] = raw.get("type")
    for key in ("description", "default", "enum", "required", "minItems", "maxItems", "minimum", "maximum"):
        if key in raw and (key != "required" or isinstance(raw[key], list)):
            out[key] = raw[key]
    if "items" in raw:
        out["items"] = normalize_schema(raw["items"])
    if "properties" in raw:
        out["properties"] = {k: normalize_schema(v) for k, v in raw["properties"].items()}
        out["additionalProperties"] = raw.get("additionalProperties", True)
    return out


def extract_tools(instruction):
    """Original ToolRL/API-Bank text schema. Fail closed if any entry is missing."""
    start = instruction.find("**Available Tools**")
    end = instruction.find("**Steps for Each Turn**", start)
    if start < 0 or end < 0:
        return [], ["unparseable_tool_section"], []
    section = instruction[start:end]
    entries = re.findall(r"^\d+\. (.*?)(?=^\d+\. |\Z)", section, re.S | re.M)
    tools, errors, repairs = [], [], []
    seen = {}
    for entry in entries:
        match = re.fullmatch(r"Name: (.*?)\nDescription: (.*?)\nParameters: (.*?)\s*", entry, re.S)
        # API-Bank level 3 uses '1. ToolName: Description' rather than Name/Description.
        if not match:
            match = re.fullmatch(r"([^\n:]+): (.*?)\nParameters: (.*?)\s*", entry, re.S)
        if not match:
            errors.append("unparseable_tool_definition")
            continue
        name, description, literal = match.groups()
        try:
            try:
                params = loads(literal)
            except ValueError:
                params = ast.literal_eval(literal)
            if not isinstance(params, dict) or not name.strip():
                raise ValueError("invalid declaration")
            tool = {"name": name, "description": description,
                    "parameters": {"type": "object", "properties": {k: normalize_schema(v) for k, v in params.items()},
                                   "required": [k for k, v in params.items() if isinstance(v, dict) and v.get("required") is True], "additionalProperties": False},
                    "required_status": "not_provided_in_source"}
        except (ValueError, SyntaxError, TypeError):
            errors.append("unparseable_parameter_schema")
            continue
        if name in seen:
            if canonical(seen[name]) == canonical(tool):
                repairs.append("deduplicated_identical_schema_metadata")
            else:
                errors.append("conflicting_duplicate_tool_name")
        else:
            seen[name] = tool
            tools.append(tool)
    if not entries or len(entries) != len(re.findall(r"^\d+\. ", section, re.M)):
        errors.append("incomplete_tool_definitions")
    return tools, sorted(set(errors)), repairs


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def type_matches(value, kind):
    return {"string": lambda: isinstance(value, str), "integer": lambda: type(value) is int,
            "number": lambda: is_number(value), "boolean": lambda: type(value) is bool,
            "array": lambda: isinstance(value, list), "object": lambda: isinstance(value, dict),
            "null": lambda: value is None}.get(kind, lambda: False)()


def validate_value(value, schema, path=""):
    errors = []
    if "x-unsupported-type" in schema:
        return [("unsupported_schema", path)]
    if "anyOf" in schema:
        if not any(not validate_value(value, choice, path) for choice in schema["anyOf"]):
            return [("wrong_argument_type", path)]
    elif "type" in schema and not type_matches(value, schema["type"]):
        return [("wrong_argument_type", path)]
    if "enum" in schema and not any(strict_equal(value, v, schema) for v in schema["enum"]):
        errors.append(("wrong_argument_value", path))
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(("missing_argument", f"{path}/{key}"))
        for key, val in value.items():
            if key in props:
                errors += validate_value(val, props[key], f"{path}/{key}")
            elif schema.get("additionalProperties", True) is False:
                errors.append(("extra_argument", f"{path}/{key}"))
            elif isinstance(schema.get("additionalProperties"), dict):
                errors += validate_value(val, schema["additionalProperties"], f"{path}/{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            errors.append(("wrong_argument_value", path))
        for i, val in enumerate(value):
            child = schema.get("items", {})
            if "prefixItems" in schema and i < len(schema["prefixItems"]):
                child = schema["prefixItems"][i]
            errors += validate_value(val, child, f"{path}/{i}")
    if is_number(value):
        if value < schema.get("minimum", -float("inf")) or value > schema.get("maximum", float("inf")):
            errors.append(("wrong_argument_value", path))
    return errors


def strict_equal(a, b, schema=None):
    schema = schema or {}
    # Numeric int/float equivalence is permitted only under an explicit number schema.
    if schema.get("type") == "number" and is_number(a) and is_number(b):
        return a == b
    if "anyOf" in schema:
        return any(not validate_value(a, child) and not validate_value(b, child) and strict_equal(a, b, child)
                   for child in schema["anyOf"])
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        fallback = schema.get('additionalProperties', {})
        fallback = fallback if isinstance(fallback, dict) else {}
        return set(a) == set(b) and all(strict_equal(a[k], b[k], schema.get("properties", {}).get(k, fallback)) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y, (schema.get("prefixItems", []) + [{}]*len(a))[i]
                                                      if "prefixItems" in schema else schema.get("items", {}))
                                      for i, (x, y) in enumerate(zip(a, b)))
    return a == b


def validate_calls(calls, tools):
    errors = []
    if not isinstance(tools, list) or any(not isinstance(t, dict) or not isinstance(t.get('name'), str)
            or not isinstance(t.get('parameters'), dict) for t in tools):
        return [('invalid_tool_schema', 'tool_declaration')]
    if any(t['parameters'].get('type') != 'object' or not schema_structure_valid(t['parameters']) for t in tools):
        return [('invalid_tool_schema', 'parameter_declaration')]
    names = {tool["name"]: tool for tool in tools}
    if len(names) != len(tools):
        errors.append(("invalid_tool_schema", "duplicate_name"))
    for i, call in enumerate(calls):
        if (not isinstance(call, dict) or set(call) != {"name", "parameters"}
                or not isinstance(call.get("name"), str) or not isinstance(call.get("parameters"), dict)):
            errors.append(("invalid_call_shape", str(i)))
            continue
        if call["name"] not in names:
            errors.append(("wrong_tool", str(i)))
            continue
        errors += validate_value(call["parameters"], names[call["name"]]["parameters"], str(i))
    return errors


def schema_structure_valid(schema):
    """Validate the supported declaration subset before accessing nested fields."""
    if not isinstance(schema, dict):
        return False
    if 'type' in schema and schema['type'] not in set(ALIASES.values()):
        return False
    if 'anyOf' in schema and (not isinstance(schema['anyOf'], list) or not schema['anyOf'] or
                              not all(schema_structure_valid(s) for s in schema['anyOf'])):
        return False
    if 'properties' in schema and (not isinstance(schema['properties'], dict) or
                                   not all(isinstance(k, str) and schema_structure_valid(v) for k, v in schema['properties'].items())):
        return False
    if 'required' in schema and (not isinstance(schema['required'], list) or not all(isinstance(k, str) for k in schema['required'])):
        return False
    if 'items' in schema and not schema_structure_valid(schema['items']):
        return False
    if 'prefixItems' in schema and (not isinstance(schema['prefixItems'], list) or not all(schema_structure_valid(s) for s in schema['prefixItems'])):
        return False
    if 'enum' in schema and not isinstance(schema['enum'], list):
        return False
    if 'additionalProperties' in schema and type(schema['additionalProperties']) is not bool and not schema_structure_valid(schema['additionalProperties']):
        return False
    for key in ('minItems', 'maxItems'):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            return False
    for key in ('minimum', 'maximum'):
        if key in schema and not is_number(schema[key]):
            return False
    return True
