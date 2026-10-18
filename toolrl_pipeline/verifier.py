"""Independent structural verifier; scores are booleans, never training rewards."""
from collections import Counter

from .protocol import MAX_CALLS, parse_completion
from .schema import strict_equal, validate_calls, validate_value

ERROR_TYPES = (
    "format_error", "wrong_tool", "missing_tool", "extra_tool", "missing_argument",
    "extra_argument", "wrong_argument_value", "wrong_argument_type", "unnecessary_tool",
    "truncated_output", "invalid_ground_truth", "wrong_call_order",
)


def _perfect_match(gt, prediction, predicate, order):
    if len(gt) != len(prediction):
        return False
    if order == "sequential":
        return all(predicate(g, p) for g, p in zip(gt, prediction))
    matched = {}

    def augment(i, visited):
        for j, p in enumerate(prediction):
            if j in visited or not predicate(gt[i], p):
                continue
            visited.add(j)
            if j not in matched or augment(matched[j], visited):
                matched[j] = i
                return True
        return False
    return all(augment(i, set()) for i in range(len(gt)))


def _best_assignment(weights):
    """Exact maximum-weight bipartite assignment (Hungarian), including dummy slots."""
    if not weights:
        return []
    n = len(weights)
    u, v, p, way = [0]*(n+1), [0]*(n+1), [0]*(n+1), [0]*(n+1)
    maximum = max(max(row) for row in weights)
    for i in range(1, n+1):
        p[0] = i
        j0, minv, used = 0, [float("inf")]* (n+1), [False]*(n+1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, n+1):
                if used[j]:
                    continue
                cur = maximum - weights[i0-1][j-1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(n+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sorted((p[j]-1, j-1) for j in range(1, n+1))


def _diagnostic_pairs(gt, prediction, schemas, order):
    if order == "sequential":
        return list(zip(range(len(gt)), range(len(prediction))))
    n = max(len(gt), len(prediction))
    if not n:
        return []
    weights = [[0]*n for _ in range(n)]
    for i, g in enumerate(gt):
        for j, pred in enumerate(prediction):
            if g["name"] != pred["name"]:
                continue
            a, b = g["parameters"], pred["parameters"]
            props = schemas[g["name"]]["parameters"].get("properties", {})
            value_count = sum(strict_equal(a[k], b[k], props.get(k)) for k in a.keys() & b.keys())
            weights[i][j] = 1000000 + 10000*int(set(a) == set(b)) + 100*value_count + len(a.keys() & b.keys())
    return [(i, j) for i, j in _best_assignment(weights)
            if i < len(gt) and j < len(prediction) and gt[i]["name"] == prediction[j]["name"]]


def verify(ground_truth, prediction, tools, metadata=None):
    """GT is a call list; prediction is raw ToolRL text or a prediction record.

    Unknown multi-call order is invalid GT, rather than silently treating it as parallel.
    A free-text answer is never assigned a semantic correctness score.
    """
    metadata = metadata or {}
    if isinstance(prediction, str):
        prediction = {"raw_output": prediction, "status": "ok", "finish_reason": "stop"}
    text = prediction.get("raw_output", "")
    parsed = parse_completion(text)
    errors, details = set(), []
    if prediction.get("finish_reason") in ("length", "max_tokens"):
        errors.add("truncated_output")
    generation_ok = prediction.get("status", "ok") == "ok" and "truncated_output" not in errors and prediction.get('finish_reason') not in ('error', 'abort')
    structure_ok = parsed.protocol_valid and parsed.json_valid and parsed.shape_valid
    pred_errors = validate_calls(parsed.calls, tools) if parsed.shape_valid else []
    schema_valid = structure_ok and not pred_errors
    if not structure_ok or not schema_valid or not generation_ok:
        errors.add("format_error")
    gt_valid = isinstance(ground_truth, list)
    gt_errors = validate_calls(ground_truth, tools) if gt_valid else [("invalid_call_shape", "ground_truth")]
    order = metadata.get("call_order", "parallel" if isinstance(ground_truth, list) and len(ground_truth) <= 1 else "unknown")
    if gt_errors or metadata.get('ground_truth_issues') or (isinstance(ground_truth, list) and (
            len(ground_truth) > MAX_CALLS or len(ground_truth) > 1 and order not in ("parallel", "sequential"))):
        gt_valid = False
        errors.add("invalid_ground_truth")
    tool_sample = isinstance(ground_truth, list) and bool(ground_truth)
    no_tool_sample = isinstance(ground_truth, list) and not ground_truth
    metrics = {
        "format_validity": bool(schema_valid and generation_ok),
        "protocol_validity": bool(parsed.protocol_valid and generation_ok),
        "json_validity": bool(parsed.json_valid and generation_ok),
        "schema_validity": bool(schema_valid and generation_ok),
        "tool_name_accuracy": None if not tool_sample else False,
        "parameter_name_accuracy": None if not tool_sample else False,
        "parameter_value_accuracy": None if not tool_sample else False,
        "exact_match_accuracy": False,
        "no_tool_accuracy": False if no_tool_sample else None,
        "parameter_value_accuracy_given_names": None,
    }
    for code, path in pred_errors:
        if code in ERROR_TYPES:
            errors.add(code)
        details.append({"error": code, "path": path})
    if not gt_valid:
        return {"metrics": metrics, "errors": sorted(errors), "details": details,
                "ground_truth_errors": gt_errors, "parse_errors": parsed.errors,
                "parsed_calls": parsed.calls, "ground_truth_valid": False,
                "free_text_semantic_accuracy": None}
    gt, pd = ground_truth, parsed.calls
    schemas = {t["name"]: t for t in tools}
    gt_names, pd_names = Counter(c["name"] for c in gt), Counter(c["name"] for c in pd)
    if gt_names - pd_names:
        errors.add("missing_tool")
    if pd_names - gt_names:
        errors.add("extra_tool")
    if gt_names - pd_names and pd_names - gt_names:
        errors.add("wrong_tool")
    if not gt and pd:
        errors.add("unnecessary_tool")

    def names_equal(g, pred):
        return g["name"] == pred["name"]

    def keys_equal(g, pred):
        return names_equal(g, pred) and set(g["parameters"]) == set(pred["parameters"])

    def values_equal(g, pred):
        if not keys_equal(g, pred):
            return False
        schema = schemas[g["name"]]["parameters"]
        return not validate_value(pred["parameters"], schema) and strict_equal(g["parameters"], pred["parameters"], schema)

    content_ok = structure_ok and generation_ok
    if gt:
        metrics["tool_name_accuracy"] = bool(content_ok and gt_names == pd_names)
        metrics["parameter_name_accuracy"] = bool(content_ok and _perfect_match(gt, pd, keys_equal, "parallel"))
        metrics["parameter_value_accuracy"] = bool(content_ok and _perfect_match(gt, pd, values_equal, order))
        if metrics["parameter_name_accuracy"]:
            metrics["parameter_value_accuracy_given_names"] = metrics["parameter_value_accuracy"]
        metrics["exact_match_accuracy"] = metrics["parameter_value_accuracy"]
        if order == "sequential" and content_ok and not metrics["exact_match_accuracy"] and _perfect_match(gt, pd, values_equal, "parallel"):
            errors.add("wrong_call_order")
    else:
        metrics["no_tool_accuracy"] = bool(content_ok and not pd)
        metrics["exact_match_accuracy"] = metrics["no_tool_accuracy"]
    for i, j in _diagnostic_pairs(gt, pd, schemas, order):
        a, b = gt[i]["parameters"], pd[j]["parameters"]
        if a.keys() - b.keys():
            errors.add("missing_argument")
        if b.keys() - a.keys():
            errors.add("extra_argument")
        props = schemas[gt[i]["name"]]["parameters"].get("properties", {})
        for key in a.keys() & b.keys():
            schema = props.get(key, {})
            value_errors = validate_value(b[key], schema, key)
            if any(code == "wrong_argument_type" for code, _ in value_errors):
                errors.add("wrong_argument_type")
            elif not strict_equal(a[key], b[key], schema):
                errors.add("wrong_argument_value")
    if gt and content_ok and not metrics["exact_match_accuracy"] and not errors:
        # Same calls in the wrong sequential order, including same-name repeated calls.
        errors.add("wrong_argument_value" if gt_names == pd_names else "wrong_tool")
    return {"metrics": metrics, "errors": sorted(errors), "details": details,
            "parse_errors": parsed.errors, "parsed_calls": parsed.calls,
            "ground_truth_valid": True, "free_text_semantic_accuracy": None}
