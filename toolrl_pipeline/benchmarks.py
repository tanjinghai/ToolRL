"""Benchmark adapters. No external service or model is imported here."""
import json
from pathlib import Path

from .io import digest, file_hash
from .protocol import parse_completion, render_completion
from .schema import extract_tools, validate_calls

BFCL_COMMIT = "a84eb805fe8d66635a8e402a78783f13044decba"


def api_bank_records(root):
    paths = sorted((Path(root)/"benchmarks/API-Bank").glob("level-*-api_processed.json"))
    hashes = {p.name: file_hash(p) for p in paths}
    version = "apibank-local-" + digest({'sources': hashes, 'adapter': file_hash(__file__),
                                       'schema': file_hash(Path(__file__).with_name('schema.py'))})[:12]
    records = []
    for path in paths:
        level = path.name.split("-")[1]
        for index, raw in enumerate(json.loads(path.read_text())):
            gt = raw["answer"] if isinstance(raw["answer"], list) else [raw["answer"]]
            tools, errors, _ = extract_tools(raw["system"])
            issues = errors + [code for code, _ in validate_calls(gt, tools)]
            sid = "api_" + digest([path.name, index, raw])
            records.append({
                "sample_id": sid, "dataset_version": version, "source": "api_bank", "split": "benchmark",
                "group_id": sid, "prompt": [{"role": "system", "content": raw["system"]}, {"role": "user", "content": raw["user"]}],
                "tools": tools, "ground_truth": gt, "target_output": render_completion(gt),
                "metadata": {"benchmark": "API-Bank", "level": level, "legacy_key": f"Level{level}_{index}",
                             "source_file": str(path.relative_to(root)), "source_file_sha256": hashes[path.name],
                             "raw_index": index, "raw_record_sha256": digest(raw), "call_order": "parallel",
                             "order_evidence": "local processed benchmark target contains one next-step call",
                             "task_type": "single" if len(gt) == 1 else "multi", "call_count": len(gt),
                             "parameter_count": sum(len(c.get("parameters", {})) for c in gt),
                             "ground_truth_issues": sorted(set(issues))}})
    if len(records) != 597:
        raise ValueError(f"Local API-Bank version changed: expected 597 rows, got {len(records)}")
    return records, hashes


def legacy_soft_match(gt, text):
    """The repository's permissive single-target score, explicitly labeled legacy.

    Reproduces ignored malformed lines, inferred tool name and extra-call tolerance.
    Its purpose is comparison, not certification of task execution.
    """
    if not gt:
        return False
    answer = gt[0]
    payload = text.split("<tool_call>")[-1].split("</tool_call>")[0].strip()
    for line in payload.splitlines():
        try:
            prediction = json.loads(line)
            if isinstance(prediction, str):
                prediction = json.loads(prediction)
            if "name" not in prediction or "parameters" not in prediction:
                name, parameters = answer["name"], prediction
            else:
                name, parameters = prediction["name"], prediction["parameters"]
            if name == answer["name"] and parameters == answer["parameters"]:
                return True
        except (ValueError, TypeError, KeyError):
            continue
    return False


def bfcl_decode_ast(text):
    """Strict ToolRL→BFCL call bridge for the pinned OSS handler contract."""
    parsed = parse_completion(text)
    if not (parsed.protocol_valid and parsed.json_valid and parsed.shape_valid):
        raise ValueError("BFCL bridge refuses malformed completions; no bad line is silently dropped")
    return [{call["name"]: call["parameters"]} for call in parsed.calls]


def bfcl_decode_execute(text):
    parsed = parse_completion(text)
    bfcl_decode_ast(text)  # validate the complete output before translating any call
    return [f"{call['name']}({', '.join(k+'='+repr(v) for k, v in call['parameters'].items())})" for call in parsed.calls]


def benchmark_questions(root, bfcl_dir=None):
    """Text-only contamination index; not an official benchmark scorer."""
    result = []
    paths = sorted((Path(root)/"benchmarks/API-Bank").glob("*processed.json"))
    for path in paths:
        for i, row in enumerate(json.loads(path.read_text())):
            result.append({"id": f"{path.name}:{i}", "text": row["user"], "source": "API-Bank"})
    for i, row in enumerate(json.loads((Path(root)/"benchmarks/Bamboogle/data.json").read_text())):
        result.append({"id": f"Bamboogle:{i}", "text": row["Question"], "source": "Bamboogle"})
    if bfcl_dir is not None:
        for path in sorted(Path(bfcl_dir).glob("BFCL_v3_*.json")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)

                def collect(value):
                    if isinstance(value, dict):
                        if value.get("role") == "user" and isinstance(value.get("content"), str):
                            yield value["content"]
                        else:
                            for child in value.values():
                                yield from collect(child)
                    elif isinstance(value, list):
                        for child in value:
                            yield from collect(child)
                for j, text in enumerate(collect(row.get("question", []))):
                    result.append({"id": f"{path.name}:{row.get('id')}:{j}", "text": text, "source": "BFCL-v3"})
    return result
