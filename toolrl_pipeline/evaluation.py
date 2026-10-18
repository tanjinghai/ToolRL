"""Generation records, fixed-denominator scoring and offline bad-case reports."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import subprocess
import time

from .benchmarks import legacy_soft_match
from .io import canonical, digest, file_hash, read_jsonl, write_json, write_jsonl, write_text
from .protocol import render_completion
from .verifier import ERROR_TYPES, verify

PREDICTION_VERSION = "toolrl.prediction.v1"
DEFAULT_GENERATION = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 1024,
                      "seed": 31415, "n": 1, "stop": None}


def evaluator_fingerprint():
    directory = Path(__file__).parent
    return digest({name: file_hash(directory/name) for name in ('evaluation.py', 'verifier.py', 'schema.py', 'protocol.py', 'io.py',
                                                                'benchmarks.py', 'cli.py', 'generation_backends.py', 'tokenization.py')})


def make_run(dataset_path, run_id, model_id, model_revision, generation=None, backend="mock", backend_config=None):
    return {"prediction_schema_version": PREDICTION_VERSION, "run_id": run_id,
            "model": {"id": model_id, "revision": model_revision},
            "generation_config": generation or dict(DEFAULT_GENERATION),
            "dataset_file_sha256": file_hash(dataset_path), "backend": backend,
            "backend_config": backend_config or {},
            "is_mock": backend.startswith("mock"),
            "evaluator_git_sha": subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
            "evaluator_code_sha256": evaluator_fingerprint()}


def _check_prediction(record, run, expected):
    if record.get('prediction_schema_version') != PREDICTION_VERSION:
        raise ValueError('Unsupported prediction schema version')
    if not isinstance(record.get('backend'), str) or record.get('is_mock') != record['backend'].startswith('mock'):
        raise ValueError('Mock provenance must be explicit and consistent with backend')
    if record.get('generation_config', {}).get('n') != 1:
        raise ValueError('Evaluation requires one fixed attempt per sample (n=1)')
    sid = record.get("sample_id")
    if sid not in expected:
        raise ValueError(f"Unknown prediction ID: {sid}")
    for key in ("prediction_schema_version", "run_id", "model", "generation_config", "dataset_file_sha256", "backend", "backend_config", "is_mock", "evaluator_git_sha", "evaluator_code_sha256"):
        if record.get(key) != run.get(key):
            raise ValueError(f"Mixed/missing prediction provenance: {key}")
    if record.get("dataset_version") != expected[sid]["dataset_version"]:
        raise ValueError("Prediction dataset version mismatch")
    if not isinstance(record.get("raw_output"), str) or record.get("status") not in ("ok", "error", "missing"):
        raise ValueError("Invalid prediction raw_output/status")
    if record.get('finish_reason') not in ('stop', 'length', 'max_tokens', 'error', 'abort'):
        raise ValueError('Missing or unsupported finish_reason')


def generate(dataset_path, output_dir, run, backend, resume=False):
    """Call a supplied backend once per ID, preserving success AND failure on resume.

    Backend receives only sample_id/prompt/tools. A real backend never sees labels.
    Model/GPU backends are intentionally left to Phase 3.
    """
    records = read_jsonl(dataset_path)
    expected = {r["sample_id"]: r for r in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate dataset IDs")
    if run["dataset_file_sha256"] != file_hash(dataset_path):
        raise ValueError("Run/dataset hash mismatch")
    path = Path(output_dir)/"predictions.jsonl"
    previous = read_jsonl(path) if path.exists() else []
    if path.exists() and not resume:
        raise ValueError("Prediction file exists; use --resume to preserve all terminal attempts")
    seen = set()
    for prediction in previous:
        _check_prediction(prediction, run, expected)
        if prediction["sample_id"] in seen:
            raise ValueError("Duplicate prediction ID; retry selection is not permitted")
        seen.add(prediction["sample_id"])
    write_json(Path(output_dir)/"run.json", run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        for row in records:
            if row["sample_id"] in seen:
                continue
            started = time.perf_counter()
            try:
                result = backend({"sample_id": row["sample_id"], "prompt": row["prompt"], "tools": row["tools"]}, run["generation_config"])
                if not isinstance(result.get("raw_output"), str):
                    raise ValueError("Backend did not return raw_output text")
                result = {k: v for k, v in result.items() if k in ("raw_output", "finish_reason", "status", "prompt_tokens", "completion_tokens", "error")}
                result.setdefault("status", "ok")
                result.setdefault("finish_reason", "stop")
            except Exception as error:
                result = {"status": "error", "raw_output": "", "finish_reason": "error", "error": type(error).__name__+": "+str(error)[:500]}
            prediction = {**run, "sample_id": row["sample_id"], "dataset_version": row["dataset_version"],
                          "elapsed_seconds": round(time.perf_counter()-started, 6), **result}
            _check_prediction(prediction, run, expected)
            f.write(canonical(prediction)+'\n')
            f.flush()
    return path


def mock_backend(records, mode="mixed"):
    fixtures = {}
    for i, row in enumerate(records):
        calls = copy.deepcopy(row["ground_truth"])
        finish, status = "stop", "ok"
        text = row["target_output"]
        case = i % 8 if mode == "mixed" else 0 if mode == "oracle" else 3
        if case == 1:
            if calls and calls[0]["parameters"]:
                key = next(iter(calls[0]["parameters"]))
                calls[0]["parameters"][key] = {"mock_wrong_type": True}
            else:
                calls.append({"name": "unknown_mock_tool", "parameters": {}})
            text = render_completion(calls)
        elif case == 2:
            calls.append(copy.deepcopy(calls[0]) if calls else {"name": "unknown_mock_tool", "parameters": {}})
            text = render_completion(calls)
        elif case == 3:
            text = '<think>mock</think>\n<tool_call>\n{broken JSON}\n</tool_call>'
        elif case == 4:
            text = render_completion([])
        elif case == 5:
            finish = "length"
        elif case == 6:
            status, text, finish = "error", "", "error"
        elif case == 7:
            if calls and calls[0]["parameters"]:
                calls[0]["parameters"].pop(next(iter(calls[0]["parameters"])))
            text = render_completion(calls)
        fixtures[row["sample_id"]] = {"raw_output": text, "status": status, "finish_reason": finish}

    def backend(request, _config):
        assert set(request) == {"sample_id", "prompt", "tools"}
        return fixtures[request["sample_id"]]
    return backend


def _rate(values):
    values = [v for v in values if v is not None]
    numerator, denominator = sum(values), len(values)
    return {"numerator": numerator, "denominator": denominator, "rate": numerator/denominator if denominator else None}


def aggregate(rows):
    names = list(rows[0]["verification"]["metrics"]) if rows else []
    counts = Counter(error for r in rows for error in r["verification"]["errors"])
    return {"samples": len(rows), "metrics": {name: _rate([r["verification"]["metrics"][name] for r in rows]) for name in names},
            "errors": {name: {"count": counts[name], "rate": counts[name]/len(rows) if rows else None} for name in ERROR_TYPES},
            "invalid_ground_truth": sum(not r["verification"]["ground_truth_valid"] for r in rows),
            "legacy_api_bank_soft_match": _rate([r.get("legacy_api_bank_soft_match") for r in rows])}


def score(dataset_path, predictions_path, output_dir, previous_results=None, run_metadata=None, allow_final_test=False):
    records, predictions = read_jsonl(dataset_path), read_jsonl(predictions_path)
    dataset_hash = file_hash(dataset_path)
    if not records:
        raise ValueError("Empty dataset")
    if any(r["split"] == "test" for r in records) and not allow_final_test:
        raise ValueError("Frozen final test is gated; use validation during development and --allow-final-test only for final reporting")
    expected = {r["sample_id"]: r for r in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate dataset IDs")
    run = run_metadata or (predictions[0] if predictions else None)
    if run is None:
        raise ValueError("Empty predictions require explicit run metadata")
    if run.get("dataset_file_sha256") != dataset_hash:
        raise ValueError("Prediction dataset hash mismatch")
    if not run.get("model", {}).get("id") or not run.get("model", {}).get("revision"):
        raise ValueError("Model id/revision are required, including for mocks")
    by_id = {}
    for pred in predictions:
        _check_prediction(pred, run, expected)
        if pred["sample_id"] in by_id:
            raise ValueError("Duplicate prediction ID: do not select the successful retry")
        by_id[pred["sample_id"]] = pred
    scored = []
    for row in records:
        prediction = by_id.get(row["sample_id"], {"raw_output": "", "status": "missing", "finish_reason": "error"})
        result = verify(row["ground_truth"], prediction, row["tools"], row["metadata"])
        item = {"sample_id": row["sample_id"], "dataset_version": row["dataset_version"], "split": row["split"],
                "metadata": row["metadata"], "ground_truth": row["ground_truth"], "prediction": prediction,
                "dataset_file_sha256": dataset_hash,
                "verification": result}
        if row["source"] == "api_bank":
            item["legacy_api_bank_soft_match"] = bool(prediction.get("status") == "ok" and legacy_soft_match(row["ground_truth"], prediction["raw_output"]))
        scored.append(item)
    summary = aggregate(scored)
    summary.update({"prediction_schema_version": PREDICTION_VERSION, "run_id": run["run_id"], "is_mock": run.get("is_mock"),
                    "model": run["model"], "generation_config": run["generation_config"],
                    "backend_config": run.get('backend_config', {}),
                    "dataset_file_sha256": dataset_hash, "predictions_sha256": file_hash(predictions_path),
                    "dataset_versions": sorted({r["dataset_version"] for r in records}),
                    "scorer_code_sha256": evaluator_fingerprint(), "generation_evaluator_code_sha256": run.get('evaluator_code_sha256'),
                    "missing_predictions": len(expected)-len(by_id),
                    "generation_errors": sum(r["prediction"].get("status") == "error" for r in scored),
                    "free_text_semantic_accuracy": None, "task_execution_success_rate": None})
    summary['valid_ground_truth_subset'] = aggregate([r for r in scored if r['verification']['ground_truth_valid']])
    slices = defaultdict(list)
    for row in scored:
        meta = row["metadata"]
        slices['task:'+meta["task_type"]].append(row)
        calls, params = meta.get("call_count", 0), meta.get("parameter_count", 0)
        slices['calls:'+('0' if calls == 0 else '1' if calls == 1 else '2-4' if calls <= 4 else '5+')].append(row)
        slices['parameters:'+('0' if params == 0 else '1-2' if params <= 2 else '3-5' if params <= 5 else '6+')].append(row)
        if "level" in meta:
            slices['api_level:'+str(meta['level'])].append(row)
    summary["slices"] = {key: aggregate(rows) for key, rows in sorted(slices.items())}
    changes = None
    if previous_results:
        previous_rows = read_jsonl(previous_results)
        previous = {r["sample_id"]: r for r in previous_rows}
        if len(previous) != len(previous_rows) or set(previous) != set(expected) or any(
                previous[r['sample_id']]['dataset_version'] != r['dataset_version'] or
                previous[r['sample_id']].get('dataset_file_sha256') != dataset_hash for r in scored):
            raise ValueError("Previous experiment must use the same complete dataset version and IDs")
        if any(r['prediction'].get('generation_config', run['generation_config']) != run['generation_config'] for r in previous_rows):
            raise ValueError('Paired comparison requires the same decoding configuration')
        if any(r['prediction'].get('backend_config', run.get('backend_config', {})) != run.get('backend_config', {}) for r in previous_rows):
            raise ValueError('Paired comparison requires the same inference precision/context/backend settings')
        changes = {"fixed": [], "regressed": [], "changed_error_labels": []}
        for row in scored:
            old = previous[row["sample_id"]]["verification"]
            new = row["verification"]
            a, b = old["metrics"]["exact_match_accuracy"], new["metrics"]["exact_match_accuracy"]
            if not a and b:
                changes["fixed"].append(row["sample_id"])
            if a and not b:
                changes["regressed"].append(row["sample_id"])
            if old["errors"] != new["errors"]:
                changes["changed_error_labels"].append(row["sample_id"])
        old_counts = Counter(e for r in previous_rows for e in r['verification']['errors'])
        new_counts = Counter(e for r in scored for e in r['verification']['errors'])
        changes['error_count_deltas'] = {e: new_counts[e]-old_counts[e] for e in ERROR_TYPES}
        old_metrics = aggregate(previous_rows)['metrics']
        changes['metric_rate_deltas'] = {m: summary['metrics'][m]['rate']-old_metrics[m]['rate']
                                         if summary['metrics'][m]['rate'] is not None and old_metrics[m]['rate'] is not None else None
                                         for m in summary['metrics']}
    summary["comparison"] = changes
    out = Path(output_dir)
    write_jsonl(out/"scored.jsonl", scored)
    write_json(out/"metrics.json", summary)
    write_jsonl(out/"manual_review.jsonl", [r for r in scored if not r["verification"]["ground_truth_valid"] or any(
        d["error"] == "unsupported_schema" for d in r["verification"]["details"])])
    _markdown_report(out/"bad_cases.md", summary, scored)
    return summary


def _markdown_report(path, summary, scored):
    lines = ["# 统一评测与 Bad Case 报告", "", f"Run: `{summary['run_id']}`；CPU Mock：**{summary['is_mock']}**。",
             f"固定分母 {summary['samples']}；缺失 {summary['missing_predictions']}；生成失败 {summary['generation_errors']}；异常GT {summary['invalid_ground_truth']}。",
             "", "工具结构匹配不等于执行成功；自由文本语义正确率与任务执行成功率均未评估。", "",
             "| 指标 | 正确/分母 | 比例 |", "| --- | ---: | ---: |"]
    for key, value in summary["metrics"].items():
        rate = f"{value['rate']:.4%}" if value["rate"] is not None else "N/A"
        lines.append(f"| {key} | {value['numerator']}/{value['denominator']} | {rate} |")
    lines += ["", "## 多标签错误", "", "同一样本可有多个错误，比例之和不要求为100%。", ""]
    for key, value in summary["errors"].items():
        lines.append(f"- {key}: {value['count']}/{summary['samples']} ({value['rate']:.2%})")
    lines += ["", "## 复杂度切片", "", "| 切片 | 样本 | EM |", "| --- | ---: | ---: |"]
    for key, value in summary["slices"].items():
        rate = value['metrics']['exact_match_accuracy']['rate']
        lines.append(f"| {key} | {value['samples']} | {rate:.2%} |")
    if summary.get("comparison") is not None:
        c = summary["comparison"]
        lines += ["", f"上次实验对比：修复 {len(c['fixed'])}，退化 {len(c['regressed'])}，错误标签变化 {len(c['changed_error_labels'])}。"]
        lines += ['', '| 错误类别 | 数量变化（本次−上次） |', '| --- | ---: |']
        lines += [f"| {e} | {delta:+d} |" for e, delta in c['error_count_deltas'].items()]
    lines += ["", "## 典型失败（每类最多2例；完整记录见 scored.jsonl）", ""]
    for error in ERROR_TYPES:
        examples = [r for r in scored if error in r["verification"]["errors"]][:2]
        if not examples:
            continue
        lines += [f"### {error}", ""]
        for row in examples:
            preview = {"sample_id": row["sample_id"], "errors": row["verification"]["errors"], "ground_truth": row["ground_truth"],
                       "prediction_preview": row["prediction"]["raw_output"][:500]}
            lines += ["```json", json.dumps(preview, ensure_ascii=False, indent=2).replace('```', '` ` `'), "```", ""]
    write_text(path, '\n'.join(lines)+'\n')
