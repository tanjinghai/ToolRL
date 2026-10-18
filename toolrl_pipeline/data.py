"""Build versioned datasets without overwriting originals or guessing labels."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .benchmarks import api_bank_records, benchmark_questions
from .grouping import assign_splits, group_records, last_query
from .io import canonical, digest, file_hash, write_json, write_jsonl
from .protocol import parse_completion, render_completion
from .schema import extract_tools, validate_calls

ROOT = Path(__file__).resolve().parents[1]
BUILD_POLICY = "phase2-grouped-schema-v1"


def prepare_record(raw, source, index, source_hash):
    sid = source + "_" + digest(raw)
    tools, schema_errors, metadata_repairs = extract_tools(raw["instruction"])
    parsed = parse_completion(raw["output"])
    reasons, repairs = list(schema_errors), list(metadata_repairs)
    gt = parsed.calls
    if not (parsed.protocol_valid and parsed.json_valid and parsed.shape_valid):
        reasons.append("unparseable_target")
    # Only exact, uniquely resolvable whitespace errors can change a label.
    # No fuzzy tool-name or parameter-value repair is allowed.
    declared = {tool["name"] for tool in tools}
    for call in gt:
        if call["name"] not in declared and call["name"].strip() in declared:
            original = call["name"]
            call["name"] = call["name"].strip()
            repairs.append({"rule": "trim_label_name_to_exact_declared_name", "before": original, "after": call["name"]})
    reasons += [code for code, _ in validate_calls(gt, tools)]
    query = last_query(raw["input"])
    ordering = bool(len(gt) > 1 and re.search(r"\b(?:then|after|before|first|subsequently|once|sequential|in order|followed by)\b", query))
    if ordering:
        reasons.append("ambiguous_multi_call_order")
    order = "unknown" if ordering else "parallel"
    target = raw["output"]
    if any(isinstance(r, dict) for r in repairs):
        target = render_completion(gt, parsed.response, parsed.think)
    task = "no_tool" if not gt else "single" if len(gt) == 1 else "multi"
    return {"sample_id": sid, "source": source, "raw_index": index, "raw": raw,
            "source_file_sha256": source_hash, "raw_record_sha256": digest(raw),
            "prompt_sha256": digest([raw["instruction"], raw["input"]]), "tools": tools,
            "ground_truth": gt, "target_output": target, "reasons": sorted(set(reasons)), "repairs": repairs,
            "action": "quarantine" if reasons else "repair" if repairs else "keep",
            "metadata": {"call_order": order, "order_evidence": "explicit ordering cue requires review" if ordering else
                         "source instruction permits simultaneous calls; no explicit order cue in current query",
                         "task_type": task, "call_count": len(gt), "parameter_count": sum(len(c["parameters"]) for c in gt),
                         "history_turns": raw["input"].count("<user>"), "prompt_characters": len(raw["instruction"])+len(raw["input"]),
                         "source_dataset": "unknown_not_provided_in_release",
                         "tool_schema_sha256": digest(tools),
                         "required_parameter_status": "only_explicit_required_enforced; source_often_omits_required"}}


def export_record(row, version):
    return {"sample_id": row["sample_id"], "group_id": row["group_id"], "dataset_version": version,
            "source": row["source"], "split": row["split"],
            "prompt": [{"role": "system", "content": row["raw"]["instruction"]}, {"role": "user", "content": row["raw"]["input"]}],
            "tools": row["tools"], "ground_truth": row["ground_truth"], "target_output": row["target_output"],
            "metadata": {**row["metadata"], "raw_index": row["raw_index"], "source_file_sha256": row["source_file_sha256"],
                         "raw_record_sha256": row["raw_record_sha256"], "prompt_sha256": row["prompt_sha256"]}}


def grpo_row(record):
    # Only prompt is read by RLHFDataset for generation. Answers live outside it.
    return {"data_source": "rlla_phase2", "prompt": record["prompt"], "ability": "tool_use",
            "reward_model": {"style": "rule", "ground_truth": record["target_output"]},
            "available_tools": canonical(record["tools"]), "sample_id": record["sample_id"],
            "data_version": record["dataset_version"], "source": record["source"], "split": record["split"],
            "extra_info": {"index": record["metadata"]["raw_index"], "sample_id": record["sample_id"],
                           "group_id": record["group_id"], "split": record["split"], "dataset_version": record["dataset_version"],
                           "source": record["source"], "call_order": record["metadata"]["call_order"]}}


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Empty split: {path}; inspect grouping/cleaning rather than silently train on nothing")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _summary(rows):
    return {"rows": len(rows), "task_types": dict(Counter(r["metadata"]["task_type"] for r in rows)),
            "call_counts": dict(Counter(r["metadata"]["call_count"] for r in rows)),
            "parameter_counts": dict(Counter(r["metadata"]["parameter_count"] for r in rows)),
            "sources": dict(Counter(r["source"] for r in rows))}


def build(output, manifests, seed=31415, root=ROOT, bfcl_dir=None, freeze=False):
    root, output, manifests = Path(root), Path(output), Path(manifests)
    # Build to a new directory. Frozen tests cannot be regenerated in place.
    if (manifests/"version.json").exists() or (manifests/"FROZEN.json").exists() or (output/"version.json").exists():
        raise ValueError("Output already exists or is frozen; use a new version/directory")
    raw_paths = {source: root/f"dataset/rlla_4k_raw/rlla_{source}.json" for source in ("rl", "sft")}
    hashes = {str(p.relative_to(root)): file_hash(p) for p in raw_paths.values()}
    for p in sorted((root/"benchmarks").glob("*/*.json")):
        if p.name not in ("secret.json", "secret.example.json"):
            hashes[str(p.relative_to(root))] = file_hash(p)
    bfcl_hashes = {p.name: file_hash(p) for p in sorted(Path(bfcl_dir).glob("BFCL_v3_*.json"))} if bfcl_dir else {}
    code_hashes = {name: file_hash(Path(__file__).parent/name) for name in ("data.py", "grouping.py", "schema.py", "protocol.py", "benchmarks.py", "io.py")}
    config = {"policy": BUILD_POLICY, "seed": seed, "raw_hashes": hashes, "code_hashes": code_hashes, "bfcl_hashes": bfcl_hashes}
    version = "phase2-v1-" + digest(config)[:12]
    records = []
    raw_counts = {}
    for source, path in raw_paths.items():
        rows = json.loads(path.read_text())
        raw_counts[source] = len(rows)
        records.extend(prepare_record(raw, source, i, hashes[str(path.relative_to(root))]) for i, raw in enumerate(rows))
    # Resolve exact prompt conflicts before split; think and free-answer wording are not call labels.
    by_prompt = defaultdict(list)
    for row in records:
        by_prompt[row["prompt_sha256"]].append(row)
    for rows in by_prompt.values():
        signatures = {canonical(sorted(r["ground_truth"], key=canonical)) for r in rows if r["action"] != "quarantine"}
        if len(signatures) > 1:
            for row in rows:
                row["reasons"] = sorted(set(row["reasons"] + ["conflicting_exact_prompt_labels"]))
                row["action"] = "quarantine"
    # Exact duplicates are removed within each source only. Cross-source overlap is preserved.
    seen = {}
    for row in records:
        key = row["source"], row["raw_record_sha256"]
        if key in seen:
            row["action"] = "delete_duplicate"
            row["reasons"].append("exact_duplicate_record")
            row["duplicate_of_raw_index"] = seen[key]
        else:
            seen[key] = row["raw_index"]
    questions = benchmark_questions(root, bfcl_dir)
    references = [{"sample_id": "bench_"+digest(q), "raw": {"instruction": "", "input": q["text"]},
                   "benchmark_ref": {"source": q["source"], "id": q["id"]}} for q in questions]
    grouping = group_records(records+references)
    benchmark_groups = defaultdict(list)
    for row in references:
        benchmark_groups[row["group_id"]].append(row["benchmark_ref"])
    benchmark_hits = []
    for row in records:
        if row["group_id"] in benchmark_groups:
            benchmark_hits.append({"sample_id": row["sample_id"], "benchmarks": benchmark_groups[row["group_id"]]})
            if row["action"] != "delete_duplicate":
                row["action"] = "quarantine"
                row["reasons"] = sorted(set(row["reasons"] + ["external_benchmark_group_overlap"]))
    for row in records:
        if row['action'] == 'quarantine':
            uncertain = {'wrong_tool', 'extra_argument', 'unparseable_target', 'conflicting_exact_prompt_labels', 'conflicting_duplicate_tool_name'}
            row['decision_category'] = 'unverifiable_without_source' if uncertain & set(row['reasons']) else 'manual_review'
            row['decision_reason'] = ';'.join(row['reasons'])
        elif row['action'] == 'delete_duplicate':
            row['decision_category'], row['decision_reason'] = 'certain_duplicate', 'exact canonical raw record already recorded within this source'
        elif row['action'] == 'repair':
            row['decision_category'], row['decision_reason'] = 'certain_repair', 'only provable metadata/unique whitespace canonicalization; all declared checks passed'
        else:
            row['decision_category'], row['decision_reason'] = 'structurally_valid', 'protocol, declared tool names, keys and types passed; no semantic correctness claim'
    split_policy = assign_splits(records, seed)
    kept = [r for r in records if r["action"] in ("keep", "repair")]
    retained = [export_record(r, version) for r in kept]
    source_prompts = {source: {r['prompt_sha256'] for r in records if r['source'] == source} for source in ('rl', 'sft')}
    shared_prompts = source_prompts['rl'] & source_prompts['sft']
    retained_train_prompts = {source: {r['metadata']['prompt_sha256'] for r in retained
                                      if r['source'] == source and r['split'] == 'train'} for source in ('rl', 'sft')}
    # Group isolation assertions run before writing any artifact.
    split_groups = {s: {r["group_id"] for r in retained if r["split"] == s} for s in ("train", "validation", "test")}
    assert not (split_groups["train"] & split_groups["validation"] or split_groups["train"] & split_groups["test"] or split_groups["validation"] & split_groups["test"])
    assert not {r["group_id"] for r in retained} & set(benchmark_groups)
    collections = {}
    for source in ("rl", "sft"):
        for split in ("train", "validation", "test"):
            subset = [r for r in retained if r["source"] == source and r["split"] == split]
            collections[f"{source}/{split}.jsonl"] = subset
            if source == "rl":
                write_parquet(output/f"grpo/{split}.parquet", [grpo_row(r) for r in subset])
            else:
                write_parquet(output/f"sft/{split}.parquet", [
                    {"prompt": r["prompt"], "response": r["target_output"], "sample_id": r["sample_id"],
                     "dataset_version": version, "split": split, "group_id": r["group_id"]} for r in subset])
    for split in ("validation", "test"):
        unique = {}
        # Prefer RL labels for identical prompts; held-out think is never model input.
        for r in sorted(retained, key=lambda r: (r["source"] != "rl", r["sample_id"])):
            if r["split"] == split:
                unique.setdefault(r["metadata"]["prompt_sha256"], r)
        collections[f"eval/{split}.jsonl"] = list(unique.values())
    apis, api_hashes = api_bank_records(root)
    collections["benchmarks/api_bank.jsonl"] = apis
    summaries = {}
    for relative, rows in collections.items():
        write_jsonl(output/relative, rows)
        manifest = [{"sample_id": r["sample_id"], "group_id": r["group_id"], "source": r["source"],
                     "raw_index": r["metadata"]["raw_index"], "split": r["split"]} for r in rows]
        write_jsonl(manifests/relative, manifest)
        summaries[relative] = _summary(rows)
    ledger = [{k: r.get(k) for k in ("sample_id", "source", "raw_index", "raw_record_sha256", "source_file_sha256", "prompt_sha256",
                                     "group_id", "action", "reasons", "repairs", "split", "duplicate_of_raw_index", "decision_category", "decision_reason")} for r in records]
    write_jsonl(output/"lineage.jsonl", ledger)
    write_jsonl(output/"quarantine.jsonl", [{**export_record(r, version), "reasons": r["reasons"], "repairs": r["repairs"]}
                                           for r in records if r["action"] == "quarantine"])
    write_jsonl(output/"group_edges.jsonl", grouping.pop("edges"))
    write_json(output/"benchmark_overlap.json", benchmark_hits)
    counts = {source: dict(Counter(r["action"] for r in records if r["source"] == source)) for source in ("rl", "sft")}
    quality = {
        "dataset_version": version, "raw_counts": raw_counts, "actions": counts, "split_policy": split_policy,
        "quarantine_reason_occurrences": dict(Counter(reason for r in records if r["action"] == "quarantine" for reason in r["reasons"])),
        "decision_categories": dict(Counter(r['decision_category'] for r in records)),
        "cross_source_prompt_overlap": {'raw_unique_prompts': len(shared_prompts),
                                        'retained_exact_train_overlap': len(retained_train_prompts['rl'] & retained_train_prompts['sft']),
                                        'shared_sft_dispositions': dict(Counter(r['action'] for r in records if r['source'] == 'sft' and r['prompt_sha256'] in shared_prompts)),
                                        'policy': 'Preserve valid shared prompts in train; quarantine only for independently recorded quality issues.'},
        "tool_definition_reuse": {"unique_schema_sets": len({r['metadata']['tool_schema_sha256'] for r in records}),
                                  "note": "Repeated tool schemas across different requests are measured, not automatically deleted or grouped."},
        "repairs": [r for r in ledger if r["repairs"]], "grouping": grouping, "splits": summaries,
        "leakage_checks": {"retained_group_intersections": 0, "retained_external_overlap_groups": 0,
                           "external_quarantined_occurrences": len(benchmark_hits),
                           "benchmark_question_counts": dict(Counter(q["source"] for q in questions)),
                           "bfcl_files_checked": len(bfcl_hashes), "bfcl_coverage": "snapshot files only" if bfcl_hashes else "not_downloaded_not_certified"},
        "api_bank_invalid_ground_truth": sum(bool(r["metadata"]["ground_truth_issues"]) for r in apis),
        "limitations": ["Schema validates declared fields/types only; omitted required information is not invented.",
                        "Unknown source provenance and non-lexical paraphrases remain unverified.",
                        "Multi-call order cues are quarantined rather than guessed; this changes task distribution.",
                        "No semantic correctness claim for free-text responses or external tool execution."]}
    write_json(output/"quality.json", quality)
    files = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}
    manifest_hashes = {str(p.relative_to(manifests)): file_hash(p) for p in sorted(manifests.rglob("*.jsonl"))}
    version_record = {"dataset_version": version, "config": config, "files_sha256": files, "manifests_sha256": manifest_hashes,
                      "api_bank_hashes": api_hashes, "original_files_unchanged": all(file_hash(root/p) == h for p, h in hashes.items())}
    assert version_record["original_files_unchanged"]
    write_json(output/"version.json", version_record)
    write_json(manifests/"version.json", version_record)
    if freeze:
        write_json(manifests/"FROZEN.json", {"dataset_version": version, "final_test_sha256": files["eval/test.jsonl"],
                                           "manifest_sha256": manifest_hashes["eval/test.jsonl"], "policy": "Never use for data/reward/checkpoint selection."})
    return quality


def select_training_subset(records, sample_ids):
    """Future bad-case-driven RL hook: train IDs only, no evaluation labels accepted."""
    requested = set(sample_ids)
    selected = [r for r in records if r["sample_id"] in requested]
    if {r["sample_id"] for r in selected} != requested or any(r["split"] != "train" for r in selected):
        raise ValueError("Training subset references missing or held-out IDs")
    return selected
