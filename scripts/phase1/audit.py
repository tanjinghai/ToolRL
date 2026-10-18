#!/usr/bin/env python3
"""只读 Phase 1 审计；仅 --output 指定的报告会写盘。依赖 numpy、pyarrow。"""
import argparse
import ast
from collections import Counter
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def norm(value):
    return re.sub(r"\s+", " ", value).strip().casefold()


def prompt(row):
    return canonical([row['instruction'], row['input']])


def last_query(text):
    users = re.findall(r'<user>(.*?)</user>', text, re.S)
    return norm(users[-1]) if users else norm(text)


def calls(text):
    # 标签字面量可能出现在蒸馏 think 中；先隔离回答主体，避免误判。
    text = text.split('</think>', 1)[-1]
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
    return [json.loads(line) for block in blocks for line in block.strip().splitlines() if line.strip()]


def dist(values):
    return dict(sorted(Counter(values).items(), key=lambda x: str(x[0])))


def quantiles(values):
    values = sorted(values)
    return {str(q): values[round((len(values)-1)*q)] for q in [0, .5, .9, .95, .99, 1]} if values else {}


def duplicate_report(keys):
    counts = Counter(keys)
    return {'unique': len(counts), 'excess_rows': sum(n-1 for n in counts.values()),
            'duplicate_groups': sum(n > 1 for n in counts.values())}


def describe(rows):
    parsed, invalid, bad_schema, missing_tool, bad_params = [], [], [], [], []
    naive_invalid, tag_mentions, format_invalid = [], [], []
    nparams, vtypes, ntools, shapes, thoughts = [], [], [], [], []
    schema_parse_failures = []
    for i, row in enumerate(rows):
        text = row['output']
        row_thoughts = re.findall(r'<think>(.*?)</think>', text, re.S)
        thoughts.extend(row_thoughts)
        body = text.split('</think>', 1)[-1]
        if any('<tool_call>' in t or '<response>' in t for t in row_thoughts):
            tag_mentions.append(i)
        shapes.append(('tool' if '<tool_call>' in body else '') + ('+response' if '<response>' in body else ''))
        pattern = r'^<think>.*?</think>\s*(?:<tool_call>\s*.*?\s*</tool_call>)?(?:\s*<response>.*?</response>)?\s*$'
        if not re.fullmatch(pattern, text, re.S):
            format_invalid.append(i)
        if '<tool_call>' in text:
            try:
                for line in text.split('<tool_call>')[1].split('</tool_call>')[0].strip().splitlines():
                    json.loads(line)
            except (ValueError, TypeError):
                naive_invalid.append(i)
        try:
            tools = calls(text)
            if '<tool_call>' in body and not tools:
                invalid.append(i)
        except (ValueError, TypeError):
            invalid.append(i)
            tools = []
        parsed.append(tools)
        ntools.append(len(tools))
        schemas = {}
        for name, literal in re.findall(r'\d+\. Name: (.*?)\nDescription: .*?\nParameters: (.*?)(?=\n|$)', row['instruction']):
            try:
                schemas[name] = json.loads(literal)
            except ValueError:
                try:
                    schemas[name] = ast.literal_eval(literal)
                except (ValueError, SyntaxError):
                    schema_parse_failures.append(i)
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get('name'), str) or not isinstance(tool.get('parameters'), dict):
                bad_schema.append(i)
                continue
            params = tool['parameters']
            nparams.append(len(params))
            vtypes.extend(type(v).__name__ for v in params.values())
            if tool['name'] not in schemas:
                missing_tool.append(i)
            else:
                unknown = set(params) - set(schemas[tool['name']])
                if unknown:
                    bad_params.append({'index': i, 'unknown_parameter_count': len(unknown)})
    return {
        'rows': len(rows), 'field_sets': dist(canonical(sorted(r)) for r in rows),
        'output_shapes': dist(shapes), 'tool_calls_per_row': dist(ntools),
        'parameters_per_call': dist(nparams), 'parameter_value_types': dist(vtypes),
        'invalid_json_indices': invalid, 'invalid_call_schema_indices': sorted(set(bad_schema)),
        'naive_reward_parser_failure_indices': naive_invalid,
        'protocol_tag_mentions_in_think_indices': tag_mentions,
        'outer_structure_invalid_indices': format_invalid,
        'unlisted_tool_indices': sorted(set(missing_tool)), 'unknown_parameters': bad_params,
        'schema_parse_failure_indices': sorted(set(schema_parse_failures)),
        'prompt_char_quantiles': quantiles([len(r['instruction'])+len(r['input']) for r in rows]),
        'output_char_quantiles': quantiles([len(r['output']) for r in rows]),
        'think_word_quantiles': quantiles([len(t.split()) for t in thoughts]),
        'unique_thoughts': len(set(t.strip() for t in thoughts)),
        'history_with_observation': sum('<obs>' in r['input'] for r in rows),
        'user_turns': dist(r['input'].count('<user>') for r in rows),
        'duplicate_rows': duplicate_report([canonical(r) for r in rows]),
        'duplicate_prompts': duplicate_report([prompt(r) for r in rows]),
        'duplicate_normalized_inputs': duplicate_report([norm(r['input']) for r in rows]),
        'duplicate_last_queries': duplicate_report([last_query(r['input']) for r in rows]),
        'duplicate_label_call_sets': duplicate_report([canonical(sorted(t, key=canonical)) for t in parsed]),
    }


def reward_probes():
    spec = importlib.util.spec_from_file_location('phase1_rlla', ROOT/'verl/utils/reward_score/rlla.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # 避免导入 verl / CUDA 训练依赖。
    flags = ['WITHLENGTH', 'SCHEDULELENGTH', 'REFINEDREWARD', 'COARSEREWARD',
             'INTERMEDIATEREWARD', 'CORRECTMAX1', 'MAX1STEP30MAX3', 'SCHEDULEREWARD', 'STRICTMATCH', 'EXPERIMENT_NAME']
    previous = {k: os.environ.get(k) for k in flags}
    for key in flags:
        os.environ.pop(key, None)
    os.environ['EXPERIMENT_NAME'] = 'phase1-qwen-probe'
    target = [{'name': 'f', 'parameters': {'x': 1}}]

    def output(tools=None, response=None):
        s = '<think>test</think>'
        if tools is not None:
            s += '\n<tool_call>\n' + '\n'.join(json.dumps(t) for t in tools) + '\n</tool_call>'
        if response is not None:
            s += '\n<response>' + response + '</response>'
        return s

    cases = {
        'exact': (output(target), output(target)),
        'extra_same_name_wrong_value': (output(target+[{'name': 'f', 'parameters': {'x': 2}}]), output(target)),
        'bool_equals_integer': (output([{'name': 'f', 'parameters': {'x': True}}]), output(target)),
        'wrong_no_tool_answer': (output(response='wrong answer'), output(response='right answer')),
        'no_tool_label_but_tool_and_response': (output(target, 'wrong'), output(response='right')),
        'non_json_but_format_valid': ('<think>test</think>\n<tool_call>\nnot JSON\n</tool_call>', output(target)),
        'wrong_value': (output([{'name': 'f', 'parameters': {'x': 2}}]), output(target)),
        'extra_parameter': (output([{'name': 'f', 'parameters': {'x': 1, 'junk': 2}}]), output(target)),
    }
    result = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for key, (prediction, answer) in cases.items():
                result[key] = mod.compute_score(prediction, answer)
            gt = [{'name': 'f', 'parameters': {'a': 1, 'b': 1}}, {'name': 'f', 'parameters': {'a': 1, 'b': 2}}]
            pd = [gt[1], {'name': 'f', 'parameters': {'a': 2, 'b': 1}}]
            result['greedy_gt_order_1'] = mod.compute_tool_call_reward(gt, pd, 3, -3)
            result['greedy_gt_order_2'] = mod.compute_tool_call_reward(gt[::-1], pd, 3, -3)
            os.environ['COARSEREWARD'] = '1'
            result['coarse_reordered_correct_calls'] = mod.compute_tool_call_reward(gt, gt[::-1], 3, -3)
            os.environ.pop('COARSEREWARD')
            os.environ['REFINEDREWARD'] = '1'
            result['refined_reordered_name_list'] = mod.match_score(['a', 'b'], ['b', 'a'])
            os.environ['EXPERIMENT_NAME'] = 'experiment_without_family'
            try:
                mod.compute_score(output(target), output(target))
            except NotImplementedError:
                result['unknown_experiment_name'] = 'NotImplementedError'
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        import numpy as np
        import pyarrow
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit('请在隔离环境安装 numpy、pyarrow；不需要安装训练依赖。') from e
    rl_path = ROOT/'dataset/rlla_4k_raw/rlla_rl.json'
    sft_path = ROOT/'dataset/rlla_4k_raw/rlla_sft.json'
    rl, sft = json.loads(rl_path.read_text()), json.loads(sft_path.read_text())
    raw_keys = {canonical(r) for r in rl}
    partitions, parquet_report = {}, {}
    paths = [rl_path, sft_path]
    for split in ['train', 'test']:
        p = ROOT/f'dataset/rlla_4k/{split}.parquet'
        paths.append(p)
        table = pq.read_table(p)
        ds = table.to_pylist()
        partitions[split] = ds
        extra_rows = [{k: r['extra_info'][k] for k in ['instruction', 'input', 'output']} for r in ds]
        parquet_report[split] = {
            'rows': len(ds), 'columns': table.column_names, 'schema': str(table.schema),
            'all_rows_in_raw': all(canonical(r) in raw_keys for r in extra_rows),
            'prompt_matches_extra_info': all(r['prompt'] == [{'role': 'system', 'content': r['extra_info']['instruction']}, {'role': 'user', 'content': r['extra_info']['input']}] for r in ds),
            'ground_truth_matches_extra_info': all(r['reward_model']['ground_truth'] == r['extra_info']['output'] for r in ds),
            'data_sources': dist(r['data_source'] for r in ds), 'abilities': dist(r['ability'] for r in ds),
            'split_tags': dist(r['extra_info']['split'] for r in ds),
            'duplicates': duplicate_report([prompt(r) for r in extra_rows]),
        }
    shuffled = list(rl)
    np.random.RandomState(31415).shuffle(shuffled)
    for split, expected in [('train', shuffled[:-80]), ('test', shuffled[-80:])]:
        actual = [{k: r['extra_info'][k] for k in ['instruction', 'input', 'output']} for r in partitions[split]]
        parquet_report[split]['matches_preprocess_seed_and_order'] = actual == expected
    train = [r['extra_info'] for r in partitions['train']]
    test = [r['extra_info'] for r in partitions['test']]
    pairs = {}
    for name, a, b in [('train_test', train, test), ('sft_rl', sft, rl), ('sft_train', sft, train), ('sft_test', sft, test)]:
        pairs[name] = {key: len({fn(r) for r in a} & {fn(r) for r in b}) for key, fn in [
            ('exact_prompt_overlap', prompt), ('normalized_input_overlap', lambda r: norm(r['input'])),
            ('normalized_last_query_overlap', lambda r: last_query(r['input']))]}
        right_indices = {}
        for j, row in enumerate(b):
            right_indices.setdefault(prompt(row), []).append(j)
        pairs[name]['exact_prompt_pairs_zero_based'] = [
            {'left_index': i, 'right_indices': right_indices[prompt(row)],
             'prompt_sha256': hashlib.sha256(prompt(row).encode()).hexdigest()}
            for i, row in enumerate(a) if prompt(row) in right_indices]
    benchmarks = {}
    for p in sorted((ROOT/'benchmarks/API-Bank').glob('*processed.json')):
        ds = json.loads(p.read_text()); paths.append(p)
        benchmarks[p.name] = {'rows': len(ds), 'fields': sorted(ds[0]),
            'raw_input_overlap': len({norm(r['input']) for r in rl+sft} & {norm(r['user']) for r in ds}),
            'last_query_overlap': len({last_query(r['input']) for r in rl+sft} & {last_query(r['user']) for r in ds}),
            'answer_list_lengths': dist(len(r['answer']) if isinstance(r['answer'], list) else 1 for r in ds)}
    p = ROOT/'benchmarks/Bamboogle/data.json'; paths.append(p)
    ds = json.loads(p.read_text())
    benchmarks['Bamboogle'] = {'rows': len(ds), 'last_query_overlap': len({last_query(r['input']) for r in rl+sft} & {norm(r['Question']) for r in ds})}
    report = {
        'audit_target_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'python': sys.version.split()[0], 'numpy': np.__version__, 'pyarrow': pyarrow.__version__,
        'raw': {'rl': describe(rl), 'sft': describe(sft)}, 'processed': parquet_report,
        'overlap': pairs, 'benchmarks': benchmarks, 'reward_probes': reward_probes(),
        'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'limitations': ['不验证来源数据集逐条 provenance；不把相同工具标签当作重复样本。',
                        '未做 embedding/模糊近重复、跨历史前缀比较、BFCL 全量污染检查。',
                        '字符/空格词数不等于 Qwen token；未下载 tokenizer 或模型。',
                        '缺省奖励探针不是 benchmark 指标。'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'已保存 {args.output}；RL={len(rl)}, SFT={len(sft)}；未更改数据。')


if __name__ == '__main__':
    main()
