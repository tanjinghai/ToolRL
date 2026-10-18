#!/usr/bin/env python3
"""Run the complete CPU acceptance checks; mock results are never model scores."""
import argparse
import ast
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from toolrl_pipeline.evaluation import generate, make_run, mock_backend, score
from toolrl_pipeline.io import file_hash, read_jsonl, write_json, write_jsonl, write_text
from toolrl_pipeline.schema import validate_calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--manifests', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks = {}
    source_paths = [p for folder in ('verl', 'toolrl_pipeline', 'benchmarks', 'scripts', 'tests') for p in (ROOT/folder).rglob('*.py')]
    for path in source_paths:
        ast.parse(path.read_text())
    checks['python_ast_files'] = len(source_paths)
    shell_paths = ['train_grpo.sh', 'train_ppo.sh', 'examples/grpo_trainer/run_grpo.sh', 'examples/ppo_trainer/run_ppo.sh']
    for path in shell_paths:
        subprocess.run(['bash', '-n', str(ROOT/path)], check=True)
    checks['shell_syntax_files'] = len(shell_paths)
    env = {**os.environ, 'TOOLRL_TOKENIZER_PATH': str(args.tokenizer.resolve())}
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/phase2', '-v'],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    write_text(args.output_dir/'unittest.log', result.stdout+result.stderr)
    result.check_returncode()
    checks['unit_tests'] = {'status': 'PASS', 'tests_passed': (result.stdout+result.stderr).count(' ... ok'),
                            'skipped': (result.stdout+result.stderr).count(' ... skipped')}
    version = json.loads((args.data_root/'version.json').read_text())
    assert version == json.loads((args.manifests/'version.json').read_text())
    for relative, digest in version['files_sha256'].items():
        assert file_hash(args.data_root/relative) == digest, relative
    for relative, digest in version['manifests_sha256'].items():
        assert file_hash(args.manifests/relative) == digest, relative
    for relative, digest in version['config']['raw_hashes'].items():
        assert file_hash(ROOT/relative) == digest, relative
    for name, digest in version['config']['code_hashes'].items():
        assert file_hash(ROOT/'toolrl_pipeline'/name) == digest, name
    assert json.loads((args.manifests/'FROZEN.json').read_text())['final_test_sha256'] == version['files_sha256']['eval/test.jsonl']
    groups, prompts = {}, {}
    retained = []
    for split in ('train', 'validation', 'test'):
        rows = read_jsonl(args.data_root/f'rl/{split}.jsonl')+read_jsonl(args.data_root/f'sft/{split}.jsonl')
        retained += rows
        groups[split] = {r['group_id'] for r in rows}
        prompts[split] = {r['metadata']['prompt_sha256'] for r in rows}
        assert all(not validate_calls(r['ground_truth'], r['tools']) for r in rows)
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        assert not groups[a] & groups[b]
        assert not prompts[a] & prompts[b]
    ledger = read_jsonl(args.data_root/'lineage.jsonl')
    assert len(ledger) == 4400 and all(r['decision_reason'] for r in ledger)
    # Small versioned review evidence; full lineage/group edges stay with generated data.
    for source in ('rl', 'sft'):
        decisions = [{'raw_index': r['raw_index'], 'sample_id': r['sample_id'], 'action': r['action'],
                      'reason_codes': r['reasons'] or ['passed_declared_structural_checks'], 'split': r['split']}
                     for r in ledger if r['source'] == source]
        write_jsonl(args.output_dir/'decisions'/(source+'.jsonl'), decisions)
    assert len({r['sample_id'] for r in retained}) == len(retained)
    checks['data_and_schema'] = {'status': 'PASS', 'raw_occurrences_accounted': len(ledger), 'retained': len(retained),
                                 'split_group_and_exact_prompt_intersections': 0,
                                 'data_hashes': len(version['files_sha256']), 'manifest_hashes': len(version['manifests_sha256'])}
    checks['data_and_schema']['decision_evidence_sha256'] = {source: file_hash(args.output_dir/'decisions'/(source+'.jsonl')) for source in ('rl', 'sft')}
    subprocess.run([sys.executable, str(ROOT/'scripts/phase2/check_tokenizer.py'), '--data-root', str(args.data_root),
                    '--tokenizer', str(args.tokenizer), '--output', str(args.output_dir/'tokenizer_validation.json')], check=True)
    summaries = {}
    dataset = args.data_root/'eval/validation.jsonl'
    rows = read_jsonl(dataset)
    previous = None
    for mode in ('mixed', 'oracle'):
        folder = args.work_dir/('validation-'+mode)
        run = make_run(dataset, 'phase2-mock-validation-'+mode, 'mock-'+mode, 'synthetic-v1', backend='mock-'+mode)
        predictions = generate(dataset, folder, run, mock_backend(rows, mode))
        summary = score(dataset, predictions, folder, previous_results=previous)
        if mode == 'oracle':
            assert summary['metrics']['exact_match_accuracy']['rate'] == 1.0
            assert not summary['comparison']['regressed']
        previous = folder/'scored.jsonl'
        summaries['validation-'+mode] = summary
    api = args.data_root/'benchmarks/api_bank.jsonl'
    rows = read_jsonl(api)
    folder = args.work_dir/'api-bank-mixed'
    run = make_run(api, 'phase2-mock-api-bank', 'mock-mixed', 'synthetic-v1', backend='mock-mixed')
    predictions = generate(api, folder, run, mock_backend(rows))
    summary = score(api, predictions, folder)
    assert summary['samples'] == 597 and summary['metrics']['exact_match_accuracy']['denominator'] == 597
    assert summary['legacy_api_bank_soft_match']['denominator'] == 597
    summaries['api-bank-mixed'] = summary
    write_json(args.output_dir/'mock_metrics.json', summaries)
    checks['mock_evaluation'] = {'status': 'PASS', 'validation_samples': len(read_jsonl(dataset)),
                                 'api_bank_fixed_denominator': 597, 'oracle_validation_em': 1.0,
                                 'comparison_and_bad_case_reports': True, 'final_test_scored': False}
    completed = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    completed.check_returncode()
    checks['pip_check'] = completed.stdout.strip()
    assert not importlib.util.find_spec('torch') and not importlib.util.find_spec('vllm')
    checks['no_torch_or_vllm_installed_in_cpu_environment'] = True
    subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
    checks['git_diff_check'] = 'PASS'
    write_json(args.output_dir/'validation.json', {'status': 'PASS', 'dataset_version': version['dataset_version'],
               'checks': checks, 'not_validated': ['GPU/CUDA/FlashAttention/Ray/FSDP', 'real model inference', 'SFT/GRPO updates',
                                                   'BFCL official execution/scoring', 'free-text semantic or task execution success', 'paid API services']})
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
