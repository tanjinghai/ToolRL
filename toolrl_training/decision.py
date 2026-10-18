"""Preregistered validation-only decisions; no test-driven tuning."""
from pathlib import Path
from statistics import mean
from toolrl_pipeline.io import loads, read_jsonl, write_json, write_text


def decide_e1(baseline_dir, sft_dir, output_dir, rules):
    base, sft = [loads((Path(p)/'metrics.json').read_text()) for p in (baseline_dir, sft_dir)]
    rows = read_jsonl(Path(sft_dir)/'scored.jsonl')
    if any(r['split'] != 'validation' for r in rows):
        raise ValueError('Decisions require validation only')
    if base['dataset_file_sha256'] != sft['dataset_file_sha256'] or base['generation_config'] != sft['generation_config']:
        raise ValueError('Unpaired E0/E1 evaluation')
    metrics = ('format_validity', 'tool_name_accuracy', 'parameter_name_accuracy',
               'parameter_value_accuracy', 'exact_match_accuracy', 'no_tool_accuracy')
    deltas = {k: sft['metrics'][k]['rate']-base['metrics'][k]['rate']
              if base['metrics'][k]['rate'] is not None and sft['metrics'][k]['rate'] is not None else None for k in metrics}
    drop = deltas['exact_match_accuracy'] <= -rules['em_drop_threshold']
    action = rules['on_sft_regression'] if drop else 'full'
    if action not in ('full', 'diagnostic', 'skip', 'manual'):
        raise ValueError('Unregistered E3 branch')
    errors = sorted(sft['errors'], key=lambda k: sft['errors'][k]['count'], reverse=True)
    lengths = [r['prediction'].get('completion_tokens') for r in rows]
    lengths = [v for v in lengths if isinstance(v, int)]
    result = {'source_split': 'validation', 'is_mock': sft['is_mock'], 'deltas': deltas,
              'format_improved': (deltas['format_validity'] or 0) > 0,
              'tool_selection_improved': (deltas['tool_name_accuracy'] or 0) > 0,
              'overall_regression': drop, 'e2': 'independent', 'e3': action,
              'e4': 'manual', 'largest_errors': errors[:3],
              'mean_output_tokens': mean(lengths) if lengths else None,
              'mean_output_characters': mean(len(r['prediction'].get('raw_output', '')) for r in rows),
              'no_tool_boundary': 'structural only; free-text semantic accuracy unavailable',
              'reward_reliability': 'legacy has known full-score false positives; strict candidate requires real bad-case review',
              'bad_cases': str(Path(sft_dir)/'bad_cases.md')}
    out = Path(output_dir)
    write_json(out/'decision_e1.json', result)
    write_text(out/'decision_e1.md', '# E1 Validation 决策\n\n' +
               '\n'.join(f'- {k}: {v}' for k, v in result.items()) + '\n')
    return result
