"""Known-answer adversarial probes. No model or final-test data involved."""
from toolrl_pipeline.protocol import render_completion
from toolrl_pipeline.verifier import verify
from .rewards import compute_reward


def cases():
    tools = [{'name': 'f', 'parameters': {'type': 'object', 'properties': {'x': {'type': 'integer'}},
                                        'required': ['x'], 'additionalProperties': False}}]
    call = lambda x: {'name': 'f', 'parameters': {'x': x}}
    gt = [call(1)]
    return tools, gt, [('exact', render_completion(gt)), ('bool_int', render_completion([call(True)])),
                       ('wrong_value', render_completion([call(2)])),
                       ('extra_call', render_completion(gt+[call(2)])),
                       ('wrong_tool', render_completion([{'name': 'z', 'parameters': {'x': 1}}])),
                       ('invalid_json', '<think>r</think>\n<tool_call>\noops\n</tool_call>')]


def run_probes():
    tools, gt, probes = cases()
    report = {'is_model_evaluation': False, 'cases': [], 'strict_fullscore_false_positives': 0,
              'legacy_fullscore_false_positives': 0}
    for name, completion in probes:
        em = verify(gt, completion, tools, {'call_order': 'parallel'})['metrics']['exact_match_accuracy']
        item = {'name': name, 'independent_exact_match': em}
        for version in ('legacy', 'strict_matching'):
            scores = compute_reward(version, completion, render_completion(gt), tools, {'call_order': 'parallel'})
            item[version] = dict(zip(('total', 'format', 'correctness', 'length'), scores))
            if scores[0] >= 4-1e-9 and not em:
                report[('strict' if version == 'strict_matching' else 'legacy')+'_fullscore_false_positives'] += 1
        report['cases'].append(item)
    return report


def audit_validation_rewards(dataset_path, scored_path, version='strict_matching'):
    """Offline E4 gate on actual saved bad cases, recomputing independent EM.

    Partial reward versus EM is expected. Only a tool sample's maximum total
    with failed independent EM is a blocking contradiction. Free text is excluded.
    """
    from toolrl_pipeline.io import read_jsonl, file_hash
    records = {r['sample_id']: r for r in read_jsonl(dataset_path)}
    scored = read_jsonl(scored_path)
    if not records or any(r['split'] != 'validation' for r in records.values()):
        raise ValueError('Reward audit requires validation only')
    if len(scored) != len(records) or {r['sample_id'] for r in scored} != set(records):
        raise ValueError('Reward audit sample coverage mismatch')
    conflicts = []
    dataset_hash = file_hash(dataset_path)
    for row in scored:
        record = records[row['sample_id']]
        if row['dataset_file_sha256'] != dataset_hash:
            raise ValueError('Reward audit dataset hash mismatch')
        prediction = row['prediction']
        scores = compute_reward(version, prediction.get('raw_output', ''), record['target_output'],
                                record['tools'], record['metadata'])
        em = verify(record['ground_truth'], prediction, record['tools'], record['metadata'])['metrics']['exact_match_accuracy']
        if record['ground_truth'] and scores[0] >= 4 - 1e-9 and not em:
            conflicts.append({'sample_id': row['sample_id'], 'reward': list(scores), 'independent_exact_match': em})
    return {'source_split': 'validation', 'reward_version': version, 'samples': len(scored),
            'scored_sha256': file_hash(scored_path), 'fullscore_false_positives': len(conflicts), 'conflicts': conflicts}
