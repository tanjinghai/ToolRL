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
