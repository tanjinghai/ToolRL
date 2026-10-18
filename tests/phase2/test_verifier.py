import copy
import itertools
import unittest

from toolrl_pipeline.protocol import parse_completion, render_completion
from toolrl_pipeline.schema import annotation_schema, strict_equal, validate_value
from toolrl_pipeline.verifier import _best_assignment, verify


def tools():
    return [{"name": name, "parameters": {"type": "object", "properties": {
        "x": {"type": "integer"}, "y": {"type": "string"}}, "required": ["x"], "additionalProperties": False}} for name in ('f', 'g')]


def call(name='f', **params):
    return {"name": name, "parameters": params}


class VerifierTests(unittest.TestCase):
    def check(self, gt, pd, order='parallel', **kwargs):
        raw = pd if isinstance(pd, (str, dict)) else render_completion(pd)
        return verify(gt, raw, tools(), {'call_order': order, **kwargs})

    def test_exact_and_independent_name_metrics(self):
        r = self.check([call(x=1)], [call(x=1)])
        self.assertTrue(r['metrics']['exact_match_accuracy'])
        self.assertEqual(r['errors'], [])
        r = self.check([call(x=1)], [call(x=2)])
        self.assertTrue(r['metrics']['tool_name_accuracy'])
        self.assertTrue(r['metrics']['parameter_name_accuracy'])
        self.assertFalse(r['metrics']['parameter_value_accuracy'])
        self.assertIn('wrong_argument_value', r['errors'])

    def test_bool_integer_numeric_string_and_integer_float(self):
        for bad in [True, '1', 1.0]:
            with self.subTest(bad=bad):
                r = self.check([call(x=1)], [call(x=bad)])
                self.assertFalse(r['metrics']['exact_match_accuracy'])
                self.assertTrue(r['metrics']['parameter_name_accuracy'])
                self.assertIn('wrong_argument_type', r['errors'])

    def test_missing_extra_arguments_and_nested_type(self):
        r = self.check([call(x=1, y='a')], [call(x=1, junk=2)])
        self.assertTrue({'missing_argument', 'extra_argument'} <= set(r['errors']))
        self.assertFalse(r['metrics']['parameter_name_accuracy'])
        schema = {'type': 'object', 'properties': {'v': {'type': 'array', 'items': {'type': 'boolean'}}}, 'additionalProperties': False}
        self.assertEqual(validate_value({'v': [True]}, schema), [])
        self.assertEqual(validate_value({'v': [1]}, schema)[0][0], 'wrong_argument_type')
        self.assertFalse(strict_equal({'v': True}, {'v': 1}, schema))

    def test_json_no_line_is_dropped_and_duplicate_keys_rejected(self):
        for raw in ['<think>x</think>\n<tool_call>\n{broken}\n</tool_call>',
                    '<think>x</think>\n<tool_call>\n{"name":"f","parameters":{"x":1}}\nbad\n</tool_call>',
                    '<think>x</think>\n<tool_call>\n{"name":"f","parameters":{"x":1,"x":2}}\n</tool_call>',
                    '<think>x</think>\n<tool_call>\n{"name":"f","parameters":{"x":NaN}}\n</tool_call>']:
            r = self.check([call(x=1)], raw)
            self.assertFalse(r['metrics']['exact_match_accuracy'])
            self.assertFalse(r['metrics']['json_validity'])
            self.assertIn('format_error', r['errors'])

    def test_thought_tags_and_outer_ambiguity(self):
        text = render_completion([call(x=1)], think='I mention <tool_call> and <response> as literal protocol names.')
        self.assertTrue(self.check([call(x=1)], text)['metrics']['exact_match_accuracy'])
        invalid = text+'\n<tool_call>\n{"name":"f","parameters":{"x":1}}\n</tool_call>'
        self.assertFalse(parse_completion(invalid).protocol_valid)
        self.assertFalse(parse_completion('<think>x</think>\n<response>a</response>\n<response>b</response>').protocol_valid)

    def test_wrong_missing_extra_unknown_and_duplicate_tools(self):
        self.assertIn('wrong_tool', self.check([call(x=1)], [call('g', x=1)])['errors'])
        self.assertIn('missing_tool', self.check([call(x=1), call('g', x=1)], [call(x=1)])['errors'])
        for pd in [[call(x=1), call(x=1)], [call(x=1), call('unknown', x=1)]]:
            r = self.check([call(x=1)], pd)
            self.assertIn('extra_tool', r['errors'])
            self.assertFalse(r['metrics']['exact_match_accuracy'])

    def test_parallel_permutations_same_names_and_multiplicity(self):
        gt = [call(x=1, y='a'), call(x=2, y='b'), call('g', x=3)]
        for perm in itertools.permutations(gt):
            self.assertTrue(self.check(gt, list(perm))['metrics']['exact_match_accuracy'])
        r = self.check([call(x=1), call(x=2)], [call(x=1), call(x=1)])
        self.assertFalse(r['metrics']['exact_match_accuracy'])
        self.assertIn('wrong_argument_value', r['errors'])

    def test_sequential_order_is_enforced(self):
        for gt in [[call(x=1), call('g', x=2)], [call(x=1), call(x=2)]]:
            r = self.check(gt, gt[::-1], 'sequential')
            self.assertFalse(r['metrics']['exact_match_accuracy'])
            self.assertTrue(r['metrics']['tool_name_accuracy'])
            self.assertIn('wrong_call_order', r['errors'])
            self.assertTrue(self.check(gt, gt, 'sequential')['metrics']['exact_match_accuracy'])

    def test_no_tool_is_structural_not_semantic(self):
        r = self.check([], '<think>x</think>\n<response>Unverifiable free text.</response>')
        self.assertTrue(r['metrics']['no_tool_accuracy'])
        self.assertIsNone(r['free_text_semantic_accuracy'])
        self.assertIsNone(r['metrics']['tool_name_accuracy'])
        r = self.check([], [call(x=1)])
        self.assertIn('unnecessary_tool', r['errors'])
        self.assertFalse(r['metrics']['no_tool_accuracy'])
        self.assertFalse(self.check([], 'bad')['metrics']['no_tool_accuracy'])

    def test_truncation_and_generation_failure_are_failed_attempts(self):
        for status, finish in [('ok', 'length'), ('error', 'error')]:
            r = self.check([call(x=1)], {'raw_output': render_completion([call(x=1)]), 'status': status, 'finish_reason': finish})
            self.assertFalse(r['metrics']['exact_match_accuracy'])
            self.assertFalse(r['metrics']['format_validity'])
        self.assertIn('truncated_output', self.check([], {'raw_output': render_completion([]), 'finish_reason': 'length'})['errors'])

    def test_invalid_gt_and_unknown_order(self):
        for gt, order in [([call(x=True)], 'parallel'), ([call('unknown', x=1)], 'parallel'), ([call(x=1), call(x=2)], 'unknown')]:
            r = self.check(gt, [call(x=1)], order)
            self.assertIn('invalid_ground_truth', r['errors'])
            self.assertFalse(r['metrics']['exact_match_accuracy'])
        r = verify([call(x=1)], render_completion([call(x=1)]), [{'name': 'f', 'parameters': {'type': 'object', 'properties': []}}])
        self.assertIn('invalid_ground_truth', r['errors'])

    def test_schema_annotations_and_number_policy(self):
        for value in ['List[List[int]]', 'List[Union[int, float]]', 'Tuple[float, float]', 'str, optional', 'Dict[str, int]']:
            self.assertNotIn('x-unsupported-type', annotation_schema(value))
        for value in ['set', 'Callable[[float], float]', 'unknown']:
            self.assertIn('x-unsupported-type', annotation_schema(value))
        self.assertTrue(strict_equal(1, 1.0, {'type': 'number'}))
        self.assertFalse(strict_equal(True, 1, {'type': 'number'}))
        self.assertFalse(strict_equal(1, '1', {'type': 'number'}))
        self.assertEqual(validate_value([1, 2], annotation_schema('Tuple[float, float]')), [])
        self.assertTrue(validate_value([1], annotation_schema('Tuple[float, float]')))

    def test_assignment_matches_bruteforce_global_optimum(self):
        matrices = [[[2, 2], [3, 0]], [[4, 1, 2], [5, 3, 0], [1, 4, 2]], [[0, 0], [0, 0]]]
        for matrix in matrices:
            pairs = _best_assignment(matrix)
            actual = sum(matrix[i][j] for i, j in pairs)
            expected = max(sum(matrix[i][j] for i, j in enumerate(perm)) for perm in itertools.permutations(range(len(matrix))))
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
