import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from toolrl_pipeline.benchmarks import api_bank_records, bfcl_decode_ast, bfcl_decode_execute, legacy_soft_match
from toolrl_pipeline.data import ROOT, export_record, prepare_record
from toolrl_pipeline.evaluation import generate, make_run, mock_backend, score
from toolrl_pipeline.io import read_jsonl, write_jsonl
from toolrl_pipeline.protocol import render_completion
from toolrl_pipeline.tokenization import encode_sft, render_prompt

from test_data import raw_record


class FakeTokenizer:
    eos_token, eos_token_id, bos_token = '<|im_end|>', 100000, None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'validation.jsonl'
        self.records = []
        for i in range(8):
            row = prepare_record(raw_record(query=f'Retrieve the next temperature in city number {i}.'), 'rl', i, 'hash')
            row.update({'split': 'validation', 'group_id': 'g'+str(i)})
            self.records.append(export_record(row, 'test-version'))
        write_jsonl(self.dataset, self.records)
        self.run = make_run(self.dataset, 'mock-test', 'mock-model', 'mock-v1', backend='mock-oracle')

    def tearDown(self):
        self.tmp.cleanup()

    def test_oracle_mock_full_pipeline_and_fixed_missing_denominator(self):
        pred = generate(self.dataset, self.root/'oracle', self.run, mock_backend(self.records, 'oracle'))
        r = score(self.dataset, pred, self.root/'scored')
        self.assertTrue(r['is_mock'])
        self.assertEqual(r['metrics']['exact_match_accuracy'], {'numerator': 8, 'denominator': 8, 'rate': 1.0})
        self.assertTrue((self.root/'scored/bad_cases.md').exists())
        write_jsonl(self.root/'missing.jsonl', read_jsonl(pred)[:2])
        r = score(self.dataset, self.root/'missing.jsonl', self.root/'missing-score')
        self.assertEqual(r['missing_predictions'], 6)
        self.assertEqual(r['metrics']['exact_match_accuracy']['denominator'], 8)
        self.assertEqual(r['metrics']['exact_match_accuracy']['numerator'], 2)

    def test_duplicate_unknown_and_mixed_provenance_refused(self):
        pred = generate(self.dataset, self.root/'oracle', self.run, mock_backend(self.records, 'oracle'))
        rows = read_jsonl(pred)
        for changes in ['duplicate', 'unknown', 'revision', 'config']:
            mutated = copy.deepcopy(rows)
            if changes == 'duplicate':
                mutated.append(mutated[0])
            elif changes == 'unknown':
                mutated[0]['sample_id'] = 'not-in-dataset'
            elif changes == 'revision':
                mutated[-1]['model']['revision'] = 'other'
            else:
                mutated[-1]['generation_config']['seed'] = 999
            write_jsonl(self.root/'bad.jsonl', mutated)
            with self.assertRaises(ValueError):
                score(self.dataset, self.root/'bad.jsonl', self.root/'no-results', run_metadata=self.run)

    def test_resume_retains_failed_attempts_and_no_labels_reach_backend(self):
        seen = []

        def backend(request, _config):
            self.assertEqual(set(request), {'sample_id', 'prompt', 'tools'})
            seen.append(request['sample_id'])
            if len(seen) == 1:
                raise RuntimeError('simulated generation failure')
            return {'raw_output': render_completion([{'name': 'f', 'parameters': {'x': 1}}])}
        pred = generate(self.dataset, self.root/'resume', self.run, backend)
        self.assertEqual(len(seen), 8)
        generate(self.dataset, self.root/'resume', self.run, backend, resume=True)
        self.assertEqual(len(seen), 8)
        rows = read_jsonl(pred)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]['status'], 'error')

    def test_bad_case_diff_and_truncation(self):
        first = generate(self.dataset, self.root/'mixed', self.run, mock_backend(self.records, 'mixed'))
        a = score(self.dataset, first, self.root/'mixed-score')
        second_run = {**self.run, 'run_id': 'mock-oracle-next'}
        second = generate(self.dataset, self.root/'oracle', second_run, mock_backend(self.records, 'oracle'))
        b = score(self.dataset, second, self.root/'oracle-score', self.root/'mixed-score/scored.jsonl')
        self.assertGreater(len(b['comparison']['fixed']), 0)
        self.assertEqual(len(b['comparison']['regressed']), 0)
        self.assertEqual(a['errors']['truncated_output']['count'], 1)
        self.assertIn('task:single', a['slices'])

    def test_final_test_gate(self):
        records = copy.deepcopy(self.records)
        for row in records:
            row['split'] = 'test'
        write_jsonl(self.dataset, records)
        run = make_run(self.dataset, 'mock-test-only', 'mock', 'v1')
        pred = generate(self.dataset, self.root/'frozen', run, mock_backend(records, 'oracle'))
        with self.assertRaises(ValueError):
            score(self.dataset, pred, self.root/'blocked')
        self.assertEqual(score(self.dataset, pred, self.root/'allowed', allow_final_test=True)['samples'], 8)

    def test_assistant_only_labels_and_no_duplicate_control_tokens(self):
        tokenizer = FakeTokenizer()
        row = self.records[0]
        encoded = encode_sft(row, tokenizer)
        n = encoded['prompt_length']
        self.assertEqual(encoded['labels'][:n], [-100]*n)
        self.assertEqual(encoded['labels'][n:], encoded['input_ids'][n:])
        self.assertEqual(encoded['input_ids'][-1], tokenizer.eos_token_id)
        self.assertEqual(encoded['input_ids'].count(tokenizer.eos_token_id), 1)
        self.assertTrue(render_prompt(row['prompt'], tokenizer).endswith('<|im_start|>assistant\n'))
        with self.assertRaises(ValueError):
            encode_sft(row, tokenizer, max_length=2)
        bad = {**row, 'target_output': row['target_output']+'<|im_end|>'}
        with self.assertRaises(ValueError):
            encode_sft(bad, tokenizer)


class BenchmarkTests(unittest.TestCase):
    def test_api_bank_fixed_manifest_and_legacy_looseness(self):
        records, hashes = api_bank_records(ROOT)
        self.assertEqual(len(records), 597)
        self.assertEqual(len({r['sample_id'] for r in records}), 597)
        self.assertEqual(len(hashes), 3)
        gt = [{'name': 'f', 'parameters': {'x': 1}}]
        text = render_completion(gt+[{'name': 'g', 'parameters': {'x': 2}}])
        self.assertTrue(legacy_soft_match(gt, text))
        text = '<think>x</think>\n<tool_call>\n{"x":1}\n</tool_call>'
        self.assertTrue(legacy_soft_match(gt, text))
        self.assertFalse(parse_strict_for_fixture(gt, text))

    def test_bfcl_bridge_and_malformed_output(self):
        text = render_completion([{'name': 'f', 'parameters': {'x': 1, 'y': 'hello'}}])
        self.assertEqual(bfcl_decode_ast(text), [{'f': {'x': 1, 'y': 'hello'}}])
        self.assertEqual(bfcl_decode_execute(text), ["f(x=1, y='hello')"])
        self.assertEqual(bfcl_decode_ast(render_completion([])), [])
        with self.assertRaises(ValueError):
            bfcl_decode_ast(text.replace('"x":1', '"x":1,broken'))

    def test_api_wrappers_are_cwd_independent_and_cpu_import_safe(self):
        for name in ('generate', 'evaluate'):
            result = subprocess.run([sys.executable, str(ROOT/'benchmarks/API-Bank'/f'{name}.py'), '--help'],
                                    cwd='/tmp', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--output-dir', result.stdout)
        result = subprocess.run([sys.executable, '-c', "import toolrl_pipeline.cli,sys; assert 'torch' not in sys.modules and 'vllm' not in sys.modules"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_api_mock_from_other_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT/'benchmarks/API-Bank/generate.py'), '--output-dir', tmp,
                                     '--run-id', 'api-mock', '--model-id', 'mock', '--model-revision', 'mock-v1'],
                                    cwd='/tmp', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((Path(tmp)/'metrics.json').read_text())
            self.assertEqual(metrics['samples'], 597)
            self.assertTrue(metrics['is_mock'])
            self.assertEqual(metrics['legacy_api_bank_soft_match']['denominator'], 597)


def parse_strict_for_fixture(gt, text):
    from toolrl_pipeline.verifier import verify
    return verify(gt, text, [{'name': 'f', 'parameters': {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'additionalProperties': False}}], {'call_order': 'parallel'})['metrics']['exact_match_accuracy']


if __name__ == '__main__':
    unittest.main()
