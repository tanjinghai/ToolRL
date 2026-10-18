import copy
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from toolrl_pipeline.data import build, prepare_record, select_training_subset
from toolrl_pipeline.grouping import assign_splits, group_records
from toolrl_pipeline.io import file_hash, read_jsonl
from toolrl_pipeline.protocol import render_completion
from toolrl_pipeline.schema import validate_calls

ROOT = Path(__file__).resolve().parents[2]


def raw_record(output=None, query='Please retrieve the temperature in London tomorrow morning.'):
    return {'instruction': '**Available Tools**\n1. Name: f\nDescription: test\nParameters: {"x":{"type":"int"}}\n\n**Steps for Each Turn**\n',
            'input': '<user>'+query+'</user>', 'output': output or render_completion([{'name': 'f', 'parameters': {'x': 1}}])}


class CleaningTests(unittest.TestCase):
    def test_wrong_type_is_quarantined_and_raw_unchanged(self):
        raw = raw_record(render_completion([{'name': 'f', 'parameters': {'x': '1'}}]))
        original = copy.deepcopy(raw)
        row = prepare_record(raw, 'rl', 0, 'hash')
        self.assertEqual(row['action'], 'quarantine')
        self.assertIn('wrong_argument_type', row['reasons'])
        self.assertEqual(raw, original)

    def test_provable_whitespace_repair_preserves_think(self):
        raw = raw_record(render_completion([{'name': 'f ', 'parameters': {'x': 1}}], think='Original long think retained.'))
        row = prepare_record(raw, 'sft', 0, 'hash')
        self.assertEqual(row['action'], 'repair')
        self.assertIn('Original long think retained.', row['target_output'])
        self.assertIn('f ', raw['output'])
        self.assertEqual(row['ground_truth'][0]['name'], 'f')

    def test_unknown_tool_not_fuzzy_repaired(self):
        row = prepare_record(raw_record(render_completion([{'name': 'f (extended)', 'parameters': {'x': 1}}])), 'rl', 0, 'h')
        self.assertEqual(row['action'], 'quarantine')
        self.assertEqual(row['ground_truth'][0]['name'], 'f (extended)')

    def test_cross_source_history_groups_and_train_anchor(self):
        a = prepare_record(raw_record(), 'rl', 0, 'h')
        b = prepare_record(raw_record(query='Please retrieve the temperature in London tomorrow morning.\n'), 'sft', 0, 'h')
        b['raw']['input'] = a['raw']['input']+'\n<obs>static observation</obs>\n<user>Please now give the next forecast for the weekend.</user>'
        group_records([a, b])
        self.assertEqual(a['group_id'], b['group_id'])
        assign_splits([a, b], 123)
        self.assertEqual((a['split'], b['split']), ('train', 'train'))

    def test_lexical_near_duplicate_group(self):
        query = ' '.join('token'+str(i) for i in range(100))
        a = prepare_record(raw_record(query=query), 'rl', 0, 'h')
        b = prepare_record(raw_record(query=query.replace('token50 ', 'replacement ')), 'rl', 1, 'h')
        result = group_records([a, b])
        self.assertEqual(a['group_id'], b['group_id'])
        self.assertEqual(result['near_pairs_above_threshold'], 1)


@unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'CPU pyarrow dependency not installed')
class FullBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)
        cls.before = {str(p): file_hash(p) for p in (ROOT/'dataset/rlla_4k_raw').glob('*.json')}
        cls.quality = build(cls.path/'data1', cls.path/'manifest1', root=ROOT, freeze=True)
        cls.version = json.loads((cls.path/'data1/version.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_raw_files_unchanged_and_every_occurrence_accounted(self):
        for p, digest in self.before.items():
            self.assertEqual(file_hash(p), digest)
        ledger = read_jsonl(self.path/'data1/lineage.jsonl')
        self.assertEqual(len(ledger), 4400)
        self.assertEqual(sum(sum(counts.values()) for counts in self.quality['actions'].values()), 4400)
        self.assertTrue(all(row['reasons'] for row in ledger if row['action'] in ('quarantine', 'delete_duplicate')))

    def test_train_validation_test_and_sft_rl_isolation(self):
        groups, prompts = defaultdict(set), defaultdict(set)
        for source in ['rl', 'sft']:
            for split in ['train', 'validation', 'test']:
                for row in read_jsonl(self.path/f'data1/{source}/{split}.jsonl'):
                    groups[split].add(row['group_id'])
                    prompts[split].add(row['metadata']['prompt_sha256'])
                    self.assertFalse(validate_calls(row['ground_truth'], row['tools']))
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            self.assertFalse(groups[a] & groups[b])
            self.assertFalse(prompts[a] & prompts[b])
        rl = {r['metadata']['prompt_sha256'] for r in read_jsonl(self.path/'data1/rl/train.jsonl')}
        sft = {r['metadata']['prompt_sha256'] for r in read_jsonl(self.path/'data1/sft/train.jsonl')}
        self.assertGreater(len(rl & sft), 0)  # Do not simply delete all cross-source overlap.

    def test_deterministic_rebuild_and_all_artifact_hashes(self):
        build(self.path/'data2', self.path/'manifest2', root=ROOT, freeze=True)
        second = json.loads((self.path/'data2/version.json').read_text())
        self.assertEqual(self.version, second)
        for path, digest in self.version['files_sha256'].items():
            self.assertEqual(file_hash(self.path/'data1'/path), digest)
        with self.assertRaises(ValueError):
            build(self.path/'data1', self.path/'manifest1', root=ROOT)

    def test_parquet_contract_and_no_label_in_rendered_input(self):
        import pyarrow.parquet as pq
        data = pq.read_table(self.path/'data1/grpo/train.parquet').to_pylist()
        originals = json.loads((ROOT/'dataset/rlla_4k_raw/rlla_rl.json').read_text())
        for row in data:
            original = originals[row['extra_info']['index']]
            self.assertEqual(row['prompt'], [{'role': 'system', 'content': original['instruction']}, {'role': 'user', 'content': original['input']}])
            self.assertEqual(row['reward_model']['style'], 'rule')
            self.assertTrue(row['reward_model']['ground_truth'])
            self.assertNotIn('ground_truth', row['prompt'][0])
            self.assertEqual(row['data_source'], 'rlla_phase2')

    def test_train_subset_cannot_use_held_out_ids(self):
        train = read_jsonl(self.path/'data1/rl/train.jsonl')
        val = read_jsonl(self.path/'data1/rl/validation.jsonl')
        self.assertEqual(len(select_training_subset(train, [train[0]['sample_id']])), 1)
        with self.assertRaises(ValueError):
            select_training_subset(train+val, [val[0]['sample_id']])


if __name__ == '__main__':
    unittest.main()
