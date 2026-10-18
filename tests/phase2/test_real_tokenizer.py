import os
from pathlib import Path
import unittest

from toolrl_pipeline.protocol import render_completion
from toolrl_pipeline.tokenization import encode_sft, load_local_tokenizer, render_prompt


@unittest.skipUnless(os.environ.get('TOOLRL_TOKENIZER_PATH'), 'Real tokenizer not supplied; run check_tokenizer.py separately')
class RealQwenTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_local_tokenizer(Path(os.environ['TOOLRL_TOKENIZER_PATH']))

    def test_original_messages_exactly_once_and_labels(self):
        row = {'prompt': [{'role': 'system', 'content': 'Tools: f(x: int).'}, {'role': 'user', 'content': 'Use f with x=1.'}],
               'target_output': render_completion([{'name': 'f', 'parameters': {'x': 1}}], think='Original thought retained.')}
        t = self.tokenizer
        result = encode_sft(row, t)
        prefix = render_prompt(row['prompt'], t)
        self.assertEqual(result['input_ids'], t.encode(prefix+row['target_output']+t.eos_token, add_special_tokens=False))
        n = result['prompt_length']
        self.assertEqual(result['labels'][:n], [-100]*n)
        self.assertEqual(result['labels'][n:], result['input_ids'][n:])
        self.assertEqual(result['input_ids'][n:].count(t.eos_token_id), 1)
        self.assertEqual(prefix.count('<|im_start|>assistant'), 1)

    def test_length_rejects_silent_truncation(self):
        row = {'prompt': [{'role': 'system', 'content': 'A system.'}, {'role': 'user', 'content': 'A query.'}],
               'target_output': render_completion([])}
        with self.assertRaises(ValueError):
            encode_sft(row, self.tokenizer, max_length=2)


if __name__ == '__main__':
    unittest.main()
