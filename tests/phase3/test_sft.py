import os
from pathlib import Path
import unittest
from toolrl_pipeline.io import read_jsonl
from toolrl_pipeline.tokenization import load_local_tokenizer, encode_sft
from toolrl_training.config import ROOT
from toolrl_training.sft_data import adapt_sft

TOKENIZER=ROOT/'data_local/tokenizers/Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


class SFTAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=load_local_tokenizer(os.environ.get('TOOLRL_TOKENIZER_PATH',str(TOKENIZER)))

    def test_all_real_train_val_labels_and_padding(self):
        for split in ('train','validation'):
            records=read_jsonl(ROOT/f'dataset/generated/phase2-v1/sft/{split}.jsonl')
            for r in records:
                with self.subTest(split=split,id=r['sample_id']):
                    raw=encode_sft(r,self.tokenizer)
                    batch=adapt_sft(r['prompt'],r['target_output'],self.tokenizer,3072)
                    n,p=len(raw['input_ids']),raw['prompt_length']
                    self.assertEqual(batch['labels'][:p],[-100]*p)
                    self.assertEqual(batch['labels'][p:n],raw['input_ids'][p:])
                    self.assertEqual(batch['labels'][n:],[-100]*(3072-n))
                    self.assertEqual(batch['attention_mask'][n:],[0]*(3072-n))
                    self.assertEqual(batch['labels'][n-1],self.tokenizer.eos_token_id)
                    self.assertEqual(batch['input_ids'][p:n].count(self.tokenizer.eos_token_id),1)
                    self.assertEqual(self.tokenizer.decode(batch['input_ids'][p:n-1]),r['target_output'])
                    # Same native prompt as the live RL dataset, with no second template/BOS.
                    rendered=self.tokenizer.apply_chat_template(r['prompt'],tokenize=False,add_generation_prompt=True)
                    self.assertEqual(batch['input_ids'][:p],self.tokenizer.encode(rendered,add_special_tokens=False))

    def test_overlength_and_role_loss_rejected(self):
        r=read_jsonl(ROOT/'dataset/generated/phase2-v1/sft/train.jsonl')[0]
        with self.assertRaises(ValueError):
            adapt_sft(r['prompt'],r['target_output'],self.tokenizer,64)
        with self.assertRaises(ValueError):
            adapt_sft('flattened prompt',r['target_output'],self.tokenizer,3072)
        with self.assertRaises(ValueError):
            adapt_sft(r['prompt'],r['target_output']+self.tokenizer.eos_token,self.tokenizer,3072)
