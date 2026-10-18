#!/usr/bin/env python3
"""Offline real-tokenizer length and assistant-label validation; never opens final Test."""
import argparse
import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from toolrl_pipeline.io import digest, file_hash, read_jsonl, write_json
from toolrl_pipeline.tokenization import encode_sft, load_local_tokenizer, render_prompt


def stats(values):
    values = sorted(values)
    return {str(q): values[round((len(values)-1)*q)] for q in [0, .5, .9, .95, .99, 1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tokenizer = load_local_tokenizer(args.tokenizer)
    report = {'tokenizer_files': {p.name: file_hash(p) for p in args.tokenizer.glob('*.json')},
              'template_sha256': digest(tokenizer.chat_template), 'tokenizer_class': type(tokenizer).__name__,
              'python': sys.version.split()[0], 'versions': {p: importlib.metadata.version(p) for p in ('transformers', 'tokenizers', 'jinja2')},
              'checks': {}, 'splits': {}, 'final_test_opened': False, 'model_weights_loaded': False}
    checked = 0
    for source in ('rl', 'sft'):
        for split in ('train', 'validation'):
            path = args.data_root/source/(split+'.jsonl')
            records = read_jsonl(path)
            prompt_lengths, target_lengths, totals = [], [], []
            for row in records:
                prefix = render_prompt(row['prompt'], tokenizer)
                prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
                direct = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True)
                assert prompt_ids == direct, 'Rendering/tokenization mismatch'
                assert prefix.endswith('<|im_start|>assistant\n'), 'Qwen assistant generation prefix mismatch'
                prompt_lengths.append(len(prompt_ids))
                target_lengths.append(len(tokenizer.encode(row['target_output'], add_special_tokens=False))+1)
                if source == 'sft':
                    encoded = encode_sft(row, tokenizer)
                    n = encoded['prompt_length']
                    assert encoded['labels'][:n] == [-100]*n
                    assert encoded['labels'][n:] == encoded['input_ids'][n:]
                    assert encoded['input_ids'][-1] == tokenizer.eos_token_id
                    assert encoded['input_ids'][n:].count(tokenizer.eos_token_id) == 1
                    full = tokenizer.encode(prefix+row['target_output']+tokenizer.eos_token, add_special_tokens=False)
                    assert encoded['input_ids'] == full, 'BPE boundary is not equivalent to full encoding'
                    native = tokenizer.apply_chat_template(row['prompt']+[{'role': 'assistant', 'content': row['target_output']}],
                                                           tokenize=False, add_generation_prompt=False)
                    assert native == prefix+row['target_output']+tokenizer.eos_token+'\n'
                    totals.append(len(encoded['input_ids']))
                    checked += 1
            report['splits'][source+'/'+split] = {
                'rows': len(records), 'file_sha256': file_hash(path), 'prompt_tokens': stats(prompt_lengths),
                'reference_target_tokens_including_eos': stats(target_lengths),
                'prompt_over_2048': sum(n > 2048 for n in prompt_lengths), 'prompt_over_4096': sum(n > 4096 for n in prompt_lengths),
                'sft_total_tokens': stats(totals) if totals else None,
                'sft_total_over_2048': sum(n > 2048 for n in totals) if totals else None}
    report['checks'] = {'real_qwen_template': True, 'assistant_only_labels_samples': checked,
                        'one_target_eos': True, 'no_double_special_tokens': True, 'full_encoding_bpe_boundary_match': True}
    write_json(args.output, report)
    print(f'PASS: {checked} SFT examples checked with real Qwen tokenizer; final Test unopened; no model weights loaded.')


if __name__ == '__main__':
    main()
