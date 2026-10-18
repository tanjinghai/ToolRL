"""Shared CPU-testable adapter used by the actual veRL SFT Dataset."""
from toolrl_pipeline.tokenization import encode_sft


def adapt_sft(prompt, response, tokenizer, max_length):
    # Arrow/pandas may return an ndarray rather than a list of role dictionaries.
    if hasattr(prompt, 'tolist'):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        raise ValueError('SFT prompt must preserve system/user messages')
    encoded = encode_sft({'prompt': prompt, 'target_output': response}, tokenizer, max_length)
    padding = max_length - len(encoded['input_ids'])
    if tokenizer.pad_token_id is None:
        raise ValueError('Explicit padding token required')
    encoded['input_ids'] += [tokenizer.pad_token_id] * padding
    encoded['attention_mask'] += [0] * padding
    encoded['labels'] += [-100] * padding
    position, positions = 0, []
    for active in encoded['attention_mask']:
        positions.append(position if active else 0)
        position += active
    return {key: encoded[key] for key in ('input_ids', 'attention_mask', 'labels')} | {'position_ids': positions}
