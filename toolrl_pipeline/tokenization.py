"""Model-free prompt/SFT preparation with the actual tokenizer's chat template."""
IGNORE_INDEX = -100


def render_prompt(prompt, tokenizer):
    if [m.get("role") for m in prompt] != ["system", "user"]:
        raise ValueError("Expected exactly system/user; do not pre-render or double-template")
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)


def encode_sft(record, tokenizer, max_length=None):
    """Unshifted labels: prompt=-100, assistant and one EOS supervised.

    The causal LM/trainer must shift these labels exactly once in Phase 3.
    No torch, padding, truncation or model weights are required.
    """
    prompt = render_prompt(record["prompt"], tokenizer)
    target = record["target_output"]
    if not isinstance(target, str) or not target:
        raise ValueError("Empty SFT target")
    reserved = [x for x in (tokenizer.eos_token, tokenizer.bos_token, "<|im_start|>") if x]
    if any(token in target for token in reserved):
        raise ValueError("Target already contains tokenizer control tokens")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    if max_length is not None and len(ids) > max_length:
        raise ValueError(f"SFT length {len(ids)} exceeds {max_length}; refusing silent truncation")
    if not target_ids or ids[-1] is None:
        raise ValueError("Missing target tokens/EOS")
    labels = [IGNORE_INDEX]*len(prompt_ids) + target_ids + [tokenizer.eos_token_id]
    assert len(ids) == len(labels)
    return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": labels,
            "prompt_length": len(prompt_ids), "target_length": len(target_ids)+1}


def load_local_tokenizer(path):
    # Optional dependency is imported lazily. No auto-download or model imports.
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
