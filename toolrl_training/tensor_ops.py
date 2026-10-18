"""Tensor objective shared by live SFT and CPU gradient tests."""

def causal_sft_loss(logits, labels, normalizer=None, dp_size=1):
    import torch.nn.functional as F
    shifted_labels = labels[..., 1:].contiguous()
    valid = (shifted_labels != -100).sum()
    denom = valid if normalizer is None else normalizer
    if denom.item() <= 0:
        raise ValueError('No supervised assistant tokens')
    return F.cross_entropy(logits[..., :-1, :].float().reshape(-1, logits.shape[-1]),
                           shifted_labels.reshape(-1), ignore_index=-100, reduction='sum') / denom * dp_size
