"""Runtime guards shared with actual GPU workers, testable with simple namespaces."""

def validate_worker_batches(config, dp, actor, rollout, reference):
    sizes = []
    if actor:
        sizes += [config.actor.ppo_mini_batch_size, config.actor.ppo_micro_batch_size]
    if rollout:
        sizes += [config.rollout.log_prob_micro_batch_size]
    if reference:
        sizes += [config.ref.log_prob_micro_batch_size]
    if any(type(x) is not int or x < dp or x % dp for x in sizes):
        raise ValueError('Worker batch sizes must divide DP before rollout.n expansion')


def sft_training_signature(config):
    from toolrl_pipeline.io import digest, file_hash
    from .config import ROOT
    code = {rel: file_hash(ROOT/rel) for rel in ('verl/trainer/fsdp_sft_trainer.py',
            'verl/utils/dataset/sft_dataset.py', 'toolrl_training/sft_data.py',
            'toolrl_training/tensor_ops.py', 'toolrl_pipeline/tokenization.py')}
    return digest({'code': code, 'data': {k: v for k, v in config['data'].items() if k not in ('train_files', 'val_files')},
                   'files': {k: file_hash(config['data'][k]) for k in ('train_files', 'val_files')},
                   'optim': config['optim'], 'epochs': config['trainer']['total_epochs'],
                   'max_steps': config['trainer'].get('max_steps'), 'seed': config['trainer']['seed'],
                   'gradient_checkpointing': config['model']['enable_gradient_checkpointing']})
