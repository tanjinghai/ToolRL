"""Small explicit configurations translated into the vendored Hydra interfaces."""
import json
from pathlib import Path
from toolrl_pipeline.io import loads, file_hash

ROOT = Path(__file__).resolve().parents[1]
DATA_VERSION = 'phase2-v1-13c266c7a084'
MODEL_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def load(path):
    text = Path(path).read_text()
    try:
        return loads(text)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(text)


def batch_report(c):
    n, dp = c['n'], c['gpus']
    sizes = [c[k] for k in ('batch', 'mini', 'micro', 'logprob_micro')]
    if any(type(x) is not int or x <= 0 or x % dp for x in sizes) or c['n'] < 2:
        raise ValueError('Positive integral batch sizes must divide data parallel world size; GRPO n>=2')
    batch, mini, micro, logprob = [x // dp * n for x in sizes]
    if batch % mini or mini % micro or batch % logprob:
        raise ValueError('Expanded minibatch/microbatch divisibility failure')
    if c['prompt_length'] + c['response_length'] > c['max_model_len']:
        raise ValueError('Prompt + response exceeds model context')
    if c['updates'] < 1 or c['epochs'] < 1 or c['gpus'] not in (1, 2):
        raise ValueError('Invalid training limits')
    if c['kl_mode'] not in ('paper-style', 'legacy-default') or c['reward_version'] not in ('legacy', 'strict_matching'):
        raise ValueError('Unknown KL/reward mode')
    return {'prompt_batch': c['batch'], 'responses_per_update': c['batch']*n,
            'per_rank_responses': batch, 'per_rank_mini': mini, 'per_rank_micro': micro,
            'per_rank_logprob_micro': logprob, 'accumulation': mini//micro,
            'optimizer_steps_per_rollout': batch//mini, 'effective_updates': c['updates']*(batch//mini),
            'reward_kl': 0. if c['kl_mode'] == 'paper-style' else .001, 'explicit_kl_loss': False,
            'std_correction': 1, 'entropy_coeff': .001,
            'memory_note': 'FP32 actor/grad/Adam ~24GB at 1.5B before reference, vLLM KV, activations; GPU unverified'}


def verify_data(data_root, manifests, version=DATA_VERSION):
    meta = loads((Path(manifests)/'version.json').read_text())
    if meta['dataset_version'] != version:
        raise ValueError('Wrong frozen data version')
    for rel, expected in meta['files_sha256'].items():
        if file_hash(Path(data_root)/rel) != expected:
            raise ValueError(f'Frozen data hash mismatch: {rel}')
    # Bind original builder, never update frozen manifests to fit new code.
    for rel, expected in meta['config']['code_hashes'].items():
        if file_hash(ROOT/'toolrl_pipeline'/rel) != expected:
            raise ValueError(f'Frozen builder changed: {rel}')
    return meta


def grpo_overrides(c, model, data_root, output):
    report = batch_report(c)
    values = {
        'algorithm.adv_estimator': 'grpo', 'algorithm.kl_ctrl.kl_coef': report['reward_kl'],
        'data.train_files': str(Path(data_root)/'grpo/train.parquet'),
        'data.val_files': str(Path(data_root)/'grpo/validation.parquet'),
        'data.max_prompt_length': c['prompt_length'], 'data.max_response_length': c['response_length'],
        'data.train_batch_size': c['batch'], 'data.val_batch_size': c['val_batch'],
        '+data.train_limit': c.get('train_limit'), '+data.truncation': 'error',
        'actor_rollout_ref.model.path': str(model),
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.ppo_mini_batch_size': c['mini'],
        'actor_rollout_ref.actor.ppo_micro_batch_size': c['micro'],
        'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.optim.lr': c['lr'],
        'actor_rollout_ref.ref.log_prob_micro_batch_size': c['logprob_micro'],
        'actor_rollout_ref.ref.fsdp_config.param_offload': c['reference_offload'],
        'actor_rollout_ref.rollout.log_prob_micro_batch_size': c['logprob_micro'],
        'actor_rollout_ref.rollout.n': c['n'],
        '+actor_rollout_ref.rollout.seed': c['seed'],
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.gpu_memory_utilization': c['vllm_utilization'],
        'actor_rollout_ref.rollout.max_num_batched_tokens': c['max_model_len'],
        'actor_rollout_ref.rollout.max_num_seqs': c['batch']*c['n'],
        '+actor_rollout_ref.rollout.max_model_len': c['max_model_len'],
        '+reward_model.reward_version': c['reward_version'],
        '+trainer.seed': c['seed'], 'trainer.n_gpus_per_node': c['gpus'],
        'trainer.nnodes': 1, 'trainer.total_epochs': c['epochs'],
        'trainer.total_training_steps': c['updates'], 'trainer.save_freq': c['save_freq'],
        'trainer.test_freq': c['test_freq'], '+trainer.val_before_train': False,
        'trainer.default_hdfs_dir': None, 'trainer.default_local_dir': str(output),
        'trainer.project_name': 'toolrl-phase3', 'trainer.experiment_name': Path(output).name,
        'trainer.logger': ['console'],
    }
    return [key + '=' + json.dumps(value) for key, value in values.items()]


def sft_overrides(c, model, data_root, output):
    if c['max_length'] < 2706 or c['batch'] % c['micro'] or c['batch'] % c['gpus'] or c['micro'] % c['gpus']:
        raise ValueError('Unsafe SFT length/batch')
    values = {'data.train_files': str(Path(data_root)/'sft/train.parquet'),
              'data.val_files': str(Path(data_root)/'sft/validation.parquet'),
              'data.prompt_key': 'prompt', 'data.response_key': 'response',
              'data.train_batch_size': c['batch'], 'data.micro_batch_size': c['micro'],
              'data.max_length': c['max_length'], 'data.truncation': 'error',
              'model.partial_pretrain': str(model), 'model.enable_gradient_checkpointing': True,
              'optim.lr': c['lr'], 'trainer.total_epochs': c['epochs'],
              '+trainer.max_steps': c['updates'], '+trainer.save_freq': c['save_freq'],
              'trainer.seed': c['seed'], 'trainer.default_local_dir': str(output),
              'trainer.default_hdfs_dir': None, 'trainer.project_name': 'toolrl-phase3',
              'trainer.experiment_name': Path(output).name, 'trainer.logger': ['console']}
    return [key + '=' + json.dumps(value) for key, value in values.items()]
