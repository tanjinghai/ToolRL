"""Unified CPU-first CLI. GPU inference is explicit and never runs during builds/tests."""
import argparse
import json
from pathlib import Path

from .benchmarks import api_bank_records
from .data import ROOT, build
from .evaluation import DEFAULT_GENERATION, generate, make_run, mock_backend, score
from .io import read_jsonl, write_jsonl


def _generation_args(parser):
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--generation-config', type=Path, default=ROOT/'configs/evaluation.json')
    parser.add_argument('--backend', choices=['mock', 'vllm'], default='mock')
    parser.add_argument('--mock-mode', choices=['mixed', 'oracle', 'invalid'], default='mixed')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--max-model-len', type=int, default=6144)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.5)
    parser.add_argument('--model-dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-final-test', action='store_true')


def _score_args(parser):
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--previous-results', type=Path)
    parser.add_argument('--run-metadata', type=Path)
    parser.add_argument('--allow-final-test', action='store_true')


def run_generation(args, dataset):
    records = read_jsonl(dataset)
    if any(r['split'] == 'test' for r in records) and not args.allow_final_test:
        raise ValueError('Use validation for development; --allow-final-test is required for final prediction generation')
    config = json.loads(args.generation_config.read_text()) if args.generation_config else dict(DEFAULT_GENERATION)
    if set(config) != set(DEFAULT_GENERATION) or config.get('n') != 1 or config['max_new_tokens'] <= 0:
        raise ValueError('Require complete single-attempt generation configuration (n=1)')
    if args.backend == 'mock':
        backend = mock_backend(records, args.mock_mode)
        backend_name = 'mock-'+args.mock_mode
        backend_config = {}
    else:
        if args.model_path is None or not args.model_path.is_dir():
            raise ValueError('vLLM requires an existing local --model-path; no implicit model download')
        from .generation_backends import VLLMBackend
        backend_config = {'max_model_len': args.max_model_len, 'tensor_parallel_size': args.tensor_parallel_size,
                          'gpu_memory_utilization': args.gpu_memory_utilization, 'dtype': args.model_dtype}
        backend = VLLMBackend(str(args.model_path), **backend_config)
        backend_name = 'vllm'
    run = make_run(dataset, args.run_id, args.model_id, args.model_revision, config, backend_name, backend_config)
    path = generate(dataset, args.output_dir, run, backend, resume=args.resume)
    return score(dataset, path, args.output_dir, run_metadata=run, allow_final_test=args.allow_final_test)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build-data')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifests', type=Path, required=True)
    p.add_argument('--seed', type=int, default=31415)
    p.add_argument('--bfcl-dir', type=Path)
    p.add_argument('--freeze', action='store_true')
    p = sub.add_parser('generate')
    p.add_argument('--dataset', type=Path, required=True)
    _generation_args(p)
    p = sub.add_parser('score')
    p.add_argument('--dataset', type=Path, required=True)
    _score_args(p)
    p = sub.add_parser('api-generate')
    _generation_args(p)
    p.add_argument('--dataset', type=Path)
    p = sub.add_parser('api-score')
    _score_args(p)
    p.add_argument('--dataset', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'build-data':
        report = build(args.output, args.manifests, args.seed, bfcl_dir=args.bfcl_dir, freeze=args.freeze)
        print(json.dumps({'dataset_version': report['dataset_version'], 'actions': report['actions'],
                          'split_counts': {k: v['rows'] for k, v in report['splits'].items()}}, ensure_ascii=False, indent=2))
        return
    if args.command.startswith('api-') and args.dataset is None:
        args.dataset = args.output_dir/'api_bank.jsonl'
        records, _ = api_bank_records(ROOT)
        write_jsonl(args.dataset, records)
    if args.command in ('generate', 'api-generate'):
        summary = run_generation(args, args.dataset)
    else:
        metadata_path = args.run_metadata or args.predictions.parent/'run.json'
        run = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        summary = score(args.dataset, args.predictions, args.output_dir, args.previous_results,
                        run_metadata=run, allow_final_test=args.allow_final_test)
    print(json.dumps({'run_id': summary['run_id'], 'is_mock': summary['is_mock'], 'samples': summary['samples'],
                      'invalid_ground_truth': summary['invalid_ground_truth'], 'metrics': summary['metrics']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
