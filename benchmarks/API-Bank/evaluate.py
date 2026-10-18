"""Phase 2 API-Bank entry: paths are explicit and failures stay in the denominator.

See docs/phase2/benchmark_adapters.md for strict vs historical soft-match semantics.
No vLLM/GPU library is imported unless --backend vllm is explicitly selected.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from toolrl_pipeline.cli import main

if __name__ == '__main__':
    main(['api-score', *sys.argv[1:]])
