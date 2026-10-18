"""Optional inference adapter. Importing this module does not import a GPU library."""


class VLLMBackend:
    """Future GPU path: implemented, not GPU-tested in Phase 2."""
    def __init__(self, model_path, max_model_len=6144, tensor_parallel_size=1, gpu_memory_utilization=0.5, dtype='bfloat16'):
        from vllm import LLM
        self.llm = LLM(model=model_path, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization, dtype=dtype)

    def __call__(self, request, config):
        from vllm import SamplingParams
        params = SamplingParams(temperature=config['temperature'], top_p=config['top_p'],
                                max_tokens=config['max_new_tokens'], seed=config['seed'], n=1, stop=config.get('stop'))
        # Tool declarations already live in system content. Do not pass native `tools=` too.
        result = self.llm.chat(request['prompt'], sampling_params=params, use_tqdm=False)[0]
        output = result.outputs[0]
        return {'raw_output': output.text, 'status': 'ok', 'finish_reason': output.finish_reason,
                'prompt_tokens': len(result.prompt_token_ids), 'completion_tokens': len(output.token_ids)}
