"""Run routing evaluation with explicitly recorded host-mediated GPU transfers."""
from pathlib import Path
from evaluation import eval_k_routing_ruler as runner
from evaluation.host_staged_dispatch import install_host_staged_dispatch


def main():
    install_host_staged_dispatch()
    original_inputs = runner.inputs
    def inputs(*args, **kwargs):
        result = original_inputs(*args, **kwargs)
        spec = result[-1]
        spec['inter_gpu_transfer'] = 'synchronous GPU-to-CPU-to-GPU via Accelerate dispatch'
        for name in ['evaluation/host_staged_dispatch.py', 'evaluation/eval_k_routing_host_transfer.py']:
            spec['source_sha256'][name] = runner.sha256(Path(name))
        return result
    runner.inputs = inputs
    runner.main()


if __name__ == '__main__':
    main()
