"""Same-checkpoint NUQ4 K/V and INT4/FP8 wire WT2 PPL on real TP4."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig

from basisserve.core.qwen3_tp4_quant_quality import (
    install_quant_quality, vocab_parallel_nll, simulate_padded_wire,
    gather_e4m3, gather_rows,
)
from basisserve.kernels.fp8_wire import FP8_E4M3_MAX, quantize_e4m3_static
from evaluation.eval_qwen3_c1_kvquant import MODEL, ROOT, UPSTREAM
from evaluation.eval_attention_o_proj_collective_ppl import _token_ids
from evaluation.v96kl_common import read_json, sha256

ARMS = {'bf16': (False, False, 'bf16'), 'kv4': (True, True, 'bf16'),
        'wire4': (False, False, 'int4'), 'kv4-wire4': (True, True, 'int4'),
        'v4': (False, True, 'bf16'), 'v4-wire8': (False, True, 'e4m3'),
        'kv4-wire8': (True, True, 'e4m3')}


def write_progress(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2)+'\n')
    temporary.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--rank', choices=(64, 96), type=int, default=64)
    parser.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=['bf16', 'kv4', 'wire4', 'kv4-wire4'])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--hf-reference', action='store_true',
                        help='single-GPU smoke-only oracle; wire uses virtual TP4 groups')
    args = parser.parse_args()
    source_sha256 = sha256(Path(__file__))
    runtime_sha256 = sha256(ROOT/'basisserve/core/qwen3_tp4_quant_quality.py')
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    checkpoint = ROOT/'ICLR-results/qwen3-8b/checkpoints'/f'Q3-8B-C1-R{args.rank}'
    calibration = ROOT/'results/q3-kvquant/full'/f'C1-R{args.rank}'
    output_root = args.output_dir or ROOT/'results/q3-kv-wire'/f'R{args.rank}'
    out = output_root/('smoke' if args.smoke else 'full')
    manifest = read_json(checkpoint/'manifest.json')
    prior = read_json(calibration/'result.json')
    assert prior['checkpoint_manifest_sha256'] == sha256(checkpoint/'manifest.json')
    assert manifest['model']['config_sha256'] == sha256(MODEL/'config.json')
    allocation = read_json(checkpoint/'result.json')
    assert sha256(checkpoint/'result.json') == manifest['artifact']['sha256']
    for record in allocation['selected_artifacts'].values():
        assert sha256(checkpoint/record['file']) == record['sha256']
    quantizers = torch.load(calibration/'quantizers.pt', map_location='cpu', weights_only=False)
    source = UPSTREAM/'quant/kvquant/simquant_module_quantizer.py'
    assert sha256(source) == prior['upstream_sha256']
    spec = importlib.util.spec_from_file_location('kvquant_official', source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    if args.hf_reference:
        assert args.smoke
        assert all(ARMS[arm][2] != 'e4m3' for arm in args.arms)
        from evaluation.eval_qwen3_8b_iclr_quality import install_c1_allocation
        from evaluation.eval_qwen3_c1_kvquant import quantized_output
        from evaluation.eval_attention_o_proj_collective_ppl import _eval_ppl_fp32_loss
        reference = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True).eval().cuda()
        assert install_c1_allocation(reference, checkpoint, manifest) is not None
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        hf_results = {}
        for arm in args.arms:
            handles = []
            if any(ARMS[arm][:2]):
                indices = {}
                for i, layer in enumerate(reference.model.layers):
                    indices[f'{i}.k'] = torch.arange(1024, device='cuda')
                    indices[f'{i}.v'] = torch.tensor([h*128+j for h, rank in
                        enumerate(manifest['compression']['layer_ranks'][i]) for j in range(rank)], device='cuda')
                    for suffix, module in (('k', layer.self_attn.k_norm), ('v', layer.self_attn.v_proj)):
                        if not ARMS[arm][0 if suffix == 'k' else 1]:
                            continue
                        name = f'{i}.{suffix}'
                        handles.append(module.register_forward_hook(lambda m, x, y, name=name:
                            quantized_output(upstream, quantizers, indices, name, y, 8, 128)))
            if ARMS[arm][2] == 'int4':
                for i, layer in enumerate(reference.model.layers):
                    ranks = manifest['compression']['layer_ranks'][i]
                    assert len(set(ranks)) == 1
                    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                        lambda m, x, rank=ranks[0]: (simulate_padded_wire(x[0], rank),)))
            hf_results[arm] = _eval_ppl_fp32_loss(reference, tokenizer, dataset='wikitext2', split='test',
                seqlen=2048, batch_size=1, max_samples=2, max_tokens=None)
            for handle in handles:
                handle.remove()
        write_progress(out/'hf_reference.json', hf_results)
        return
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation='sdpa',
        distributed_config=DistributedConfig(tp_size=4), local_files_only=True).eval()
    assert dist.get_world_size() == 4
    if args.smoke:
        torch.manual_seed(45)
        all_logits = torch.randn(5, 44, device='cuda')
        labels = torch.tensor([0, 10, 11, 32, 43], device='cuda')
        local_logits = all_logits[:, dist.get_rank()*11:(dist.get_rank()+1)*11]
        torch.testing.assert_close(vocab_parallel_nll(local_logits, labels),
            F.cross_entropy(all_logits, labels, reduction='sum'), rtol=1e-6, atol=1e-5)
        sample = torch.randn(3, 16, device='cuda', dtype=torch.bfloat16)*(dist.get_rank()+1)
        scales = torch.tensor([.02, .03, .04, .05], device='cuda')
        expected = gather_rows((quantize_e4m3_static(sample, scales[dist.get_rank()]).float()*scales[dist.get_rank()]).to(sample.dtype))
        torch.testing.assert_close(gather_e4m3(sample, scales), expected, rtol=0, atol=0)
    modules = install_quant_quality(model, checkpoint, quantizers, upstream)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    fp8_calibration = None
    fp8_arms = [arm for arm in args.arms if ARMS[arm][2] == 'e4m3']
    if fp8_arms:
        # Each experiment calibrates one frozen upstream K/V setting.
        assert len(fp8_arms) == 1
        calibration_k, calibration_v, _ = ARMS[fp8_arms[0]]
        calibration_path = output_root/'fp8_scales.json'
        identity = dict(checkpoint_manifest_sha256=sha256(checkpoint/'manifest.json'),
                        quantizer_sha256=sha256(calibration/'quantizers.pt'),
                        kv_mode='K+V NUQ4' if calibration_k else 'V NUQ4; K BF16', tp=4,
                        runtime_sha256=runtime_sha256,
                        dataset='WT2 train', seqlen=2048,
                        starts=prior['protocol']['calibration_starts'])
        if calibration_path.exists():
            fp8_calibration = read_json(calibration_path)
            assert fp8_calibration['identity'] == identity
            all_scales = torch.tensor(fp8_calibration['scales'], device='cuda', dtype=torch.float32)
        else:
            train = _token_ids(tokenizer, 'wikitext2', 'train', None).reshape(-1)
            for module in modules:
                module.quant_k, module.quant_v, module.wire = calibration_k, calibration_v, 'bf16'
                module.observe_wire = True
                module.wire_amax.zero_()
            for i, start in enumerate(identity['starts']):
                for module in modules:
                    module.reset_cache()
                model.model(input_ids=train[start:start+2048][None].cuda(), use_cache=False)
                if dist.get_rank() == 0:
                    print(f'FP8_CALIBRATION {i+1}/{len(identity["starts"])}', flush=True)
            local_amax = torch.stack([module.wire_amax for module in modules])
            gathered_amax = [torch.empty_like(local_amax) for _ in range(4)]
            dist.all_gather(gathered_amax, local_amax)
            all_amax = torch.stack(gathered_amax, -1)
            all_scales = (all_amax/FP8_E4M3_MAX).clamp_min(1e-30)
            fp8_calibration = dict(identity=identity, scales=all_scales.cpu().tolist(),
                                   observed_amax=all_amax.cpu().tolist(),
                                   rule='per-layer/source absmax over frozen training windows / 448')
            if dist.get_rank() == 0:
                write_progress(calibration_path, fp8_calibration)
            dist.barrier()
        assert all_scales.shape == (36, 4) and torch.isfinite(all_scales).all() and (all_scales > 0).all()
        for module, scales in zip(modules, all_scales):
            module.observe_wire = False
            module.wire_scales.copy_(scales)
    tokens = _token_ids(tokenizer, 'wikitext2', 'test', None).reshape(-1)
    seqlen = 2048
    windows = min(2, tokens.numel()//seqlen) if args.smoke else tokens.numel()//seqlen
    assert model.lm_head.weight.shape[0]*4 == model.config.vocab_size
    results = {}
    for arm in args.arms:
        started = time.perf_counter()
        for module in modules:
            module.quant_k, module.quant_v, module.wire = ARMS[arm]
            module.wire_clipped.zero_()
            module.wire_elements = 0
        total_nll, count = 0., 0
        for i in range(windows):
            for module in modules:
                module.reset_cache()
            batch = tokens[i*seqlen:(i+1)*seqlen][None].cuda()
            hidden = model.model(input_ids=batch, use_cache=False).last_hidden_state[0]
            nll = torch.zeros((), device='cuda', dtype=torch.float32)
            for start in range(0, seqlen-1, 128):
                stop = min(start+128, seqlen-1)
                logits = F.linear(hidden[start:stop], model.lm_head.weight)
                nll += vocab_parallel_nll(logits, batch[0, start+1:stop+1])
            assert torch.isfinite(nll)
            total_nll += nll.item()
            count += seqlen-1
            if dist.get_rank() == 0:
                print(f'{arm} {i+1}/{windows} nll={total_nll/count:.6f} ppl={math.exp(total_nll/count):.6f}', flush=True)
            del hidden, logits, batch
        clipped = torch.stack([module.wire_clipped for module in modules])
        dist.all_reduce(clipped)
        wire_elements = sum(module.wire_elements for module in modules)*4
        results[arm] = dict(ppl=math.exp(total_nll/count), nll_sum=total_nll,
                            quant_k=ARMS[arm][0], quant_v=ARMS[arm][1], wire=ARMS[arm][2],
                            fp8_clipped_elements=int(clipped.sum()), fp8_total_elements=wire_elements,
                            tokens=count, windows=windows, wall_seconds=time.perf_counter()-started)
        if dist.get_rank() == 0:
            write_progress(out/'results.json', dict(
                status='complete' if len(results)==len(args.arms) else 'running',
                results=results, requested_arms=args.arms,
                protocol=dict(model=str(MODEL), tp=4, dtype='bfloat16', batch_size=1,
                    dataset='WT2 test', seqlen=seqlen, smoke=args.smoke,
                    kv='official NUQ4 + percentile outliers, quantize/dequantize simulation',
                    v_groups='per-token across ALL 8 KV heads; auxiliary reference-only BF16 gather',
                    wire='actual uint8 NCCL: INT4 with dynamic scale or E4M3 with frozen static source scales',
                    packed_kv_cache=False, performance_benchmark=False,
                    calibration='frozen KV quantizers; FP8 scales recalibrated on matching K/V-quantized WT2 train only when selected',
                    fp8_calibration=fp8_calibration,
                    first_token_policy='same quantization rules as every other token; no exclusion',
                    source_sha256=source_sha256,
                    runtime_sha256=runtime_sha256,
                    checkpoint_manifest_sha256=sha256(checkpoint/'manifest.json'),
                    quantizer_sha256=sha256(calibration/'quantizers.pt'),
                    factor_sha256=[m.factor_sha256 for m in modules],
                    evaluation_token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest())))
    for module in modules:
        module.reset_cache()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
