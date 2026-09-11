"""Uniform C1 V96 ShadowKV on the frozen local KL96 RULER prompts."""
import argparse
from pathlib import Path
import shlex
import sys
import time
from types import MethodType

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import configure, read_json, write_json, sha256, MODEL
from evaluation import eval_llama31_shadow as llama
from evaluation.eval_llama31_shadow_v96 import forward as llama_forward
from evaluation.eval_ruler_kl96 import shadow_generate
from basisserve.checkpoint.gqa_vo_qwen3 import GQATiedVOQwen3Attention
from basisserve.checkpoint.c1_shadowkv_qwen3 import install_c1_shadowkv
from evaluation.eval_qwen3_8b_residual_rank_ruler import _eos_ids
from evaluation.ruler_v1 import sample_score

REVISION = 'f1a6253b5d5c747a2475cbf9e704a67d97930b31'


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-family', choices=['qwen3', 'llama31'], required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(0)
    is_qwen = args.model_family == 'qwen3'
    model_path = MODEL if is_qwen else llama.MODEL
    name = 'Q3-8B-C1U-R96' if is_qwen else 'L31-8B-C1U-R96'
    checkpoint = ROOT / f'results/hf/ICLR-results/{args.model_family}-8b/checkpoints/{name}'
    reference = ROOT / ('results/evaluation/ruler_kl96_seed42' if is_qwen else
                        'results/evaluation/llama31_base_shadow_v96')
    output = ROOT / f'results/evaluation/{args.model_family}_uniform96_shadow'
    manifest = read_json(checkpoint / 'manifest.json')
    assert manifest['status'] == 'complete'
    assert manifest['compression']['allocation'] == 'uniform_per_layer_per_head'
    assert manifest['model']['config_sha256'] == sha256(model_path / 'config.json')
    assert manifest['model']['safetensors_index_sha256'] == sha256(model_path / 'model.safetensors.index.json')
    assert sha256(checkpoint / manifest['artifact']['file']) == manifest['artifact']['sha256']
    layers = 36 if is_qwen else 32
    assert len(manifest['layers']) == layers
    for i, record in enumerate(manifest['layers']):
        assert record['layer'] == i and record['ranks'] == [96] * 8
        assert sha256(checkpoint / record['file']) == record['sha256']
    prior_summary = read_json(reference / 'result.json')
    assert prior_summary['status'] == 'complete'
    previous = [read_json(reference / 'shadowkv/evaluate' / f'sample_{i:03d}.json') for i in range(88)]
    assert all(d['status'] == 'complete' and d['protocol'] == prior_summary['protocol'] for d in previous)
    rows = [d['sample'] for d in previous]
    assert [r['index'] for r in rows] == list(range(88))
    spec = dict(model=str(model_path), checkpoint=str(checkpoint), hf_revision=REVISION,
                checkpoint_sha256=sha256(checkpoint / 'manifest.json'),
                reference_summary_sha256=sha256(reference / 'result.json'),
                reference_sample_sha256={str(i): sha256(reference / 'shadowkv/evaluate' / f'sample_{i:03d}.json') for i in range(88)},
                dtype='bfloat16', allocation='uniform_per_layer_per_head', rank_v=96,
                seed=0, data_seed=42, generation='greedy, frozen original task caps and EOS',
                prefill='full causal C1 Triton', rank_k=160, chunk=8, routed=2048,
                outlier_chunks=48, local='last4 full prompt chunks plus remainder and all generated tokens',
                primary_excluded_indices=[86], reference=str(reference),
                code_sha256={p: sha256(ROOT / p) for p in [
                    'evaluation/eval_uniform96_shadow.py', 'evaluation/eval_llama31_shadow.py',
                    'evaluation/eval_llama31_shadow_v96.py', 'evaluation/eval_ruler_kl96.py',
                    'evaluation/eval_longbench_c1_twosided_denseprefill.py',
                    'basisserve/core/c1_shadowkv.py', 'basisserve/checkpoint/c1_shadowkv_qwen3.py',
                    'basisserve/checkpoint/gqa_vo_qwen3.py', 'basisserve/kernels/compressed_v_decode_attention.py']})
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                local_files_only=True, attn_implementation='sdpa')
    if is_qwen:
        model = model.cuda().eval()
        for layer, record in zip(model.model.layers, manifest['layers'], strict=True):
            m = layer.self_attn
            tensors = load_file(str(checkpoint / record['file']))
            encoder, decoder = tensors['value_coordinate_encoders'], tensors['head_output_decoders']
            assert encoder.shape == (8, 128, 96) and decoder.shape == (32, 96, 4096)
            assert encoder.dtype == decoder.dtype == torch.bfloat16
            assert torch.isfinite(encoder).all() and torch.isfinite(decoder).all()
            assert m.v_proj.bias is None and m.o_proj.bias is None
            weight = torch.bmm(encoder.cuda().float().transpose(1, 2), m.v_proj.weight.float().reshape(8, 128, 4096)).reshape(768, 4096)
            out_weight = decoder.cuda().float().permute(2, 0, 1).reshape(4096, 3072)
            layer.self_attn = GQATiedVOQwen3Attention(m, v_proj_compressed_weight=weight,
                o_decoder_weight=out_weight, attention_backend='sdpa', value_coordinate_encoder=encoder)
            assert layer.self_attn.q_norm is m.q_norm and layer.self_attn.k_norm is m.k_norm
            assert torch.equal(layer.self_attn.v_proj.weight, weight.bfloat16())
            assert torch.equal(layer.self_attn.o_proj.weight, out_weight.bfloat16())
        install_c1_shadowkv(model)
    else:
        for layer, record in zip(model.model.layers, manifest['layers'], strict=True):
            m = layer.self_attn
            tensors = load_file(str(checkpoint / record['file']))
            encoder, decoder = tensors['value_coordinate_encoders'], tensors['head_output_decoders']
            assert encoder.shape == (8, 128, 96) and decoder.shape == (32, 96, 4096)
            assert encoder.dtype == decoder.dtype == torch.bfloat16
            assert torch.isfinite(encoder).all() and torch.isfinite(decoder).all()
            assert m.v_proj.bias is None and m.o_proj.bias is None
            weight = torch.bmm(encoder.float().transpose(1, 2), m.v_proj.weight.float().reshape(8, 128, 4096)).reshape(768, 4096)
            out_weight = decoder.float().permute(2, 0, 1).reshape(4096, 3072)
            m.v_proj = torch.nn.Linear(4096, 768, bias=False, dtype=torch.bfloat16)
            m.o_proj = torch.nn.Linear(3072, 4096, bias=False, dtype=torch.bfloat16)
            m.v_proj.weight.copy_(weight)
            m.o_proj.weight.copy_(out_weight)
            m.value_head_dim = 96
            m.shadow_enabled = True
            m.forward = MethodType(llama_forward, m)
        model = model.cuda().eval()
    eos = _eos_ids(tokenizer, model)

    def run(row, cap):
        if is_qwen:
            ids, first, stats, _ = shadow_generate(model, tokenizer, torch.tensor(row['input_ids']), cap)
            return ids, first, stats
        return llama.generate(model, tokenizer, row, cap)

    for stage, selected in [('smoke', [rows[64], rows[0]]), ('evaluate', rows[64:72] + rows[:64] + rows[72:])]:
        for row in selected:
            path = output / stage / f"sample_{row['index']:03d}.json"
            if path.exists():
                saved = read_json(path)
                assert saved['status'] == 'complete' and saved['protocol'] == spec and saved['sample'] == row
                continue
            cap = min(4, row['maximum_tokens']) if stage == 'smoke' else row['maximum_tokens']
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            ids, first, stats = run(row, cap)
            if stage == 'smoke':
                again, first2, _ = run(row, cap)
                assert ids == again
                torch.testing.assert_close(first, first2, atol=0, rtol=0)
            prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result = dict(ids=ids, prediction=prediction, score=sample_score(prediction, row['answers'], row['match_type']),
                          routing=stats, seconds=time.monotonic()-started, peak_gib=torch.cuda.max_memory_allocated()/2**30)
            write_json(path, dict(status='complete', protocol=spec, sample=row, result=result,
                                 command=shlex.join(sys.argv), python=sys.executable, gpu=torch.cuda.get_device_name(0)))
            print(stage, row['index'], row['task'], result['score'], result['seconds'], flush=True)
    scores = []
    for row in rows:
        saved = read_json(output / 'evaluate' / f"sample_{row['index']:03d}.json")
        assert saved['status'] == 'complete' and saved['protocol'] == spec and saved['sample'] == row
        r = saved['result']; ids = r['ids']
        assert 0 < len(ids) <= row['maximum_tokens'] and not any(i in eos for i in ids[:-1])
        assert ids[-1] in eos or len(ids) == row['maximum_tokens']
        assert tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) == r['prediction']
        assert sample_score(r['prediction'], row['answers'], row['match_type']) == r['score']
        scores.append(r['score'])
    tasks = {t: 100*sum(scores[r['index']] for r in rows if r['task'] == t)/8 for t in dict.fromkeys(r['task'] for r in rows)}
    result = dict(status='complete', verified=88, protocol=spec, mean87=100*sum(s for i,s in enumerate(scores) if i != 86)/87,
                  mean88=100*sum(scores)/88, tasks=tasks, reference_mean87=prior_summary['means87']['shadowkv'])
    write_json(output / 'result.json', result)
    print('VERIFIED', {k:v for k,v in result.items() if k != 'protocol'}, flush=True)


if __name__ == '__main__':
    main()
