"""Read-only model diagnostic for the anomalous two-window quantization smoke."""
import importlib.util
import argparse
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.eval_qwen3_c1_kvquant import MODEL, ROOT, UPSTREAM, quantized_output
from evaluation.eval_qwen3_8b_iclr_quality import install_c1_allocation
from evaluation.eval_attention_o_proj_collective_ppl import _token_ids
from evaluation.eval_qwen3_tp4_quant_ppl import write_progress
from evaluation.v96kl_common import read_json, sha256
from basisserve.core.qwen3_tp4_quant_quality import simulate_padded_wire


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prefix-probe', action='store_true')
    options = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = ROOT/'ICLR-results/qwen3-8b/checkpoints/Q3-8B-C1-R64'
    manifest = read_json(checkpoint/'manifest.json')
    params = torch.load(ROOT/'results/q3-kvquant/full/C1-R64/quantizers.pt', map_location='cpu', weights_only=False)
    source = UPSTREAM/'quant/kvquant/simquant_module_quantizer.py'
    spec = importlib.util.spec_from_file_location('kvquant_diag', source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').eval().cuda()
    assert install_c1_allocation(model, checkpoint, manifest) is not None
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    corpus = _token_ids(tokenizer, 'wikitext2', 'test', None).reshape(-1)
    windows = [0] if options.prefix_probe else [0, 1, 40, 80, 120, 145]
    inputs = {i: corpus[i*2048:(i+1)*2048][None].cuda() for i in windows}
    metrics, losses, current, handles = {}, {}, {}, []
    indices = {}
    for i in range(36):
        indices[f'{i}.k'] = torch.arange(1024, device='cuda')
        indices[f'{i}.v'] = torch.tensor([h*128+j for h, r in
            enumerate(manifest['compression']['layer_ranks'][i]) for j in range(r)], device='cuda')
    arms = (('bf16', 'v4-first1', 'v4-except1', 'wire4-first1', 'wire4-except1',
             'v4-first8', 'v4-except8') if options.prefix_probe else
            ('bf16', 'k4', 'v4', 'kv4', 'wire4', 'kv4-wire4', 'bf16-repeat'))
    for arm in arms:
        quant_k = arm in ('k4', 'kv4', 'kv4-wire4')
        quant_v = arm in ('v4', 'kv4', 'kv4-wire4') or arm.startswith('v4-')
        wire = arm in ('wire4', 'kv4-wire4') or arm.startswith('wire4-')
        def select_positions(original, quantized):
            if options.prefix_probe:
                n = 8 if arm.endswith('8') else 1
                quantized = quantized.clone()
                if '-first' in arm:
                    quantized[:, n:] = original[:, n:]
                elif '-except' in arm:
                    quantized[:, :n] = original[:, :n]
            return quantized
        for i, layer in enumerate(model.model.layers):
            for suffix, module, enabled in [('k', layer.self_attn.k_norm, quant_k),
                                             ('v', layer.self_attn.v_proj, quant_v)]:
                if enabled:
                    name = f'{i}.{suffix}'
                    handles.append(module.register_forward_hook(lambda m, x, y, name=name:
                        select_positions(y, quantized_output(upstream, params, indices, name, y, 8, 128))))
            rank = manifest['compression']['layer_ranks'][i][0]
            def output_hook(module, args, layer=i, rank=rank):
                x = args[0]
                q = select_positions(x, simulate_padded_wire(x, rank))
                # Same-input local quantization error, not propagation error.
                active = x.reshape(-1, 32, 128)[:, :, :rank].float()
                qactive = q.reshape(-1, 32, 128)[:, :, :rank].float()
                points = torch.linspace(0, x.shape[1]-1, 32, device=x.device).long()
                y = F.linear(x[:, points], module.weight).float()
                yq = F.linear(q[:, points], module.weight).float()
                local_groups = active.reshape(-1, 4, 8*rank)
                amax = local_groups.abs().amax(-1)
                rms = local_groups.square().mean(-1).sqrt().clamp_min(1e-20)
                current['layers'].append(dict(layer=layer, rank=rank,
                    absmax=active.abs().max().item(),
                    amax_over_rms=(amax/rms).mean().item(),
                    zero_fraction=(qactive == 0).float().mean().item(),
                    latent_rel_mse=((qactive-active).square().sum()/active.square().sum().clamp_min(1e-20)).item(),
                    output_rel_mse=((yq-y).square().sum()/y.square().sum().clamp_min(1e-20)).item()))
                return (q,) if wire else args
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(output_hook))
        metrics[arm], losses[arm] = {}, {}
        selected_windows = [0, 1] if arm in ('k4', 'v4', 'bf16-repeat') else windows
        for window in selected_windows:
            current = {'layers': []}
            tokens = inputs[window]
            logits = model(tokens, use_cache=False).logits
            loss = F.cross_entropy(logits[0, :-1].float(), tokens[0, 1:], reduction='none').cpu()
            assert torch.isfinite(loss).all()
            losses[arm][window] = loss
            current['ppl'] = math.exp(loss.mean().item())
            current['nll'] = loss.mean().item()
            current['position_nll'] = {f'{a}:{b}': loss[a:b].mean().item()
                                      for a, b in [(0, 32), (32, 128), (128, 512), (512, 2047)]}
            metrics[arm][window] = current
            print(arm, window, current['ppl'], current['position_nll'], flush=True)
            del logits
        for handle in handles:
            handle.remove()
        handles.clear()
    comparisons = {}
    for arm in arms[1:]:
        comparisons[arm] = {}
        for window, loss in losses[arm].items():
            delta = loss-losses['bf16'][window]
            top = delta.abs().topk(12).indices.tolist()
            comparisons[arm][window] = dict(delta_nll=delta.mean().item(),
                max_abs_loss_delta=delta.abs().max().item(),
                top_changes=[dict(position=t, delta=delta[t].item(),
                    baseline=losses['bf16'][window][t].item(), loss=loss[t].item(),
                    target=tokenizer.decode(inputs[window][0, t+1:t+2])) for t in top])
    write_progress(ROOT/'results/q3-kv-wire'/('prefix_probe.json' if options.prefix_probe else 'diagnostic.json'), dict(
        scope='first-window prefix intervention' if options.prefix_probe else
              'six-window HF diagnostic, NOT full PPL; k-only/v-only/repeat use first two',
        metrics=metrics, comparisons=comparisons,
        source_sha256=sha256(Path(__file__)), checkpoint_manifest_sha256=sha256(checkpoint/'manifest.json')))


if __name__ == '__main__':
    main()
