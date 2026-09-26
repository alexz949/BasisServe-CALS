"""KVQuant NUQ4 quantizers for the deployed Llama V96 model (evaluation/kvquant_nuq4_simulation.py consumes them).
Stage `capture` (GPU): the identity checkpoint (compressed V96 encoder and decoder) is installed with the evaluation runtime,
every attention layer runs a differentiable copy of its prefill (SDPA with the zero-padded V latent), and 16 random 2048-token
WikiText-2 train segments (KVQuant's calibration set) are pushed through with gradients w.r.t. the input embeddings; per layer
the pre-RoPE Keys (k_proj output) and the V96 latent (installed v_proj output) are recorded with their Fisher information
(squared gradient of the summed next-token cross-entropy), as KVQuant's calibration does.  Stage `fit` (CPU, layer shards):
KVQuant's SimQuant.quantize — static per-channel thresholds and Fisher-weighted 16-signpost k-means for Keys (qchannel 0),
per-token for the Value latent (qchannel -1), 1% dense-and-sparse outliers.  Stage `merge` writes quantizers.pt + manifest.json."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

import torch

from evaluation.eval_k_routing_ruler import install
from evaluation.eval_k_routing_ruler_p16 import flash_prefill_attention, load_evaluation_model
from evaluation.k_routing_config import routing_config
from evaluation.kvquant_nuq4_simulation import BITS, SPARSITY_THRESHOLD, UPSTREAM, load_upstream
from evaluation.v96kl_common import configure, read_json, sha256, write_json

FORMAT = 'basisserve.llama_v96.kvquant_nuq4_quantizers.v1'
SEGMENTS, LENGTH, SEED = 16, 2048, 0


def calibration_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    batch, length, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(batch, length, self.num_attention_heads, self.head_dim).transpose(1, 2)
    pre = self.k_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.value_head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, pre, cos, sin)
    output = flash_prefill_attention(q, k, v, scale=self.scaling)
    return self.o_proj(output.transpose(1, 2).reshape(batch, length, -1)), None


def segments(tokenizer, count, length, seed):
    from datasets import load_dataset
    rows = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    ids = torch.tensor(tokenizer('\n\n'.join(rows['text']), add_special_tokens=False)['input_ids'])
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - length, (count,), generator=generator).tolist()
    return ids, starts


def capture(args, identity):
    from transformers import AutoTokenizer
    config = routing_config(identity, rope='native', sequence_length=131072)
    assert config.model_type == 'llama'
    model = load_evaluation_model(identity, config)
    checkpoint = Path(identity['checkpoint'])
    manifest = read_json(checkpoint / 'manifest.json')
    install(model, checkpoint, manifest, 'full', {})
    # install() runs under inference mode; its new parameters/buffers are inference tensors, which autograd refuses to save.
    for module in model.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is not None and parameter.is_inference():
                module._parameters[name] = torch.nn.Parameter(parameter.detach().clone(), requires_grad=False)
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.is_inference():
                module._buffers[name] = buffer.detach().clone()
    model.requires_grad_(False)
    layers = [int(x) for x in args.layers.split(',')] if args.layers else list(identity['attention_layers'])
    for index in identity['attention_layers']:
        # every layer needs the cache-free differentiable prefill; the installed routing forward requires a cache
        attention = model.model.layers[index].self_attn
        attention.forward = MethodType(calibration_forward, attention)
    modules = {}
    for index in layers:
        attention = model.model.layers[index].self_attn
        modules[f'{index}.k'] = attention.k_proj
        modules[f'{index}.v'] = attention.v_proj
    tokenizer = AutoTokenizer.from_pretrained(identity['model'], local_files_only=True)
    count, length = (1, 256) if args.smoke else (SEGMENTS, LENGTH)
    ids, starts = segments(tokenizer, count, length, SEED)
    records = {name: [] for name in modules}

    def hook(name, module, inputs, output):
        entry = [output.detach().float().reshape(-1, output.shape[-1]).cpu(), None]
        records[name].append(entry)

        def gradient(grad):
            entry[1] = grad.float().reshape(-1, grad.shape[-1]).square().cpu()
        output.register_hook(gradient)

    handles = [module.register_forward_hook(lambda m, x, y, name=name: hook(name, m, x, y)) for name, module in modules.items()]
    losses = []
    for i, start in enumerate(starts):
        tokens = ids[start:start + length][None].cuda()
        embedded = model.get_input_embeddings()(tokens).detach().requires_grad_(True)
        started = time.monotonic()
        logits = model(inputs_embeds=embedded, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction='sum')
        loss.backward()
        losses.append(float(loss.detach()))
        print(json.dumps(dict(stage='capture', segment=i, start=start, loss=round(losses[-1], 2), seconds=round(time.monotonic() - started, 1),
                              peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))), flush=True)
        del embedded, logits, loss, tokens
    for handle in handles:
        handle.remove()
    out = args.output / 'capture'
    out.mkdir(parents=True, exist_ok=True)
    for index in layers:
        tensors = {}
        for kind in ('k', 'v'):
            name = f'{index}.{kind}'
            tensors[kind] = torch.cat([r[0] for r in records[name]])
            fisher = torch.cat([r[1] for r in records[name]])
            assert torch.isfinite(tensors[kind]).all() and torch.isfinite(fisher).all() and fisher.sum() > 0
            tensors[f'{kind}_fisher'] = fisher / fisher.mean()
        path = out / f'layer_{index:03d}.pt'
        assert not path.exists(), path
        torch.save(tensors, path)
    write_json(out / 'manifest.json', dict(status='complete', identity_sha256=sha256(args.identity), checkpoint_manifest_sha256=identity['manifest_sha256'],
        model=identity['model'], model_config_sha256=identity['model_config_sha256'], layers=layers, smoke=args.smoke,
        calibration=dict(dataset='Salesforce/wikitext wikitext-2-raw-v1 train', joined='\\n\\n', special_tokens=False, segments=count, length=length, seed=SEED, starts=starts,
                         losses=losses, fisher='squared gradient of the summed next-token cross-entropy w.r.t. the module output, normalized by its mean per module'),
        capture=dict(key='k_proj output (pre-RoPE, Llama has no k_norm), [tokens, kv_heads x head_dim]', value='installed v_proj output = V96 latent, [tokens, kv_heads x 96]',
                     forward='differentiable prefill: SDPA with zero-padded V latent, GQA, causal; decoder in o_proj'),
        gpu=torch.cuda.get_device_name(), command=shlex.join(sys.argv)))


def fit(args, identity):
    upstream = load_upstream()
    capture_manifest = read_json(args.output / 'capture' / 'manifest.json')
    assert capture_manifest['status'] == 'complete' and capture_manifest['identity_sha256'] == sha256(args.identity)
    layers = capture_manifest['layers'][args.layer_shard::args.layer_shards]
    out = args.output / 'fit'
    out.mkdir(parents=True, exist_ok=True)
    for index in layers:
        path = out / f'layer_{index:03d}.pt'
        if path.exists():
            continue
        tensors = torch.load(args.output / 'capture' / f'layer_{index:03d}.pt', map_location='cpu', weights_only=True)
        quantizers = {}
        for kind in ('k', 'v'):
            started = time.monotonic()
            fitter = SimpleNamespace(out=tensors[kind], perchannel=True, qchannel=0 if kind == 'k' else -1, bits=BITS,
                                     nsamples=capture_manifest['calibration']['segments'])
            hi, lo, lut = upstream.SimQuant.quantize(fitter, include_sparse=True, sparsity_threshold=SPARSITY_THRESHOLD, nuq=True,
                                                     fisher=tensors[f'{kind}_fisher'], first_few_fp16=-1)
            quantizers[kind] = (hi, lo, lut)
            print(json.dumps(dict(stage='fit', layer=index, kind=kind, seconds=round(time.monotonic() - started, 1),
                                  lut=[round(float(x), 4) for x in sorted(lut[0].flatten().tolist())])), flush=True)
        torch.save(quantizers, path)


def merge(args, identity):
    capture_manifest = read_json(args.output / 'capture' / 'manifest.json')
    quantizers, hashes = {}, {}
    for index in capture_manifest['layers']:
        path = args.output / 'fit' / f'layer_{index:03d}.pt'
        fitted = torch.load(path, map_location='cpu', weights_only=False)
        for kind in ('k', 'v'):
            quantizers[f'{index}.{kind}'] = fitted[kind]
        hashes[path.name] = sha256(path)
    destination = args.output / 'quantizers.pt'
    assert not destination.exists(), destination
    torch.save(quantizers, destination)
    write_json(args.output / 'manifest.json', dict(status='complete', format=FORMAT, identity_sha256=capture_manifest['identity_sha256'],
        checkpoint_manifest_sha256=capture_manifest['checkpoint_manifest_sha256'], model=capture_manifest['model'], layers=capture_manifest['layers'],
        smoke=capture_manifest['smoke'], calibration=capture_manifest['calibration'], capture=capture_manifest['capture'],
        protocol=dict(bits=BITS, sparsity_threshold=SPARSITY_THRESHOLD, first_few_fp16=-1, rotation='none', nuq='16 k-means signposts per layer and tensor, Fisher-weighted',
                      key='pre-RoPE K, static per-channel thresholds and range, 1% dense-and-sparse outliers; RoPE after dequantization',
                      value='V96 latent, dynamic per-token range across KV heads, 1% outliers', packed_cache=False),
        quantizers_sha256=sha256(destination), fit_sha256=hashes,
        upstream=dict(repo='https://github.com/SqueezeAILab/KVQuant', commit=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=UPSTREAM.parents[2]).stdout.strip(),
                      module='quant/kvquant/simquant_module_quantizer.py', module_sha256=sha256(UPSTREAM)),
        source_sha256={n: sha256(Path(__file__).resolve().parents[1] / n) for n in ('evaluation/calibrate_llama_v96_kvquant.py', 'evaluation/kvquant_nuq4_simulation.py')},
        command=shlex.join(sys.argv)))
    print(json.dumps(dict(stage='merge', quantizers=len(quantizers), file=str(destination))), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=('capture', 'fit', 'merge'))
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layers', help='capture: comma list (default every attention layer)')
    p.add_argument('--layer-shard', type=int, default=0)
    p.add_argument('--layer-shards', type=int, default=1)
    p.add_argument('--smoke', action='store_true', help='capture: one 256-token segment')
    args = p.parse_args()
    configure()
    torch.set_num_threads(8)
    identity = read_json(args.identity)
    assert identity['status'] == 'complete'
    dict(capture=capture, fit=fit, merge=merge)[args.stage](args, identity)


if __name__ == '__main__':
    main()
