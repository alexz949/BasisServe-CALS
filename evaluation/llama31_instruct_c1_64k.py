"""C4 preparation and full-token C1 covariance capture for Llama Instruct 64K."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import shlex
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.v96kl_common import sha256, read_json, write_json, tensor_hash

MODEL = Path.home() / '.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
RUN = ROOT / 'results/llama31i_v96_64k'
C4 = Path.home() / '.cache/huggingface/hub/datasets--allenai--c4/snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en'
FIT, DIAGNOSTIC, LENGTH = 64, 16, 65536


def configure():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def install_chunked_mlp(model):
    """Keep the original tokenwise MLP, with bounded intermediate allocations."""
    for layer in model.layers:
        original = layer.mlp.forward
        def forward(x, original=original):
            return torch.cat([original(piece) for piece in x.split(1024, dim=1)], dim=1)
        layer.mlp.forward = forward


def prepare(args):
    directory = args.run / 'windows'
    protocol = dict(model_config_sha256=sha256(args.model/'config.json'),
        model=str(args.model), fit_windows=FIT, diagnostic_windows=DIAGNOSTIC,
        sequence_length=LENGTH, seed=20260914, excerpt_length=4096,
        dataset_revision=C4.parent.name,
        packing='16 document-disjoint 4096-token excerpts; no separators',
        source_sha256=sha256(Path(__file__)))
    if (directory/'manifest.json').exists():
        saved = read_json(directory/'manifest.json')
        assert saved['protocol'] == protocol
        assert sha256(directory/'windows.safetensors') == saved['sha256']
        print('windows verified', flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rng = random.Random(protocol['seed'])
    sources = sorted(C4.glob('c4-train.*.json.gz'))
    assert sources
    rng.shuffle(sources)
    pieces, records, seen, used_sources = [], [], set(), {}
    needed = (FIT+DIAGNOSTIC)*16
    for source in sources:
        used_sources[source.name] = sha256(source)
        with gzip.open(source, 'rt') as stream:
            for line_index, line in enumerate(stream):
                text = json.loads(line)['text']
                if len(text) < 10000:
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in seen:
                    continue
                ids = tokenizer(text, add_special_tokens=False)['input_ids']
                if len(ids) < 4096:
                    continue
                start = rng.randint(0, len(ids)-4096)
                piece = torch.tensor(ids[start:start+4096], dtype=torch.int32)
                pieces.append(piece)
                seen.add(digest)
                records.append(dict(window=(len(pieces)-1)//16, source=source.name,
                    line=line_index, document_sha256=digest, token_start=start,
                    token_sha256=tensor_hash(piece)))
                if len(pieces)%16 == 0:
                    print(f'windows={len(pieces)//16}/{FIT+DIAGNOSTIC}', flush=True)
                if len(pieces) == needed:
                    break
        if len(pieces) == needed:
            break
    assert len(pieces) == len(seen) == needed
    windows = torch.stack(pieces).reshape(FIT+DIAGNOSTIC, LENGTH)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'windows.safetensors'
    assert not path.exists()
    save_file({'input_ids': windows}, str(path))
    write_json(directory/'manifest.json', dict(status='complete', protocol=protocol,
        sha256=sha256(path), fit_ids=list(range(FIT)),
        validation_ids=list(range(FIT, FIT+DIAGNOSTIC)), records=records,
        sources_sha256=used_sources, command=shlex.join(sys.argv)))


@torch.inference_mode()
def capture(args):
    window_path = args.run/'windows/windows.safetensors'
    window_record = read_json(window_path.parent/'manifest.json')
    assert window_record['status'] == 'complete'
    assert sha256(window_path) == window_record['sha256']
    assert window_record['protocol']['model_config_sha256'] == sha256(args.model/'config.json')
    windows = load_file(str(window_path))['input_ids']
    assert windows.shape == (FIT+DIAGNOSTIC, LENGTH)
    documents = list(range(args.shard, FIT+DIAGNOSTIC, 4))
    if args.smoke:
        documents = [args.shard]
    directory = args.run/('smoke' if args.smoke else 'capture')/f'shard_{args.shard}'
    protocol = dict(model_config_sha256=sha256(args.model/'config.json'),
        windows_sha256=sha256(window_path), documents=documents, shard=args.shard,
        num_shards=4, sequence_length=LENGTH, dtype='bfloat16',
        covariance_dtype='float32', covariance_accumulation='float64 CPU across windows',
        attention='PyTorch SDPA flash only; full causal support', mlp_chunk=1024,
        source_sha256=sha256(Path(__file__)))
    if (directory/'manifest.json').exists():
        saved = read_json(directory/'manifest.json')
        assert saved['status'] == 'complete' and saved['protocol'] == protocol
        for item in saved['artifacts'].values():
            assert sha256(directory/item['file']) == item['sha256']
        print('capture shard verified', args.shard, flush=True)
        return
    assert torch.cuda.get_device_name(0) == 'NVIDIA RTX A5000'
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa').eval().cuda()
    assert model.config.model_type == 'llama' and len(model.layers) == 32
    install_chunked_mlp(model)
    sums = {split: torch.zeros(32, 4096, 4096, dtype=torch.float64)
            for split in ('fit', 'heldout')}
    active = {}
    def hook(layer):
        def collect(module, inputs):
            x = inputs[0].reshape(-1, 4096)
            assert len(x) == LENGTH
            gram = torch.zeros(4096, 4096, device=x.device, dtype=torch.float32)
            for block in x.split(4096):
                fp = block.float()
                gram.addmm_(fp.T, fp)
            sums[active['split']][layer].add_(gram.cpu().double())
            active['seen'].add(layer)
        return collect
    handles = [layer.self_attn.o_proj.register_forward_pre_hook(hook(i))
               for i, layer in enumerate(model.layers)]
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for document in documents:
        active.update(split='fit' if document < FIT else 'heldout', seen=set())
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = model(input_ids=windows[document:document+1].long().cuda(),
                        use_cache=False, return_dict=True)
        assert torch.isfinite(out.last_hidden_state).all()
        assert active['seen'] == set(range(32))
        del out
        print(f'shard={args.shard} document={document} seconds={time.monotonic()-started:.1f} '
              f'peak_GiB={torch.cuda.max_memory_allocated()/2**30:.3f}', flush=True)
    for handle in handles:
        handle.remove()
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for layer in range(32):
        path = directory/f'layer_{layer:03d}.safetensors'
        assert not path.exists()
        save_file(dict(fit_sum=sums['fit'][layer], heldout_sum=sums['heldout'][layer],
            weight=model.layers[layer].self_attn.o_proj.weight.cpu().contiguous()), str(path))
        artifacts[str(layer)] = dict(file=path.name, sha256=sha256(path))
    write_json(directory/'manifest.json', dict(status='complete', protocol=protocol,
        artifacts=artifacts, elapsed_seconds=time.monotonic()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), python=sys.executable,
        command=shlex.join(sys.argv)))


def merge(args):
    directory = args.run/'covariance'
    if (directory/'manifest.json').exists():
        existing = read_json(directory/'manifest.json')
        for record in existing['artifacts'].values():
            assert sha256(directory/record['file']) == record['sha256']
        print('covariance already verified', flush=True)
        return
    records = [read_json(args.run/f'capture/shard_{s}/manifest.json') for s in range(4)]
    for s, record in enumerate(records):
        assert record['status'] == 'complete'
        assert record['protocol']['documents'] == list(range(s, FIT+DIAGNOSTIC, 4))
        assert record['protocol']['model_config_sha256'] == sha256(args.model/'config.json')
        assert record['protocol']['windows_sha256'] == sha256(args.run/'windows/windows.safetensors')
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for layer in range(32):
        sums = {}
        weight = None
        for s, record in enumerate(records):
            item = record['artifacts'][str(layer)]
            path = args.run/f'capture/shard_{s}'/item['file']
            assert sha256(path) == item['sha256']
            data = load_file(str(path))
            if weight is None:
                weight = data['weight']
            assert torch.equal(weight, data['weight'])
            for name in ('fit_sum', 'heldout_sum'):
                if name not in sums:
                    sums[name] = data[name]
                else:
                    sums[name].add_(data[name])
        path = directory/f'layer_{layer:03d}.safetensors'
        assert not path.exists()
        fit = sums['fit_sum'].div_(FIT*LENGTH).float()
        heldout = sums['heldout_sum'].div_(DIAGNOSTIC*LENGTH).float()
        assert torch.isfinite(fit).all() and torch.isfinite(heldout).all()
        save_file(dict(fit_covariance=fit, heldout_covariance=heldout, weight=weight), str(path))
        artifacts[str(layer)] = dict(file=path.name, sha256=sha256(path),
            fit_covariance_shape=[4096,4096], heldout_covariance_shape=[4096,4096],
            weight_shape=[4096,4096], covariance_dtype='torch.float32', weight_dtype='torch.bfloat16')
    write_json(directory/'manifest.json', dict(format='basisserve.attention_o_proj_covariances.v1',
        model=dict(path=str(args.model), config_sha256=sha256(args.model/'config.json'),
            model_type='llama', attention_type='gqa', num_hidden_layers=32,
            num_attention_heads=32, num_key_value_heads=8, head_dim=128, hidden_size=4096),
        layers=list(range(32)), calibration=dict(window_count=FIT+DIAGNOSTIC,
            fit_windows=FIT, heldout_windows=DIAGNOSTIC, sequence_length=LENGTH,
            positions_per_window=LENGTH, fit_rows=FIT*LENGTH, heldout_rows=DIAGNOSTIC*LENGTH,
            rows_per_layer=(FIT+DIAGNOSTIC)*LENGTH,
            windows_file=str(args.run/'windows/windows.safetensors'),
            windows_sha256=sha256(args.run/'windows/windows.safetensors')),
        artifacts=artifacts, source_manifests=records, command=shlex.join(sys.argv)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'capture', 'merge'))
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert 0 <= args.shard < 4
    configure()
    {'prepare': prepare, 'capture': capture, 'merge': merge}[args.stage](args)


if __name__ == '__main__':
    main()
