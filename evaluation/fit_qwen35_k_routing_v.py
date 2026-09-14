"""Fit gated V on all 64 native 32K C4 windows with bounded row reads."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

from basisserve.core.qwen35_gated_v_als import GatedVBlock, gated_v_update, initialize_gated_v
from evaluation.qwen35_hybrid_common import atomic_save, sha256
from evaluation.qwen35_v_capture_reader import WindowCapture


def fit_one(train, rank, *, sweeps, encoder_cg, decoder_cg, chunk_rows, progress):
    encoder, decoder = initialize_gated_v(train, rank, device='cuda:0', chunk_rows=chunk_rows)
    history = []
    for sweep in range(sweeps+1):
        operator = GatedVBlock(train, encoder, block='decoder', chunk_rows=chunk_rows)
        decoder, record = gated_v_update(operator, decoder, linear_max_iter=decoder_cg,
            encoder_preconditioner='separable', progress=progress)
        history.append(dict(sweep=sweep, **record))
        progress(history[-1])
        if sweep == sweeps:
            break
        operator = GatedVBlock(train, decoder, block='encoder', chunk_rows=chunk_rows)
        encoder, record = gated_v_update(operator, encoder, linear_max_iter=encoder_cg,
            encoder_preconditioner='separable', progress=progress)
        history.append(dict(sweep=sweep, **record))
        progress(history[-1])
        q, t = torch.linalg.qr(encoder, mode='reduced')
        encoder, decoder = q, t @ decoder
    return encoder, decoder, history


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--ranks', default='32,48,64,80,96,112,128,160,192,224')
    p.add_argument('--sweeps', type=int, default=12)
    p.add_argument('--encoder-cg', type=int, default=16)
    p.add_argument('--decoder-cg', type=int, default=200)
    p.add_argument('--chunk-rows', type=int, default=2048)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert args.layer in (3, 7, 11, 15, 19, 23, 27, 31)
    paths, digests, windows_hash = [], {}, None
    for index in range(80):
        root = args.capture / f'w{index:03d}'
        manifest = json.loads((root/'manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['index'] == index
        assert manifest['length'] == 32768 and manifest['wo_compression'] is False
        assert manifest['split'] == ('fit' if index < 64 else 'diagnostic')
        windows_hash = windows_hash or manifest['windows_sha256']
        assert manifest['windows_sha256'] == windows_hash
        path = root/f'l{args.layer:02d}.safetensors'
        digest = sha256(path)
        assert digest == manifest['files'][path.name]
        paths.append(path)
        digests[str(path)] = digest
    mapping = torch.arange(16)//4
    train, diagnostic = WindowCapture(paths[:64], mapping), WindowCapture(paths[64:], mapping)
    train.validate(4)
    diagnostic.validate(4)
    assert len(train.z) == 64*32768 and len(diagnostic.z) == 16*32768
    train.preload_inputs('cuda:0', args.chunk_rows)
    print(json.dumps(dict(inputs_resident=True,
        input_gib=(train.z.nbytes+train.gate.nbytes)/2**30,
        allocated_gib=torch.cuda.memory_allocated()/2**30)), flush=True)
    with torch.inference_mode():
        for rank in map(int, args.ranks.split(',')):
            assert 0 < rank < 256
            destination = args.output/f'l{args.layer:02d}_r{rank:03d}.pt'
            if destination.exists():
                saved = torch.load(destination, map_location='cpu', weights_only=True)
                assert saved['capture_hashes'] == digests and saved['encoder_sweeps'] == args.sweeps
                assert saved['encoder_cg'] == args.encoder_cg and saved['decoder_cg'] == args.decoder_cg
                continue
            started = time.monotonic()
            def progress(row):
                print(json.dumps(dict(layer=args.layer, rank=rank, **row)), flush=True)
            encoder, decoder, history = fit_one(train, rank, sweeps=args.sweeps,
                encoder_cg=args.encoder_cg, decoder_cg=args.decoder_cg,
                chunk_rows=args.chunk_rows, progress=progress)
            e, r = encoder.bfloat16(), decoder.bfloat16()
            metrics = {}
            for split, data in [('fit', train), ('diagnostic', diagnostic)]:
                operator = GatedVBlock(data, e.float(), block='decoder', chunk_rows=args.chunk_rows)
                metrics[split+'_export_relative_mse'] = 2*len(data.z)*operator.loss(r.float())/data.target_energy
            atomic_save(destination, dict(status='complete', layer=args.layer, rank=rank,
                E_V=e.cpu(), R_V=r.cpu(), encoder_sweeps=args.sweeps, encoder_cg=args.encoder_cg,
                decoder_cg=args.decoder_cg, chunk_rows=args.chunk_rows, history=history, metrics=metrics,
                input_storage='GPU resident z/gate; streamed target',
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                selection='fixed final decoder-closed sweep; diagnostic never selects factors',
                windows_sha256=windows_hash, capture_hashes=digests, wo_compression=False,
                command=sys.argv, python=sys.executable, seconds=time.monotonic()-started))
            progress(dict(complete=True, **metrics))


if __name__ == '__main__':
    main()
