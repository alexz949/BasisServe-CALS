"""Fit uniform gated V (rank 192 by default) per full-attention layer from the 128K captures of ``capture_qwen35_128k_v.py``.

Same ALS as ``fit_qwen35_k_routing_v.py`` (decoder-closed sweeps with CG-solved encoder/decoder blocks); the fit set is every
capture window whose manifest says split == 'fit', the optional diagnostic set every 'diagnostic' window (report only)."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

from basisserve.core.qwen35_gated_v_als import GatedVBlock, gated_v_update, initialize_gated_v
from evaluation.qwen35_hybrid_common import atomic_save, sha256
from evaluation.qwen35_v_capture_reader import WindowCapture
from evaluation.fit_qwen35_k_routing_v import fit_one


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--ranks', default='192')
    p.add_argument('--sweeps', type=int, default=12)
    p.add_argument('--encoder-cg', type=int, default=16)
    p.add_argument('--decoder-cg', type=int, default=50)
    p.add_argument('--chunk-rows', type=int, default=2048)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert args.layer in (3, 7, 11, 15, 19, 23, 27, 31)
    fit_paths, diag_paths, digests, windows_hash, stride, length = [], [], {}, None, None, None
    for root in sorted(args.capture.glob('w*')):
        manifests = [m for m in root.glob('manifest_l*.json') if args.layer in json.loads(m.read_text())['layers']]
        if not manifests:
            continue
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text())
        assert manifest['status'] == 'complete' and manifest['wo_compression'] is False
        windows_hash = windows_hash or manifest['windows_sha256']; stride = stride or manifest['row_stride']; length = length or manifest['length']
        assert manifest['windows_sha256'] == windows_hash and manifest['row_stride'] == stride and manifest['length'] == length
        path = root / f'l{args.layer:02d}.safetensors'
        digest = sha256(path)
        assert digest == manifest['files'][path.name]
        (fit_paths if manifest['split'] == 'fit' else diag_paths).append(path)
        digests[str(path)] = digest
    assert fit_paths, 'no fit captures for this layer'
    mapping = torch.arange(16) // 4
    train = WindowCapture(fit_paths, mapping)
    train.validate(4)
    diagnostic = WindowCapture(diag_paths, mapping) if diag_paths else None
    if diagnostic is not None:
        diagnostic.validate(4)
    assert len(train.z) == len(fit_paths) * length
    train.preload_inputs('cuda:0', args.chunk_rows)
    print(json.dumps(dict(layer=args.layer, fit_windows=len(fit_paths), diagnostic_windows=len(diag_paths), rows=len(train.z), row_stride=stride,
        input_gib=(train.z.nbytes + train.gate.nbytes) / 2**30, allocated_gib=torch.cuda.memory_allocated() / 2**30)), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for rank in map(int, args.ranks.split(',')):
            assert 0 < rank < 256
            destination = args.output / f'l{args.layer:02d}_r{rank:03d}.pt'
            if destination.exists():
                saved = torch.load(destination, map_location='cpu', weights_only=True)
                assert saved['capture_hashes'] == digests and saved['encoder_sweeps'] == args.sweeps
                assert saved['encoder_cg'] == args.encoder_cg and saved['decoder_cg'] == args.decoder_cg
                continue
            started = time.monotonic()
            def progress(row):
                print(json.dumps(dict(layer=args.layer, rank=rank, **row)), flush=True)
            encoder, decoder, history = fit_one(train, rank, sweeps=args.sweeps, encoder_cg=args.encoder_cg,
                                                decoder_cg=args.decoder_cg, chunk_rows=args.chunk_rows, progress=progress)
            e, r = encoder.bfloat16(), decoder.bfloat16()
            metrics = {}
            for split, data in [('fit', train)] + ([('diagnostic', diagnostic)] if diagnostic is not None else []):
                operator = GatedVBlock(data, e.float(), block='decoder', chunk_rows=args.chunk_rows)
                metrics[split + '_export_relative_mse'] = 2 * len(data.z) * operator.loss(r.float()) / data.target_energy
            atomic_save(destination, dict(status='complete', layer=args.layer, rank=rank, E_V=e.cpu(), R_V=r.cpu(),
                encoder_sweeps=args.sweeps, encoder_cg=args.encoder_cg, decoder_cg=args.decoder_cg, chunk_rows=args.chunk_rows,
                history=history, metrics=metrics, input_storage='GPU resident z/gate; streamed target',
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                selection='fixed final decoder-closed sweep; diagnostic never selects factors', windows_sha256=windows_hash,
                fit_windows=len(fit_paths), diagnostic_windows=len(diag_paths), row_stride=stride, rows_per_window=length,
                capture_hashes=digests, wo_compression=False, command=sys.argv, python=sys.executable, seconds=time.monotonic() - started))
            progress(dict(complete=True, **metrics))


if __name__ == '__main__':
    main()
