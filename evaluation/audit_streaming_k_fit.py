"""Verify complete 64K routing factors and their Base/V/data provenance."""
import argparse
from pathlib import Path
import torch
from evaluation.v96kl_common import configure, read_json, write_json, sha256
from evaluation.fit_k_routing_streaming import verified, encoder_for


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--base-rank', type=int, default=16)
    p.add_argument('--residual-rank', type=int, default=16)
    args = p.parse_args()
    configure()
    root, identity = args.root, read_json(args.identity)
    assert identity['status'] == 'complete' and identity['attention_layers'] == list(range(32))
    data_root=args.data_root or root
    b,r=args.base_rank,args.residual_rank
    wm = read_json(data_root/'calibration/manifest.json')
    assert wm['sha256'] == sha256(data_root/'calibration/windows.safetensors')
    hashes, metrics = {}, {}
    for layer in range(32):
        path = root/f'ours_b{b}r{r}'/f'layer_{layer:03d}.safetensors'
        tensors, meta = verified(path)
        protocol = meta['protocol']
        assert protocol['format'] == 'basisserve.k_router.streaming.v1'
        assert protocol['base_rank']==b and protocol['residual_rank']==r
        assert protocol['sequence_length'] == 65536 and not protocol['smoke']
        assert protocol['fit_ids'] == list(range(64)) and protocol['diagnostic_ids'] == list(range(64,80))
        assert protocol['fit_queries'] == 64 and protocol['diagnostic_queries'] == 32
        assert protocol['windows_sha256'] == wm['sha256']
        assert protocol['windows_manifest_sha256'] == sha256(data_root/'calibration/manifest.json')
        assert meta['identity_sha256'] == sha256(args.identity)
        assert meta['sweeps'] == 40 and meta['pcg_iterations'] == 100
        assert len(meta['losses'][f'b{b}_r{r}']['sweeps']) == 40
        base_path = root/'base'/path.name
        base, bm = verified(base_path)
        assert meta['base_sha256'] == sha256(base_path)
        assert bm['protocol'] == protocol and bm['identity_sha256'] == meta['identity_sha256']
        assert torch.equal(base['encoder'], encoder_for(identity, layer))
        assert meta['v_rank'] == identity['layer_ranks'][layer]
        for name in ('left','right','bias'):
            assert torch.equal(tensors[f'base_{name}_b{b}'],base[name].float())
        assert tensors[f'residual_encoder_b{b}_r{r}'].shape == (8,128,r)
        assert tensors[f'residual_query_b{b}_r{r}'].shape == (32,128,r)
        assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in tensors.values())
        for name, digest in protocol['source_sha256'].items():
            assert sha256(Path(name)) == digest
        hashes[str(layer)] = meta['sha256']
        metrics[str(layer)] = meta['losses'][f'b{b}_r{r}']
    write_json(root/'manifests/fit_audit.json',dict(status='complete', layers=32,
        identity_sha256=sha256(args.identity), bank_sha256=hashes, metrics=metrics,
        scope='factor tensors, Base/V binding, frozen data and fitting protocol; Fisher payloads verified by fitter'))
    print(f'Verified all 32 streaming Base{b}/Residual{r} factors',flush=True)


if __name__ == '__main__':
    main()
