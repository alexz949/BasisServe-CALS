"""Convert the released Qwen3.5-9B 128K B16R16 Page-Fisher router (HF ``routers/b16r16/l{layer:02d}.pt``) into the
safetensors + JSON record layout that ``eval_qwen35_128k_ruler.py`` reads, keeping every protocol field of the release."""
import argparse
import json
from pathlib import Path

import torch

from evaluation.fit_k_routing_streaming import layer_file, save_record
from evaluation.v96kl_common import sha256


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True, help='directory with l03.pt ... l31.pt and manifest.json')
    p.add_argument('--v-bank', type=Path, required=True)
    p.add_argument('--gdn-bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    release = json.loads((args.source/'manifest.json').read_text())
    assert release['checkpoint']['v_bank_sha256'] == sha256(args.v_bank)
    assert release['checkpoint']['gdn_wo_bank_sha256'] == sha256(args.gdn_bank)
    for layer in (3, 7, 11, 15, 19, 23, 27, 31):
        path = args.source/f'l{layer:02d}.pt'
        assert sha256(path) == release['router_artifacts'][path.name]['sha256']
        payload = torch.load(path, map_location='cpu', weights_only=True)
        assert payload['status'] == 'complete' and payload['layer'] == layer and payload['v_rank'] == 192
        source = payload['protocol']
        assert source['base_rank'] == 16 and source['residual_rank'] == 16 and source['page_size'] == 32
        assert source['excluded_prefix_pages'] == 0 and source['excluded_recent_tokens'] == 64 and source['routing_budget'] == 2048
        assert source['v_bank_sha256'] == sha256(args.v_bank) and source['gdn_wo_bank_sha256'] == sha256(args.gdn_bank)
        tensors = {name: value.float().contiguous() for name, value in payload['tensors'].items()}
        assert set(tensors) == {'base_left_b16', 'base_right_b16', 'base_bias_b16', 'residual_encoder_b16_r16', 'residual_query_b16_r16'}
        protocol = dict(format='basisserve.qwen35.k_router.release.b16r16.page_fisher.v1', source_format=source['format'],
            objective='page_fisher', objective_definition=source['residual_objective'], base_objective=source['base_objective'],
            v_bank_sha256=source['v_bank_sha256'], gdn_bank_sha256=source['gdn_wo_bank_sha256'],
            model_config_sha256=source['model_config_sha256'], windows_sha256=source['windows_sha256'],
            sequence_length=int(source['sequence_length']), fit_ids=list(source['fit_ids']), diagnostic_ids=list(source['diagnostic_ids']),
            fit_queries=int(source['fit_queries']), diagnostic_queries=int(source['diagnostic_queries']),
            base_rank=16, residual_rank=16, page_size=32, excluded_prefix_pages=0, excluded_recent_tokens=64,
            deployment_page_budget_tokens=int(source['routing_budget']), sink_tokens_within_page_budget=0,
            recent_tokens_within_page_budget=64, sweeps=int(source['bcd_sweeps']), pcg_iterations=int(source['pcg_iterations']),
            smoke=False, release=dict(repo_id=release['checkpoint']['repo_id'], revision=release['checkpoint']['revision'],
                                      file=path.name, sha256=sha256(path)))
        save_record(layer_file(args.output, 'ours_b16r16', layer), tensors,
            dict(protocol=protocol, layer=layer, v_rank=192, losses=payload.get('losses'), metrics=payload.get('metrics')))
        print(dict(layer=layer, tensors={k: tuple(v.shape) for k, v in tensors.items()}), flush=True)


if __name__ == '__main__':
    main()
