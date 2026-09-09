"""Extend the frozen Q32 Gram pivot sequence to Q64 without new whitening."""
import json
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from basisserve.core.query_position_sampling import candidate_positions, select_positions_pivoted_gram
from evaluation.select_query_positions import load_position_manifest
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json


def main():
    torch.set_num_threads(2)
    source=ROOT/'results/evaluation/qgram32'
    old=load_position_manifest(source/'positions.json')
    assert old['queries_per_bin']==8 and old['num_bins']==4 and old['num_fit_windows']==64
    grams=load_file(str(source/'statistics.safetensors'))
    new=json.loads(json.dumps(old));new['queries_per_bin']=16
    grid=candidate_positions(old['context_length'],old['candidate_stride'])
    for layer,record in new['layers'].items():
        selected=[]
        for b in range(4):
            positions=[p for p in grid if p*4//old['context_length']==b]
            gram=grams[f'layer{int(layer):03d}.gram{b}']
            pivots,diag,residual=select_positions_pivoted_gram(gram,positions,16)
            assert pivots[:8]==old['layers'][layer]['bins'][b]['pivot_order']
            repeated,_,_=select_positions_pivoted_gram(gram,positions,16)
            assert pivots==repeated
            record['bins'][b].update(pivot_order=pivots,selected_positions=sorted(pivots),pivots=diag,**residual)
            selected.extend(pivots)
        record['selected_positions']=sorted(selected)
        record['prefix_at_most_2048_count']=sum(p+1<=2048 for p in selected)
        assert set(old['layers'][layer]['selected_positions'])<=set(selected) and len(set(selected))==64
    new['extension']={'source_manifest_sha256':sha256(source/'positions.json'),
        'source_statistics_sha256':sha256(source/'statistics.safetensors'),
        'code_sha256':sha256(Path(__file__)),
        'method':'extend unchanged fit-only position Gram pivots from8 to16 per bin; no new whitening'}
    target=ROOT/'results/evaluation/qgram64/positions.json'
    if target.exists(): assert load_position_manifest(target)==new
    else: write_json(target,new)
    assert load_position_manifest(target)==new
    print('Q64:36 layers; old Q32 subset and repeated pivots verified',flush=True)


if __name__=='__main__': main()
