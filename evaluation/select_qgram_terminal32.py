"""Select terminal Q32 from the original fit-only, full-window-whitened Gram."""
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from basisserve.core.query_position_sampling import candidate_positions,select_positions_pivoted_gram
from evaluation.select_query_positions import load_position_manifest
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json

def main():
    torch.set_num_threads(2)
    source=ROOT/'results/evaluation/qgram32'
    old=load_position_manifest(source/'positions.json')
    grams=load_file(str(source/'statistics.safetensors'))
    new=json.loads(json.dumps(old))
    new.update(num_bins=1,queries_per_bin=32,method='terminal_qgram_q32')
    grid=[p for p in candidate_positions(old['context_length'],old['candidate_stride']) if p>=24576]
    assert old['context_length']==32768 and len(grid)>=32
    for layer,record in new['layers'].items():
        pivots,diag,residual=select_positions_pivoted_gram(grams[f'layer{int(layer):03d}.gram3'],grid,32)
        assert pivots[:8]==old['layers'][layer]['bins'][3]['pivot_order']
        again,_,_=select_positions_pivoted_gram(grams[f'layer{int(layer):03d}.gram3'],grid,32)
        assert pivots==again and len(set(pivots))==32 and min(pivots)>=24576
        record['bins']=[dict(source_bin=3,pivot_order=pivots,selected_positions=sorted(pivots),pivots=diag,**residual)]
        record['selected_positions']=sorted(pivots)
        record['prefix_at_most_2048_count']=0
    new['extension']=dict(source_manifest_sha256=sha256(source/'positions.json'),
        source_statistics_sha256=sha256(source/'statistics.safetensors'),
        code_sha256=sha256(Path(__file__)),method='terminal32 pivots of unchanged gram3; original full-window fit-only whitening')
    target=ROOT/'results/evaluation/qgram_terminal32/positions.json'
    if target.exists(): assert load_position_manifest(target)==new
    else: write_json(target,new)
    assert load_position_manifest(target)==new
    print('36 layers: terminal32 range, uniqueness, old terminal8 prefix and determinism verified',flush=True)

if __name__=='__main__': main()
