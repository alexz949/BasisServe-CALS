"""Extract Q64 fit queries from immutable BF16 candidate captures, no model forward."""
import json
from pathlib import Path
import sys
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.select_query_positions import load_position_manifest
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256,write_json


def main():
    torch.set_num_threads(2)
    position_path=ROOT/'results/evaluation/qgram64/positions.json'
    selected=load_position_manifest(position_path)
    source=Path(selected['candidate_capture_root'])
    assert sha256(source/'manifest.json')==selected['candidate_manifest_sha256']
    capture=json.loads((source/'manifest.json').read_text())
    assert capture['status']=='complete' and selected['fit_document_ids']==list(range(64))
    spec=dict(capture['protocol'])
    spec.update(format='basisserve.selected_queries.v1',position_manifest_sha256=sha256(position_path),
        positions_by_layer={l:r['selected_positions'] for l,r in selected['layers'].items()},
        extraction_source_sha256=sha256(Path(__file__)),candidate_manifest_sha256=selected['candidate_manifest_sha256'])
    target=ROOT/'results/calibration/qgram64_fit';target.mkdir(parents=True,exist_ok=True)
    if (target/'manifest.json').exists():
        existing=json.loads((target/'manifest.json').read_text())
        assert existing['status']=='complete' and existing['protocol']==spec
        for item in existing['artifacts'].values(): assert sha256(target/item['file'])==item['sha256']
        return
    artifacts={}
    for doc in range(64):
        item=capture['artifacts'][str(doc)];path=source/item['file']
        assert sha256(path)==item['sha256']
        q=load_file(str(path))['queries'];parts=[]
        for i,layer in enumerate(spec['layers']):
            grid=capture['protocol']['positions_by_layer'][str(layer)]
            slots=[grid.index(p) for p in spec['positions_by_layer'][str(layer)]]
            parts.append(q[i,slots])
        out=torch.stack(parts).contiguous()
        assert out.shape==(36,64,32,128) and out.dtype==torch.bfloat16 and torch.isfinite(out).all()
        path=target/f'window_{doc:03d}.safetensors'
        if path.exists(): assert torch.equal(load_file(str(path))['queries'],out)
        else: save_file({'queries':out},str(path))
        artifacts[str(doc)]=dict(file=path.name,sha256=sha256(path),reused_candidates=True)
        print(f'fit window={doc} Q64 extracted',flush=True)
    write_json(target/'manifest.json',dict(status='complete',protocol=spec,artifacts=artifacts))


if __name__=='__main__': main()
