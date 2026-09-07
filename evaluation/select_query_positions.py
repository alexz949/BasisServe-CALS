"""Select immutable per-layer Q positions using only fit-window candidate Q."""

import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from basisserve.core.query_position_sampling import (
    candidate_positions, canonical_hash, select_stratified_query_positions, uniform_query_positions,
)
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json


def load_position_manifest(path):
    record = json.loads(Path(path).read_text())
    assert record['format'] == 'basisserve.query_position_manifest.v1'
    assert record['fit_document_ids'] == list(range(record['num_fit_windows']))
    assert 0 < record['num_fit_windows'] <= 64
    grid = candidate_positions(record['context_length'],record['candidate_stride'])
    count = record['num_bins']*record['queries_per_bin']
    for layer,data in record['layers'].items():
        assert 0 <= int(layer) < 36
        positions = data['selected_positions']
        assert positions == sorted(set(positions)) and len(positions) == count
        assert all(p in grid and p >= 32 for p in positions)
        if record['method'] in ('terminal_uniform_q8', 'terminal_qgram_q8'):
            assert count == 8 and all(p >= 3*record['context_length']//4 for p in positions)
        elif record['method'] != 'terminal_uniform_q32':
            assert all(sum(p*record['num_bins']//record['context_length'] == b for p in positions) == record['queries_per_bin'] for b in range(record['num_bins']))
    return record


def manifest_queries(position_path, capture_root, split, layer):
    """Load sorted selected queries from candidate-fit or fully selected captures."""
    selected = load_position_manifest(position_path)
    capture_root = Path(capture_root)
    capture = json.loads((capture_root/'manifest.json').read_text())
    assert capture['status'] == 'complete'
    spec = capture['protocol']
    assert spec['model_config_sha256'] == selected['model_config_sha256']
    assert spec['fit_token_sha256'] == selected['fit_token_sha256']
    assert spec['context_length'] == selected['context_length']
    assert split in ('fit','validation')
    docs = selected['fit_document_ids'] if split == 'fit' else list(range(64,80))
    if spec['format'] == 'basisserve.candidates_queries.v1':
        assert split == 'fit' and sha256(capture_root/'manifest.json') == selected['candidate_manifest_sha256']
    else:
        assert spec['format'] == 'basisserve.selected_queries.v1'
        assert spec['position_manifest_sha256'] == sha256(position_path)
    positions = selected['layers'][str(layer)]['selected_positions']
    captured_positions = spec['positions_by_layer'][str(layer)]
    slots = [captured_positions.index(p) for p in positions]
    queries = []
    for doc in docs:
        artifact = capture['artifacts'][str(doc)]
        file = capture_root/artifact['file']
        assert sha256(file) == artifact['sha256']
        tensor = load_file(str(file))['queries']
        assert tensor.dtype == torch.bfloat16 and torch.isfinite(tensor).all()
        queries.append(tensor[spec['layers'].index(layer),slots])
    return torch.stack(queries),torch.tensor(positions,dtype=torch.long)


def load_fit_candidates(root, layer):
    root=Path(root)
    capture=json.loads((root/'manifest.json').read_text())
    assert capture['status']=='complete'
    spec=capture['protocol']
    assert spec['format']=='basisserve.candidates_queries.v1'
    docs=spec['fit_document_ids']
    assert docs==list(range(len(docs))) and 0 < len(docs) <= 64
    assert spec['documents']==docs and set(capture['artifacts'])==set(map(str,docs))
    observations=[]
    for doc in docs:
        item=capture['artifacts'][str(doc)]
        file=root/item['file']
        assert sha256(file)==item['sha256']
        tensor=load_file(str(file))['queries']
        observations.append(tensor[spec['layers'].index(layer)])
    return torch.stack(observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-capture',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--num-bins',type=int,default=4)
    parser.add_argument('--queries-per-bin',type=int,default=8)
    parser.add_argument('--whitening-eps',type=float,default=1e-6)
    parser.add_argument('--policy',choices=('stratified_query_gram_pivot','terminal_uniform_q32','full_uniform_q32'),default='stratified_query_gram_pivot')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    capture=json.loads((args.candidate_capture/'manifest.json').read_text())
    assert capture['status']=='complete'
    spec=capture['protocol']
    assert spec['format']=='basisserve.candidates_queries.v1'
    docs=spec['fit_document_ids']
    assert docs==list(range(len(docs))) and 0 < len(docs) <= 64
    assert spec['documents']==docs and set(capture['artifacts'])==set(map(str,docs))
    assert not (args.output_dir/'positions.json').exists()
    layers, tensors = {},{}
    for layer in spec['layers']:
        q=load_fit_candidates(args.candidate_capture,layer).to(args.device)
        positions=spec['positions_by_layer'][str(layer)]
        assert positions==candidate_positions(spec['context_length'],spec['candidate_stride'])
        if args.policy=='stratified_query_gram_pivot':
            options=dict(context_length=spec['context_length'],num_bins=args.num_bins,
                         queries_per_bin=args.queries_per_bin,whitening_eps=args.whitening_eps)
            result,whitening,grams=select_stratified_query_positions(q,positions,**options)
            repeated,_,_=select_stratified_query_positions(q,positions,**options)
            assert canonical_hash(result)==canonical_hash(repeated)
            tensors[f'layer{layer:03d}.whitening']=whitening.cpu()
            for b,gram in enumerate(grams):
                tensors[f'layer{layer:03d}.gram{b}']=gram.cpu()
        else:
            selected=uniform_query_positions(spec['context_length'],args.num_bins*args.queries_per_bin,
                              terminal_fraction=.25 if args.policy=='terminal_uniform_q32' else 1.)
            assert all(p in positions for p in selected)
            result={'selected_positions':selected,'bins':[]}
        result['prefix_at_most_2048_count']=sum(p+1<=2048 for p in result['selected_positions'])
        layers[str(layer)]=result
        print('LAYER',layer,json.dumps(result),flush=True)
    manifest={'format':'basisserve.query_position_manifest.v1','method':args.policy,
        'context_length':spec['context_length'],'candidate_stride':spec['candidate_stride'],
        'num_bins':args.num_bins,'queries_per_bin':args.queries_per_bin,
        'num_fit_windows':len(docs),'fit_document_ids':docs,'num_query_heads':q.shape[2],'head_dim':q.shape[3],
        'whitening':'per_head_uncentered_second_moment' if tensors else None,'whitening_eps':args.whitening_eps,
        'layers':layers,'candidate_capture_root':str(args.candidate_capture.resolve()),
        'candidate_manifest_sha256':sha256(args.candidate_capture/'manifest.json'),
        'source_query_hash':canonical_hash(capture['artifacts']),
        'fit_token_sha256':spec['fit_token_sha256'],'model_config_sha256':spec['model_config_sha256'],
        'code_version':{p:sha256(ROOT/p) for p in ('basisserve/core/query_position_sampling.py','evaluation/select_query_positions.py')},
        'scope':'smoke; not the 64-window sampler' if len(docs)<64 else '64 fit windows only; no diagnostic selection',
        'determinism_scope':'identical artifact, code and numerical environment; no timestamp in manifest'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if tensors:
        save_file(tensors,str(args.output_dir/'statistics.safetensors'))
    write_json(args.output_dir/'positions.json',manifest)
    assert load_position_manifest(args.output_dir/'positions.json')==manifest
    print('manifest canonical hash',canonical_hash(manifest),flush=True)


if __name__=='__main__':
    with torch.inference_mode():
        main()
