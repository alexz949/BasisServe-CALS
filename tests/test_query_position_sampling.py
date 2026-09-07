import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import save_file

from basisserve.core.query_position_sampling import (
    candidate_positions, canonical_hash, select_positions_pivoted_gram,
    select_stratified_query_positions, uniform_query_positions,
)
from evaluation.select_query_positions import load_fit_candidates, manifest_queries
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256


def test_terminal_q8_derivation_keeps_parent_pivots():
    from evaluation.prepare_terminal_q8 import terminal_positions
    from evaluation.select_query_positions import load_position_manifest
    parent = {'context_length':32768, 'method':'stratified_query_gram_pivot',
              'layers':{'15':{'selected_positions':list(range(1023,32768,1024))}}}
    fixed = terminal_positions(parent,'uniform')
    selected = terminal_positions(parent,'qgram')
    assert fixed['15'] == selected['15'] == list(range(25599,32768,1024))
    parent['layers']['15']['selected_positions'][-8] = 24639
    selected = terminal_positions(parent,'qgram')
    assert selected['15'][0] == 24639 and fixed['15'][0] == 25599
    with tempfile.TemporaryDirectory(prefix='terminal-q8-test-') as folder:
        for policy, positions in [('terminal_uniform_q8',fixed),('terminal_qgram_q8',selected)]:
            record = {'format':'basisserve.query_position_manifest.v1', 'method':policy,
                      'fit_document_ids':[0,1], 'num_fit_windows':2, 'context_length':32768,
                      'candidate_stride':64, 'num_bins':1, 'queries_per_bin':8,
                      'layers':{l:{'selected_positions':v} for l,v in positions.items()}}
            path = Path(folder)/f'{policy}.json'
            path.write_text(json.dumps(record))
            assert load_position_manifest(path) == record


def test_pivots_match_explicit_residuals():
    torch.manual_seed(8)
    z=torch.randn(17,9,dtype=torch.float64)
    positions=list(range(17))
    observed,_,_=select_positions_pivoted_gram(z@z.T,positions,7)
    residual=z.clone()
    chosen=[]
    for _ in range(7):
        energy=residual.square().sum(-1)
        energy[chosen]=-torch.inf
        pivot=int(energy.argmax())
        chosen.append(pivot)
        direction=residual[pivot]/residual[pivot].norm()
        residual-= (residual@direction)[:,None]*direction[None]
    assert observed==chosen


def test_degenerate_and_tied_grams():
    for gram in (torch.zeros(10,10),torch.ones(10,10),torch.eye(10)):
        selected,diagnostics,residual=select_positions_pivoted_gram(gram,list(range(10)),8)
        assert selected==list(range(8)) and len(diagnostics)==8
        assert residual['minimum_residual_diagonal']>=0


def test_whitening_sampling_and_determinism():
    torch.manual_seed(31)
    positions=candidate_positions(32768)
    q=torch.randn(2,len(positions),3,8)
    q[...,7]=q[...,6]  # singular moment, not an inverse() path
    kwargs=dict(context_length=32768)
    a,w,grams=select_stratified_query_positions(q,positions,**kwargs)
    b,_,_=select_stratified_query_positions(q,positions,**kwargs)
    assert canonical_hash(a)==canonical_hash(b) and torch.isfinite(w).all()
    selected=a['selected_positions']
    assert len(selected)==len(set(selected))==32
    assert all(p in positions and p>=32 for p in selected)
    assert all(sum(p//8192==bin for p in selected)==8 for bin in range(4))
    assert all(r==7 for r in a['whitening_effective_ranks'])
    for gram in grams:
        torch.testing.assert_close(gram,gram.T)
        assert torch.linalg.eigvalsh(gram).min()>-1e-9
    for length in (32768,65536,131072,10000):
        grid=candidate_positions(length)
        assert all(32<=p<length and (p+1)%64==0 for p in grid)
    assert uniform_query_positions(32768,32,.25)==list(range(24831,32768,256))
    assert uniform_query_positions(32768,32)==list(range(1023,32768,1024))


def test_diagnostic_files_cannot_influence_fit_selection():
    torch.manual_seed(12)
    positions=candidate_positions(4096)
    with tempfile.TemporaryDirectory(prefix='query-pivot-test-') as folder:
        root=Path(folder)
        artifacts={}
        for doc in range(2):
            path=root/f'window_{doc:03d}.safetensors'
            save_file({'queries':torch.randn(1,len(positions),2,6).bfloat16()},str(path))
            artifacts[str(doc)]={'file':path.name,'sha256':sha256(path)}
        record={'status':'complete','protocol':{'format':'basisserve.candidates_queries.v1',
                'fit_document_ids':[0,1],'documents':[0,1],'layers':[15]},'artifacts':artifacts}
        (root/'manifest.json').write_text(json.dumps(record))
        q=load_fit_candidates(root,15)
        a,_,_=select_stratified_query_positions(q,positions,context_length=4096)
        save_file({'queries':torch.full((1,len(positions),2,6),float('nan'))},str(root/'window_064.safetensors'))
        q2=load_fit_candidates(root,15)
        b,_,_=select_stratified_query_positions(q2,positions,context_length=4096)
        assert torch.equal(q,q2) and canonical_hash(a)==canonical_hash(b)


def test_selected_capture_layer_positions_and_diagnostic_loading():
    positions=uniform_query_positions(4096,32)
    with tempfile.TemporaryDirectory(prefix='query-manifest-test-') as folder:
        root=Path(folder)
        manifest={'format':'basisserve.query_position_manifest.v1','method':'full_uniform_q32',
            'context_length':4096,'num_fit_windows':2,'fit_document_ids':[0,1],
            'candidate_stride':64,'num_bins':4,'queries_per_bin':8,
            'model_config_sha256':'model','fit_token_sha256':'fit',
            'layers':{'15':{'selected_positions':positions}}}
        file=root/'positions.json'
        file.write_text(json.dumps(manifest))
        artifacts={}
        for doc in [0,1]+list(range(64,80)):
            path=root/f'window_{doc:03d}.safetensors'
            q=torch.stack((torch.zeros(32,4,8),torch.full((32,4,8),float(doc)))).bfloat16()
            save_file({'queries':q},str(path))
            artifacts[str(doc)]={'file':path.name,'sha256':sha256(path)}
        capture={'status':'complete','protocol':{'format':'basisserve.selected_queries.v1',
            'context_length':4096,'model_config_sha256':'model','fit_token_sha256':'fit',
            'position_manifest_sha256':sha256(file),'layers':[3,15],
            'positions_by_layer':{'3':positions,'15':positions}},'artifacts':artifacts}
        (root/'manifest.json').write_text(json.dumps(capture))
        fit,p=manifest_queries(file,root,'fit',15)
        diagnostic,p2=manifest_queries(file,root,'validation',15)
        assert fit.shape==(2,32,4,8) and diagnostic.shape==(16,32,4,8)
        assert fit[1].eq(1).all() and diagnostic[0].eq(64).all() and diagnostic[-1].eq(79).all()
        assert p.tolist()==p2.tolist()==positions
