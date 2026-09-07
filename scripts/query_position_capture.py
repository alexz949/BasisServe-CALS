"""Q-only candidate and manifest-selected modes for the existing Q capture CLI."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from basisserve.core.query_position_sampling import candidate_positions
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from scripts.capture_qwen3_8b_q16 import selected_rotated_queries


@torch.inference_mode()
def capture_position_queries(args):
    assert args.stage in ('candidates','selected') and args.num_shards == 1
    path = args.windows or args.calibration_root/'qwen3_8b_c4_64f16h_s32768/windows.safetensors'
    windows = load_file(str(path))['input_ids']
    assert windows.ndim == 2 and len(windows) >= 64
    sequence = windows.shape[1]
    selected = None
    if args.stage == 'candidates':
        assert 1 <= args.candidate_window_count <= 64
        documents = list(range(args.candidate_window_count))
        layers = sorted(set(args.candidate_layers))
        assert layers and all(0 <= l < 36 for l in layers)
        grid = candidate_positions(sequence,args.candidate_stride)
        positions = {str(l):grid for l in layers}
    else:
        assert args.query_position_manifest is not None
        selected = json.loads(args.query_position_manifest.read_text())
        assert selected['format'] == 'basisserve.query_position_manifest.v1'
        assert selected['context_length'] == sequence
        assert selected['model_config_sha256'] == sha256(args.model/'config.json')
        layers = sorted(map(int,selected['layers']))
        positions = {str(l):selected['layers'][str(l)]['selected_positions'] for l in layers}
        assert len({len(p) for p in positions.values()}) == 1
        documents = selected['fit_document_ids']+list(range(64,80))
        assert len(windows) >= 80
    assert not (args.output_dir/'manifest.json').exists()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fit_ids = [i for i in documents if i < 64]
    token_hash = hashlib.sha256(windows[fit_ids].contiguous().numpy().tobytes()).hexdigest()
    specification = {'format':f'basisserve.{args.stage}_queries.v1','context_length':sequence,
        'layers':layers,'positions_by_layer':positions,'documents':documents,
        'fit_document_ids':fit_ids,'fit_token_sha256':token_hash,
        'document_token_sha256':{str(d):hashlib.sha256(windows[d].contiguous().numpy().tobytes()).hexdigest() for d in documents},
        'model_config_sha256':sha256(args.model/'config.json'),'dtype':'bfloat16',
        'capture':'dense teacher SDPA; post-Q-norm/post-RoPE; Q-only storage',
        'source_sha256':sha256(Path(__file__))}
    if selected is not None:
        assert selected['fit_token_sha256'] == token_hash
        specification['position_manifest_sha256'] = sha256(args.query_position_manifest)
    else:
        specification['candidate_stride'] = args.candidate_stride
    reference_root = args.calibration_root/'q128_terminal8k'
    reference = json.loads((reference_root/'manifest.json').read_text()) if sequence == 32768 else None
    if reference is not None:
        assert reference['protocol']['model_config_sha256'] == specification['model_config_sha256']
    model, captured = None, {}
    def hook(layer):
        def collect(module,positional,kwargs):
            hidden = kwargs.get('hidden_states',positional[0] if positional else None)
            captured[layer] = selected_rotated_queries(module,hidden,kwargs['position_embeddings'],positions[str(layer)])[0].cpu().bfloat16().contiguous()
        return collect
    artifacts = {}
    for document in documents:
        path = args.output_dir/f'window_{document:03d}.safetensors'
        assert not path.exists()
        captured.clear()
        reuse = selected is not None and document < 64
        if reuse:
            root = Path(selected['candidate_capture_root'])
            source = json.loads((root/'manifest.json').read_text())
            assert sha256(root/'manifest.json') == selected['candidate_manifest_sha256']
            file = root/source['artifacts'][str(document)]['file']
            assert sha256(file) == source['artifacts'][str(document)]['sha256']
            data = load_file(str(file))['queries']
            for layer in layers:
                grid = source['protocol']['positions_by_layer'][str(layer)]
                slots = [grid.index(p) for p in positions[str(layer)]]
                captured[layer] = data[source['protocol']['layers'].index(layer),slots].contiguous()
        else:
            if model is None:
                assert torch.cuda.is_available()
                model = AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
                    local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa',device_map={'':0}).eval()
                handles = [model.model.layers[l].self_attn.register_forward_pre_hook(hook(l),with_kwargs=True) for l in layers]
            result = model.model(input_ids=windows[document:document+1].to('cuda:0'),use_cache=False)
            del result
        assert set(captured) == set(layers)
        if reference is not None and document < 64:
            file = reference_root/reference['artifacts'][str(document)]['file']
            assert sha256(file) == reference['artifacts'][str(document)]['sha256']
            prior = load_file(str(file))['queries']
            old_positions = reference['protocol']['query_positions']
            for layer in layers:
                current = positions[str(layer)]
                overlap = [p for p in current if p in old_positions]
                if overlap:
                    assert torch.equal(captured[layer][[current.index(p) for p in overlap]],prior[layer,[old_positions.index(p) for p in overlap]])
        save_file({'queries':torch.stack([captured[l] for l in layers])},str(path))
        artifacts[str(document)] = {'file':path.name,'sha256':sha256(path),'reused_candidates':reuse}
        print(f'[{args.stage}] window={document} layers={layers} Q={len(positions[str(layers[0])])}',flush=True)
    if model is not None:
        for handle in handles:
            handle.remove()
    write_json(args.output_dir/'manifest.json',{'status':'complete','protocol':specification,'artifacts':artifacts})
