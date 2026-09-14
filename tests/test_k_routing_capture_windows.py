import torch
from safetensors.torch import save_file

from evaluation.k_routing_capture_windows import CapturedWindows, read_capture_windows
from evaluation.v96kl_common import write_json, sha256
from basisserve.core.query_position_sampling import candidate_positions


def test_window_reader_matches_eager_fit_input_order_and_slices(tmp_path):
    torch.manual_seed(9)
    protocol=dict(sequence_length=256,num_shards=2)
    identity=dict(hkv=2,hq=4,head_dim=8)
    windows={}
    for shard,ids in enumerate(([64,0],[1])):
        path=tmp_path/'layer_000'/f'shard_{shard}.safetensors'
        path.parent.mkdir(parents=True,exist_ok=True)
        tensors={name:torch.randn(len(ids),*shape).bfloat16() for name,shape in
            dict(rows=(256,2,16),pre_rope_keys=(256,2,8),
                candidate_queries=(len(candidate_positions(256)),4,8)).items()}
        save_file(tensors,str(path))
        for offset,window in enumerate(ids):
            windows[window]={name:tensor[offset].clone() for name,tensor in tensors.items()}
        write_json(path.with_suffix('.json'),dict(status='complete',protocol=protocol,layer=0,
            candidate_positions=candidate_positions(256),window_ids=ids,sha256=sha256(path)))
    eager={name:torch.stack([windows[i][name] for i in (0,1,64)]) for name in windows[0]}
    streamed,actual_records=read_capture_windows(tmp_path,0,[0,1,64],protocol,identity)
    assert [record['window_ids'] for record in actual_records]==[[64,0],[1]]
    assert isinstance(streamed['rows'],CapturedWindows)
    for name in ('rows','pre_rope_keys'):
        source=streamed[name]
        assert source.shape==eager[name].shape and source.ndim==4
        torch.testing.assert_close(torch.stack(list(source)),eager[name],rtol=0,atol=0)
        torch.testing.assert_close(source[1,...,:8],eager[name][1,...,:8],rtol=0,atol=0)
        torch.testing.assert_close(source[2,:17,:,3:],eager[name][2,:17,:,3:],rtol=0,atol=0)
        torch.testing.assert_close(torch.stack(list(source[1:])),eager[name][1:],rtol=0,atol=0)
        # Returned tensors own their storage and cannot mutate the capture.
        source[0].zero_()
        torch.testing.assert_close(source[0],eager[name][0],rtol=0,atol=0)
    torch.testing.assert_close(streamed['candidate_queries'],eager['candidate_queries'],rtol=0,atol=0)
