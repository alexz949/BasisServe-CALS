"""Read one immutable captured activation window at a time."""
from dataclasses import dataclass

import torch
from safetensors import safe_open

from evaluation.v96kl_common import read_json, sha256
from basisserve.core.query_position_sampling import candidate_positions


@dataclass(frozen=True)
class CapturedWindows:
    locations: tuple
    name: str
    window_shape: tuple

    @property
    def shape(self):
        return (len(self.locations), *self.window_shape)

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return len(self.locations)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        first, *remaining = index if isinstance(index, tuple) else (index,)
        if isinstance(first, slice):
            assert not remaining
            return CapturedWindows(self.locations[first], self.name, self.window_shape)
        assert isinstance(first, int) and -len(self) <= first < len(self)
        path, offset = self.locations[first]
        with safe_open(str(path), framework='pt', device='cpu') as source:
            window = source.get_slice(self.name)[offset]
            if remaining:
                window = window[tuple(remaining)]
            # Detach the requested window from the shard's memory mapping.
            return window.clone()


def read_capture_windows(directory, layer, window_ids, expected, identity):
    length = expected['sequence_length']
    g, h, d = identity['hkv'], identity['hq'], identity['head_dim']
    shapes = dict(rows=(length,g,2*d), pre_rope_keys=(length,g,d),
        candidate_queries=(len(candidate_positions(length)),h,d))
    assert len(window_ids)==len(set(window_ids))
    locations, records = {}, []
    for shard in range(expected['num_shards']):
        path = directory/f'layer_{layer:03d}'/f'shard_{shard}.json'
        record = read_json(path)
        assert record['status']=='complete' and record['protocol']==expected and record['layer']==layer
        assert record['candidate_positions']==candidate_positions(length)
        tensor_path=path.with_suffix('.safetensors')
        assert sha256(tensor_path)==record['sha256']
        assert not set(locations).intersection(record['window_ids'])
        assert len(record['window_ids'])==len(set(record['window_ids']))
        assert set(record['window_ids'])<=set(window_ids)
        with safe_open(str(tensor_path),framework='pt',device='cpu') as source:
            assert set(source.keys())==set(shapes)
            for name,shape in shapes.items():
                assert tuple(source.get_slice(name).get_shape())==(len(record['window_ids']),*shape)
                assert source.get_tensor(name).dtype==torch.bfloat16
        for offset,window in enumerate(record['window_ids']):
            locations[window]=(tensor_path,offset)
        records.append(dict(manifest=str(path.resolve()),manifest_sha256=sha256(path),
            sha256=record['sha256'],window_ids=record['window_ids']))
    assert set(locations)==set(window_ids)
    ordered=tuple(locations[window] for window in window_ids)
    tensors={name:CapturedWindows(ordered,name,shape) for name,shape in shapes.items()}
    # Query selection uses all candidate queries; full tokenwise V/K stay on disk.
    queries=torch.empty((len(window_ids),*shapes['candidate_queries']),dtype=torch.bfloat16)
    for index,window in enumerate(tensors['candidate_queries']):
        queries[index].copy_(window)
    tensors['candidate_queries']=queries
    return tensors,records
