"""Read native gated-V captures in bounded row slices."""

from pathlib import Path

import torch
from safetensors import safe_open


class WindowRows:
    def __init__(self, paths, key):
        self.paths, self.key = tuple(map(Path, paths)), key
        assert self.paths
        with safe_open(str(self.paths[0]), framework='pt', device='cpu') as f:
            self.window_shape = tuple(f.get_slice(key).get_shape())
        self.shape = (len(self.paths) * self.window_shape[0], *self.window_shape[1:])
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, selection):
        assert isinstance(selection, slice)
        start, stop, step = selection.indices(len(self))
        assert step == 1 and start < stop
        rows = []
        width = self.window_shape[0]
        while start < stop:
            window, offset = divmod(start, width)
            count = min(stop-start, width-offset)
            with safe_open(str(self.paths[window]), framework='pt', device='cpu') as f:
                view = f.get_slice(self.key)
                assert tuple(view.get_shape()) == self.window_shape
                rows.append(view[offset:offset+count])
            start += count
        return rows[0] if len(rows) == 1 else torch.cat(rows)


class WindowCapture:
    def __init__(self, paths, head_to_group):
        self.z, self.gate, self.target = (WindowRows(paths, key) for key in ('z', 'gate', 'target'))
        with safe_open(str(paths[0]), framework='pt', device='cpu') as f:
            self.weight = f.get_tensor('weight')
        self.head_to_group = head_to_group
        self.validated_groups = None
        self.target_energy = None

    def preload_inputs(self, device, chunk_rows=2048):
        """Keep repeated normal-operator inputs resident without staging a full CPU copy."""
        assert self.validated_groups is not None and chunk_rows > 0
        for name in ('z', 'gate'):
            source = getattr(self, name)
            dtype = source[:1].dtype
            resident = torch.empty(source.shape, dtype=dtype, device=device)
            for start in range(0, len(source), chunk_rows):
                selection = slice(start, start+chunk_rows)
                resident[selection].copy_(source[selection])
            setattr(self, name, resident)

    def validate(self, groups):
        if self.validated_groups == groups:
            return
        assert self.z.ndim == 3 and self.gate.shape == self.z.shape
        assert tuple(self.weight.shape[:2]) == self.z.shape[1:]
        assert self.target.shape == (len(self.z), self.weight.shape[-1])
        assert self.head_to_group.shape == (self.z.shape[1],)
        assert self.head_to_group.dtype == torch.long
        assert set(self.head_to_group.tolist()) == set(range(groups))
        assert torch.isfinite(self.weight).all()
        energy = 0.0
        for start in range(0, len(self.z), 2048):
            selection = slice(start, start+2048)
            z, gate, target = self.z[selection], self.gate[selection], self.target[selection]
            assert all(torch.isfinite(t).all() for t in (z, gate, target))
            assert ((gate >= 0) & (gate <= 1)).all()
            energy += float(target.double().square().sum())
        self.target_energy = energy
        self.validated_groups = groups
