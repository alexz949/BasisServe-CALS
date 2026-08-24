# Added prototype: asynchronous feature-major ragged transport

New files:

- `basisserve/kernels/csrc/feature_ragged_allgather.cpp`
- `basisserve/kernels/feature_ragged_allgather.py`
- `basisserve/core/c1_tp_feature_decode.py`
- `benchmarks/bench_feature_ragged_allgather.py`
- `tests/test_feature_ragged_layout.py`
- `docs/feature_ragged_one_sided.md`

The transport implementation is opt-in, but it reuses the existing
`StaticRaggedPlan` and validated `PackedC1TPLayer` serving boundary. The TP
benchmark includes the existing registered ragged path and padded AllGather as
controls. The one-sided receive workspace alternates between two slots so a
faster rank cannot overwrite a slower rank's in-flight decoder input.

Start with the CPU layout test:

```bash
conda run -n lowrank python -m pytest -q tests/test_feature_ragged_layout.py
```

Then run the TP benchmark shown in `docs/feature_ragged_one_sided.md`.
