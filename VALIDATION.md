# Feature-major ragged validation

Date: 2026-08-23. All Python validation used the repository-required `lowrank`
conda environment.

## Environment

```text
torch=2.11.0+cu130
torch CUDA=13.0
NCCL=2.28.9 (22809)
CUDA available on login node=False
```

Consequently `feature_direct` can be built, while the NCCL 2.29+
`feature_rma` runtime is unavailable in this environment.

## Python syntax and lint

```bash
/home/zhangal/.conda/envs/lowrank/bin/python -m py_compile \
  basisserve/kernels/ragged_allgather.py \
  basisserve/kernels/feature_ragged_allgather.py \
  basisserve/core/c1_tp_feature_decode.py \
  benchmarks/bench_feature_ragged_allgather.py \
  tests/test_feature_ragged_layout.py \
  tests/test_capsule_imports.py

PATH=/home/zhangal/.conda/envs/lowrank/bin:$PATH ruff check \
  basisserve/kernels/ragged_allgather.py \
  basisserve/kernels/feature_ragged_allgather.py \
  basisserve/core/c1_tp_feature_decode.py \
  benchmarks/bench_feature_ragged_allgather.py \
  tests/test_feature_ragged_layout.py \
  tests/test_capsule_imports.py
```

Both commands exited 0; Ruff reported `All checks passed!`.

## CPU tests

```bash
/home/zhangal/.conda/envs/lowrank/bin/python -m pytest -q \
  tests/test_feature_ragged_layout.py \
  tests/test_c1_tp_decode.py \
  tests/test_c1_variable_v_attention.py \
  tests/test_capsule_imports.py
```

```text
....................                                                     [100%]
20 passed in 1.74s
```

## C++ extension builds

The actual feature extension loader was invoked with one build worker after
temporarily bypassing only its login-node GPU visibility guard. It compiled and
loaded without creating a communicator or launching GPU work:

```text
loader_nccl_version 22809
```

The double-buffered RMA branch was separately compiled to an object file
against NVIDIA's official NCCL 2.29.7 header. This validates the public
`ncclPutSignal`/`ncclWaitSignal` signatures without linking or loading a second
NCCL runtime:

```text
compile_only_nccl_header=22907
```

The pre-existing ragged extension could not be rebuilt in this login
environment because the installed `nvcc` is CUDA 13.2 while the installed
CUDA/CCCL headers are 13.0. The NVIDIA compatibility check rejected that mixed
toolchain. The new feature extension contains no `.cu` source and is unaffected.

## Multi-GPU execution

Not run: the login node has no visible GPU, and the current NCCL 2.28.9 runtime
does not expose host RMA. TP correctness, repeated-signal stress, and throughput
remain required on a coherent NCCL 2.29+ TP8 environment.
