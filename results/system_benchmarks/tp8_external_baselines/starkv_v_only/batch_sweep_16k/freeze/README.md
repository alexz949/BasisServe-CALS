# Source Snapshot Before 16K Batch Sweep

`source.tar.gz` contains 14 source and test files used by this batch-sweep protocol, including the shared V-only model runner, the new grid/summarizer, prompt preparation, and reused TP8 helpers. The repository base commit is `bae46c1409d3fa00030bab88df8b359e20b886bf`; the worktree contains uncommitted experiment files. The archive identifies the exact source snapshot proposed for the formal run. No SHA validation was performed.

Artifact revisions: Qwen3-8B-Base `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; complete STAR `fused.pt` from HF revision `0ef83dff27205b131c82df6d62636129e9dac7b9`; Basis V64 factors from HF revision `0872566b1da66eb4c813d7a1cb3313325f22b287`. The environment is `basis`: PyTorch `2.13.0+cu130`, CUDA `13.0`, NCCL `2.29.7`, Transformers `5.17.0`, and FlashAttention `2.8.3.post1` on eight NVIDIA L40S GPUs.
