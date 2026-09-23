# Qwen3 TP8 V-only Source Freeze

The 11 source files in `source.tar.gz` were archived before the formal memory grid. `tar -tzf source.tar.gz` listed all 11 successfully. The archive includes the V-only attention/placement module, memory runner, grid launcher, summarizer, Qwen prompt preparation, three smoke/test files, and reused TP8 CPU-affinity/metadata helpers.

The repository base commit at archive time was `bae46c1409d3fa00030bab88df8b359e20b886bf`; the worktree was dirty, including these new experiment files. The source archive, not the base commit alone, identifies the implementation used for the proposed grid. No SHA validation was performed.

Formal settings: Qwen3-8B-Base revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, complete STAR `fused.pt` from HF revision `0ef83dff27205b131c82df6d62636129e9dac7b9`, Basis V64 factors from HF revision `0872566b1da66eb4c813d7a1cb3313325f22b287`, TP8 on eight L40S GPUs, BF16, TF32 off, dense exact K, full Flash SDPA, no offload, 4096-token chunks, one decode step after prefill. The user-confirmed grid has 36 trials: six lengths, three batches, two arms, and preselected cohort 0. The actual STAR export retains 54.0473% of global V rank.
