# ICLR uniform C1 checkpoints: Hugging Face upload completed

Five models; uniform per-layer/per-KV-head ranks 32, 48, 64, 80, 96, 112. All 30 factor banks report completed decoder-refitted endpoints after encoder sweep 6. Existing factor files are reused; no training or evaluation is launched.

| Model | Ranks | Missing HF files | Upload GB |
|---|---|---:|---:|
| llama2-7b | 32, 48, 64, 80, 96, 112 | 204 | 3.782 |
| llama31-70b | 32, 48, 64, 80, 96, 112 | 492 | 36.343 |
| llama31-8b | 32, 48, 64, 80, 96, 112 | 204 | 3.665 |
| qwen3-32b | 32, 48, 64, 80, 96, 112 | 396 | 18.202 |
| qwen3-8b | 32, 48, 64, 80, 96, 112 | 190 | 3.360 |

Inventory: 1524 files, 66.116 GB; pending: 1486 files, 65.352 GB. Remote revision checked: `35fb02cb6361bc1caa01aba7806fa14c3bf4148b`. Existing remote file conflicts: 0.

Paths: `ICLR-results/<model>/checkpoints/<model-prefix>-C1U-R<rank>/manifest.json`, plus the referenced `ICLR-results/<model>/c1/factor-banks/R<rank>-S6/results.json` and per-layer `.safetensors`. The previously existing Q3-8B-C1U-R80 manifest is preserved. Calibration snapshots and unrelated experiment files are excluded.

## Validation

Preparation smoke passed: model configuration hashes, rank, layer coverage, sweep-6 endpoint metadata, BF16 fit configuration, safetensors tensor shapes, and remote metadata compatibility. Full local weight hashes are checked before any upload in the formal command. Completion requires remote hash verification for every payload file and preservation of existing remote payloads. These checks do not constitute new model accuracy evaluation.

## Commands and execution

Environment: basis. Preparation:
```bash
/home/zhangal/.conda/envs/basis/bin/python scripts/upload_iclr_uniform_hf.py
```

Formal validation and upload:
```bash
/home/zhangal/.conda/envs/basis/bin/python -u scripts/upload_iclr_uniform_hf.py --upload
```

Slurm allocation: small partition, one CPU node, 2 CPUs, 8 GiB RAM, no GPU, 12-hour limit. Log: `results/uploads/iclr-uniform-hf/upload.log`. Submit script will be deleted immediately after submission. Reuse the same report/cache directories if interrupted.

Machine-readable file list: `results/uploads/iclr-uniform-hf/plan.json`. Full hash inventory and final verification will be saved alongside it. Preparation was followed by the successful upload described below.

## Completion

Completed 2026-09-09 using Slurm job 8303219 in the basis environment. The existing Hugging Face repository is public, as explicitly requested. Anonymous repository access was verified.

All 30 uniform checkpoints and 1,524 payload files (66,115,743,146 bytes; 61.575 GiB) passed final remote hash verification. No existing payload files changed. Revision: `f1a6253b5d5c747a2475cbf9e704a67d97930b31`. Final verification: `results/uploads/iclr-uniform-hf/result.json`; progress log: `results/uploads/iclr-uniform-hf/upload.log`.

Earlier attempts encountered API/commit rate limits and the private storage quota. The completed uploader uses fixed batches of at most 250 files. All remaining 731 files were committed after the repository became public.
