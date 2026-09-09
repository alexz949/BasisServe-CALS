#!/usr/bin/env python3
"""Evaluate Qwen3-8B Dense and C1 generation arms with local vLLM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

import lm_eval
from lm_eval.api.model import TemplateLM
from lm_eval.models.utils import (
    handle_stop_sequences,
    normalize_gen_kwargs,
    postprocess_generated_text,
)
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table
import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.vllm import (  # noqa: E402
    QWEN3_8B_FOLDED_C1_MODEL_ARCHITECTURE,
    register as register_basisserve_vllm,
)


FORMAT = "basisserve.qwen3_8b.r80_generation_vllm.v1"
STAGE_FORMAT = "basisserve.qwen3_8b.r80_generation_vllm.task.v1"
CHECKPOINT_FORMAT = "basisserve.qwen3_8b.iclr_v_factors.v1"
TASKS = ("ifeval", "gsm8k", "humaneval")
TASK_FEWSHOT = {"ifeval": 0, "gsm8k": 5, "humaneval": 0}
TASK_MAX_GEN_TOKS = {"ifeval": 1280, "gsm8k": 512, "humaneval": 1024}
REQUIRED_MAX_LENGTH = 4096
TENSOR_PARALLEL_SIZE = 1
TASK_INCLUDE_PATH = None
TASK_GEN_KWARGS = {}
UNSAFE_TASKS = frozenset(("humaneval",))
EXPECTED_ARMS = {
    "Q3-8B-Dense": {"method": "dense"},
    "Q3-8B-C1-R80": {
        "method": "c1-two-sided-kl",
        "equivalent_rank_target": 80,
    },
    "Q3-8B-C1U-R80": {
        "method": "c1-uniform",
        "equivalent_rank_target": 80,
    },
    "Q3-8B-C1-R96": {
        "method": "c1-two-sided-kl",
        "equivalent_rank_target": 96,
    },
}
SUPPORTED_ARMS = (
    "Q3-8B-Dense",
    "Q3-8B-C1-R80",
    "Q3-8B-C1U-R80",
    "Q3-8B-C1-R96",
)
PINNED_VLLM_VERSION = "0.18.1.dev0+gbcf2be961.d20260828.cu128"
EXPECTED_GPU = "NVIDIA L40S"
FOLDED_ARCHITECTURE = QWEN3_8B_FOLDED_C1_MODEL_ARCHITECTURE
ATTENTION_CONFIG = None


class LocalVLLMGenerationLM(TemplateLM):
    """Minimal lm-eval adapter for generation-only local vLLM tasks."""

    backend = "causal"

    def __init__(
        self,
        engine: Any,
        tokenizer: Any,
        *,
        max_length: int,
        default_max_gen_toks: int,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.tokenizer = tokenizer
        self._max_length = int(max_length)
        self._default_max_gen_toks = int(default_max_gen_toks)
        self.generation_records = []

    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._default_max_gen_toks

    @property
    def tokenizer_name(self) -> str:
        return str(self.tokenizer.name_or_path).replace("/", "__")

    def tok_encode(
        self,
        string: str | list[str],
        add_special_tokens: bool | None = None,
        **kwargs: Any,
    ) -> list[int] | list[list[int]]:
        add_special = False if add_special_tokens is None else add_special_tokens
        if isinstance(string, str):
            return list(
                map(
                    int,
                    self.tokenizer.encode(
                        string,
                        add_special_tokens=add_special,
                        **kwargs,
                    ),
                )
            )
        return [
            list(
                map(
                    int,
                    self.tokenizer.encode(
                        value,
                        add_special_tokens=add_special,
                        **kwargs,
                    ),
                )
            )
            for value in string
        ]

    def _loglikelihood_tokens(
        self,
        requests: list[Any],
        **kwargs: Any,
    ) -> list[tuple[float, bool]]:
        del kwargs
        assert not requests
        return []

    def loglikelihood_rolling(
        self,
        requests: list[Any],
        disable_tqdm: bool = False,
    ) -> list[float]:
        del disable_tqdm
        assert not requests
        return []

    def generate_until(
        self,
        requests: list[Any],
        disable_tqdm: bool = False,
    ) -> list[str]:
        from vllm import SamplingParams, TokensPrompt

        eos = self.tokenizer.decode(self.eot_token_id)
        prompts = []
        sampling_parameters = []
        cache_records = []
        metadata = []
        occurrences = {}
        for request in requests:
            context, raw_kwargs = request.args
            kwargs = normalize_gen_kwargs(
                raw_kwargs,
                default_max_gen_toks=self._default_max_gen_toks,
            )
            until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            max_gen_toks = int(kwargs.pop("max_gen_toks"))
            kwargs.pop("do_sample", None)
            kwargs.pop("max_length", None)
            token_ids = self.tok_encode(context)
            assert isinstance(token_ids, list)
            original_tokens = len(token_ids)
            prompt_hash = hashlib.sha256(context.encode()).hexdigest()
            key = (request.task_name, request.doc_id, prompt_hash)
            sample_index = occurrences.get(key, 0)
            occurrences[key] = sample_index + 1
            if float(kwargs.get("temperature", 0)) > 0:
                kwargs["seed"] = (int(prompt_hash[:8], 16) + sample_index) % (2**31)
            maximum_context = self._max_length - max_gen_toks
            assert maximum_context > 0
            token_ids = token_ids[-maximum_context:]
            metadata.append({
                "task": request.task_name, "doc_id": request.doc_id,
                "sample_index": sample_index, "prompt_sha256": prompt_hash,
                "seed": kwargs.get("seed"), "original_prompt_tokens": original_tokens,
                "retained_prompt_tokens": len(token_ids), "max_gen_toks": max_gen_toks,
                "stop_strings": until,
            })
            prompts.append(TokensPrompt(prompt_token_ids=token_ids))
            sampling_parameters.append(
                SamplingParams(
                    max_tokens=max_gen_toks,
                    stop=until,
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                    **kwargs,
                )
            )
            cache_records.append(
                (
                    context,
                    kwargs
                    | {"until": until, "max_gen_toks": max_gen_toks},
                )
            )
        outputs = self.engine.generate(
            prompts,
            sampling_params=sampling_parameters,
            use_tqdm=not disable_tqdm,
        )
        assert len(outputs) == len(requests)
        for output, record in zip(outputs, metadata, strict=True):
            completion = output.outputs[0]
            self.generation_records.append(record | {
                "generated_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
            })
        responses = []
        for output, (context, generation_kwargs) in zip(
            outputs,
            cache_records,
            strict=True,
        ):
            text = postprocess_generated_text(
                output.outputs[0].text,
                generation_kwargs["until"],
                None,
            )
            responses.append(text)
            self.cache_hook.add_partial(
                "generate_until",
                (context, generation_kwargs),
                text,
            )
        return responses

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_num_parameters": -1,
            "model_dtype": "torch.bfloat16",
        }


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _metric(
    task_metrics: Mapping[str, Any],
    metric: str,
    filter_name: str,
) -> float | None:
    value = task_metrics.get(f"{metric},{filter_name}")
    return None if value is None else float(value)


def _summarize_task(
    task: str,
    evaluation: Mapping[str, Any],
) -> dict[str, float] | None:
    task_metrics = evaluation.get("results", {}).get(task, {})
    if task == "ifeval":
        selected = {
            "prompt_level_strict_accuracy": _metric(
                task_metrics, "prompt_level_strict_acc", "none"
            ),
            "instruction_level_strict_accuracy": _metric(
                task_metrics, "inst_level_strict_acc", "none"
            ),
            "prompt_level_loose_accuracy": _metric(
                task_metrics, "prompt_level_loose_acc", "none"
            ),
            "instruction_level_loose_accuracy": _metric(
                task_metrics, "inst_level_loose_acc", "none"
            ),
        }
    elif task == "gsm8k":
        selected = {
            "strict_exact_match": _metric(
                task_metrics, "exact_match", "strict-match"
            ),
            "flexible_exact_match": _metric(
                task_metrics, "exact_match", "flexible-extract"
            ),
        }
    else:
        selected = {
            "pass_at_1": _metric(task_metrics, "pass@1", "create_test"),
        }
    if not _check(
        all(value is not None for value in selected.values()),
        f"missing expected metrics for {task}: {task_metrics}",
    ):
        return None
    return {name: float(value) for name, value in selected.items()}


def _validate_arm(run_id: str, compression: Mapping[str, Any]) -> bool:
    expected = EXPECTED_ARMS[run_id]
    return all(
        _check(
            compression.get(key) == value,
            f"{run_id} has {key}={compression.get(key)!r}, expected {value!r}",
        )
        for key, value in expected.items()
    )


def _stage_matches(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    task: str,
    checkpoint_manifest_sha256: str,
    engine_configuration: Mapping[str, Any],
    limit: float | None,
    confirm_run_unsafe_code: bool,
) -> bool:
    return all(
        (
            payload.get("format") == STAGE_FORMAT,
            payload.get("status") == "complete",
            payload.get("run_id") == run_id,
            payload.get("task") == task,
            payload.get("checkpoint", {}).get("manifest_sha256")
            == checkpoint_manifest_sha256,
            payload.get("runtime", {}).get("engine_configuration")
            == engine_configuration,
            payload.get("protocol", {}).get("limit") == limit,
            payload.get("protocol", {}).get("confirm_run_unsafe_code")
            == confirm_run_unsafe_code,
        )
    )


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> int:
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device_names = [
        torch.cuda.get_device_name(index) for index in range(cuda_device_count)
    ]
    selected_tasks = (args.task,)
    valid = all(
        (
            _check(cuda_available, "CUDA is required"),
            _check(
                cuda_device_count == TENSOR_PARALLEL_SIZE,
                f"evaluation must see {TENSOR_PARALLEL_SIZE} GPUs",
            ),
            _check(
                all(name == EXPECTED_GPU for name in cuda_device_names),
                f"unexpected GPUs: {cuda_device_names}",
            ),
            _check(args.max_length == REQUIRED_MAX_LENGTH, f"protocol requires max length {REQUIRED_MAX_LENGTH}"),
            _check(args.max_num_seqs > 0, "max_num_seqs must be positive"),
            _check(args.max_num_batched_tokens >= args.max_length, "batched-token limit must cover max length"),
            _check(0.0 < args.gpu_memory_utilization < 1.0, "invalid GPU memory utilization"),
            _check(importlib.metadata.version("vllm") == PINNED_VLLM_VERSION, f"vLLM must be {PINNED_VLLM_VERSION}"),
            _check(
                args.task not in UNSAFE_TASKS or args.confirm_run_unsafe_code,
                "HumanEval requires --confirm-run-unsafe-code",
            ),
            _check(
                args.task not in UNSAFE_TASKS
                or os.environ.get("HF_ALLOW_CODE_EVAL") == "1",
                "HumanEval requires HF_ALLOW_CODE_EVAL=1",
            ),
        )
    )
    if not valid:
        return 2
    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    model_path = Path(args.model).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "manifest.json"
    if not _check(manifest_path.is_file(), f"missing checkpoint manifest: {manifest_path}"):
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_manifest_sha256 = _sha256(manifest_path)
    run_id = str(manifest.get("run_id"))
    manifest_valid = all(
        (
            _check(manifest.get("format") == CHECKPOINT_FORMAT, "checkpoint format mismatch"),
            _check(manifest.get("status") == "complete", "checkpoint is incomplete"),
            _check(run_id == args.run_id, "run ID mismatch"),
            _check(run_id in SUPPORTED_ARMS, "unsupported vLLM generation arm"),
            _check(manifest.get("model", {}).get("config_sha256") == _sha256(model_path / "config.json"), "checkpoint model hash mismatch"),
            _check(_validate_arm(run_id, manifest.get("compression", {})), "arm manifest mismatch"),
        )
    )
    if not manifest_valid:
        return 2
    compression = manifest["compression"]
    dense = compression["method"] == "dense"
    artifact_sha256 = None if dense else str(manifest["artifact"]["sha256"])
    architecture = (
        "Qwen3ForCausalLM"
        if dense
        else FOLDED_ARCHITECTURE
    )
    hf_overrides = None
    if not dense:
        hf_overrides = {
            "architectures": [architecture],
            "basisserve_c1_checkpoint_dir": str(checkpoint_dir),
            "basisserve_c1_manifest_sha256": checkpoint_manifest_sha256,
        }
    engine_configuration = {
        "attention_config": ATTENTION_CONFIG,
        "vllm_version": PINNED_VLLM_VERSION,
        "model_architecture": architecture,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "data_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": args.max_length,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "enforce_eager": args.enforce_eager,
        "c1_runtime": None if dense else "factors folded into standard dense V/O slots",
        "compact_c1_cache": False,
    }
    final_path = output_dir / "result.json"
    if final_path.is_file():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if _check(
            existing.get("format") == FORMAT
            and existing.get("status") == "complete"
            and existing.get("run_id") == run_id
            and existing.get("checkpoint", {}).get("manifest_sha256")
            == checkpoint_manifest_sha256
            and existing.get("runtime", {}).get("engine_configuration")
            == engine_configuration
            and existing.get("protocol", {}).get("tasks")
            == list(selected_tasks)
            and existing.get("protocol", {}).get("limit") == args.limit
            and existing.get("protocol", {}).get("confirm_run_unsafe_code")
            == args.confirm_run_unsafe_code,
            f"existing final vLLM result does not match: {final_path}",
        ):
            print(f"[Resume] complete result already exists: {final_path}", flush=True)
            return 0
        return 2
    stage_paths = {
        task: output_dir / f"{task}.json" for task in selected_tasks
    }
    stages: dict[str, dict[str, Any] | None] = {}
    for task, path in stage_paths.items():
        if not path.is_file():
            stages[task] = None
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not _check(
            _stage_matches(
                payload,
                run_id=run_id,
                task=task,
                checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                engine_configuration=engine_configuration,
                limit=args.limit,
                confirm_run_unsafe_code=args.confirm_run_unsafe_code,
            ),
            f"existing vLLM stage does not match: {path}",
        ):
            return 2
        stages[task] = payload
        print(f"[Resume] reusing {path}", flush=True)

    from vllm import LLM

    register_basisserve_vllm()
    started = time.perf_counter()
    engine = LLM(
        attention_config=ATTENTION_CONFIG,
        model=str(model_path),
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        data_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.max_length,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        hf_overrides=hf_overrides,
        disable_log_stats=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    adapter = LocalVLLMGenerationLM(
        engine,
        tokenizer,
        max_length=args.max_length,
        default_max_gen_toks=TASK_MAX_GEN_TOKS[args.task],
    )
    runtime = {
        "engine_configuration": engine_configuration,
        "cuda_device_names": cuda_device_names,
    }
    for task in selected_tasks:
        if stages[task] is not None:
            continue
        task_started = time.perf_counter()
        adapter.generation_records = []
        evaluation = lm_eval.simple_evaluate(
            model=adapter,
            tasks=[task],
            num_fewshot=TASK_FEWSHOT[task],
            task_manager=TaskManager(include_path=TASK_INCLUDE_PATH),
            gen_kwargs=TASK_GEN_KWARGS.get(task),
            log_samples=True,
            limit=args.limit,
            confirm_run_unsafe_code=args.confirm_run_unsafe_code,
        )
        if not _check(evaluation is not None, "lm-eval returned no results"):
            return 2
        _write_json(output_dir / "generation_records.json", adapter.generation_records)
        metrics = _summarize_task(task, evaluation)
        if metrics is None:
            return 2
        stage = {
            "format": STAGE_FORMAT,
            "status": "complete",
            "run_id": run_id,
            "task": task,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "model": manifest["model"],
            "checkpoint": {
                "directory": str(checkpoint_dir),
                "manifest_sha256": checkpoint_manifest_sha256,
                "artifact_sha256": artifact_sha256,
                "format": CHECKPOINT_FORMAT,
            },
            "compression": compression,
            "protocol": {
                "task": task,
                "task_default_num_fewshot": TASK_FEWSHOT[task],
                "use_task_default_num_fewshot": True,
                "max_gen_toks": TASK_MAX_GEN_TOKS[task],
                "max_length": args.max_length,
                "apply_chat_template": False,
                "fewshot_as_multiturn": False,
                "log_samples": True,
                "limit": args.limit,
                "confirm_run_unsafe_code": args.confirm_run_unsafe_code,
                "generation_backend": "vllm-local-continuous-batching",
            },
            "runtime": runtime,
            "metrics": metrics,
            "evaluation": evaluation,
            "elapsed_seconds": time.perf_counter() - task_started,
        }
        _write_json(stage_paths[task], stage)
        stages[task] = stage
        print(make_table(evaluation), flush=True)
        print(f"[{task}] metrics={json.dumps(metrics, sort_keys=True)}", flush=True)

    if not _check(all(stages.values()), "one or more vLLM stages are incomplete"):
        return 2
    result = {
        "format": FORMAT,
        "status": "complete",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "model": manifest["model"],
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "manifest_sha256": checkpoint_manifest_sha256,
            "artifact_sha256": artifact_sha256,
            "format": CHECKPOINT_FORMAT,
        },
        "compression": compression,
        "protocol": {
            "tasks": list(selected_tasks),
            "limit": args.limit,
            "confirm_run_unsafe_code": args.confirm_run_unsafe_code,
            "task_max_gen_toks": {
                task: TASK_MAX_GEN_TOKS[task] for task in selected_tasks
            },
            "generation_backend": "vllm-local-continuous-batching",
        },
        "metrics": {
            task: stages[task]["metrics"]
            for task in selected_tasks  # type: ignore[index]
        },
        "runtime": runtime,
        "stages": {
            task: {
                "file": path.name,
                "sha256": _sha256(path),
            }
            for task, path in stage_paths.items()
        },
        "elapsed_seconds_this_attempt": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "lm_eval": importlib.metadata.version("lm-eval"),
            "vllm": importlib.metadata.version("vllm"),
            "cuda_devices": cuda_device_names,
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(0)
            ),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _write_json(final_path, result)
    print(f"[Result] wrote {final_path}", flush=True)
    print("[Shutdown] closing vLLM engine", flush=True)
    engine.llm_engine.engine_core.shutdown(timeout=30)
    print("[Shutdown] vLLM engine closed", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", choices=SUPPORTED_ARMS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--confirm-run-unsafe-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(evaluate(parse_args()))
