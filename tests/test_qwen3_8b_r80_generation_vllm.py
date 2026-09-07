from __future__ import annotations

from evaluation.eval_qwen3_8b_r80_generation_vllm import (
    LocalVLLMGenerationLM,
    _stage_matches,
    _summarize_task,
    _validate_arm,
)


class _Tokenizer:
    eos_token_id = 2
    name_or_path = "Qwen/Qwen3-8B-Base"

    @staticmethod
    def encode(value: str, add_special_tokens: bool, **kwargs):
        del add_special_tokens, kwargs
        return [ord(character) for character in value]


class _Engine:
    pass


def test_local_vllm_adapter_matches_no_bos_tokenization_protocol() -> None:
    adapter = LocalVLLMGenerationLM(
        _Engine(),
        _Tokenizer(),
        max_length=4096,
        default_max_gen_toks=1280,
    )
    assert adapter.tok_encode("ab") == [97, 98]
    assert adapter.tok_encode(["a", "bc"]) == [[97], [98, 99]]
    assert adapter.eot_token_id == 2
    assert adapter.max_length == 4096
    assert adapter.max_gen_toks == 1280
    assert adapter.tokenizer_name == "Qwen__Qwen3-8B-Base"


def test_vllm_stage_resume_pins_the_complete_engine_configuration() -> None:
    configuration = {
        "vllm_version": "0.18.1.dev0+gbcf2be961.d20260828.cu128",
        "model_architecture": "Qwen3ForCausalLM",
        "max_num_seqs": 64,
    }
    payload = {
        "format": "basisserve.qwen3_8b.r80_generation_vllm.task.v1",
        "status": "complete",
        "run_id": "Q3-8B-Dense",
        "task": "ifeval",
        "checkpoint": {"manifest_sha256": "abc"},
        "runtime": {"engine_configuration": configuration},
        "protocol": {
            "limit": None,
            "confirm_run_unsafe_code": False,
        },
    }
    assert _stage_matches(
        payload,
        run_id="Q3-8B-Dense",
        task="ifeval",
        checkpoint_manifest_sha256="abc",
        engine_configuration=configuration,
        limit=None,
        confirm_run_unsafe_code=False,
    )
    assert not _stage_matches(
        payload,
        run_id="Q3-8B-Dense",
        task="ifeval",
        checkpoint_manifest_sha256="wrong",
        engine_configuration=configuration,
        limit=None,
        confirm_run_unsafe_code=False,
    )


def test_humaneval_summary_selects_pass_at_one() -> None:
    evaluation = {
        "results": {
            "humaneval": {
                "pass@1,create_test": 0.25,
            }
        }
    }
    assert _summarize_task("humaneval", evaluation) == {"pass_at_1": 0.25}


def test_c1_r96_arm_requires_matching_method_and_rank() -> None:
    assert _validate_arm(
        "Q3-8B-C1-R96",
        {
            "method": "c1-two-sided-kl",
            "equivalent_rank_target": 96,
        },
    )
    assert not _validate_arm(
        "Q3-8B-C1-R96",
        {
            "method": "c1-two-sided-kl",
            "equivalent_rank_target": 80,
        },
    )
