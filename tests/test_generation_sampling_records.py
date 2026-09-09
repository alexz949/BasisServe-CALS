from types import SimpleNamespace
from evaluation.eval_qwen3_8b_r80_generation_vllm import LocalVLLMGenerationLM


class Tokenizer:
    eos_token_id = 2
    name_or_path = "test"
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def decode(self, token):
        return "<eos>"


class Engine:
    def generate(self, prompts, sampling_params, use_tqdm):
        self.parameters = sampling_params
        return [SimpleNamespace(outputs=[SimpleNamespace(
            text="answer", token_ids=[1, 3], finish_reason="stop", stop_reason=2,
        )]) for p in prompts]


def test_three_reproducible_distinct_seeds_and_records():
    request = SimpleNamespace(task_name="mbpp_plus_pass3", doc_id=0, args=(
        "prompt", {"until": ["[DONE]"], "do_sample": True, "temperature": 0.2, "top_p": 0.95},
    ))
    sequences = []
    for _ in range(2):
        engine = Engine()
        adapter = LocalVLLMGenerationLM(engine, Tokenizer(), max_length=8192, default_max_gen_toks=2048)
        assert adapter.generate_until([request] * 3, disable_tqdm=True) == ["answer"] * 3
        seeds = [p.seed for p in engine.parameters]
        assert len(set(seeds)) == 3
        assert [r["sample_index"] for r in adapter.generation_records] == [0, 1, 2]
        assert all(r["finish_reason"] == "stop" for r in adapter.generation_records)
        sequences.append(seeds)
    assert sequences[0] == sequences[1]
