"""Deployed memory accounting must use the actual TP degree, including uneven groups."""

import json

import pytest

from benchmarks.system.summarize_palu_g4_tp4 import per_token_bytes, read_cell


@pytest.mark.parametrize("tp", [4, 8])
def test_tp_cache_accounting(tp):
    def total(name, deployment):
        keys, values = per_token_bytes(name, deployment, tp)
        return sum(keys + value for value in values)

    dense = total("dense", {})
    basis = total("basis_v96", {})
    replicated = total("palu_g4_r96", {"latent_bytes_per_token": [384 * 2] * 32})
    sharded = total("palu_g4_sharded_r96",
                    {"latent_bytes_per_token": [384 * 2 // (tp // 2)] * 32})
    assert dense * tp == 131072
    assert basis == sharded == dense * .875
    assert replicated == dense * (1.25 if tp == 4 else 2)


def test_read_cell_sums_all_eight_ranks_and_reports_peer_traffic(tmp_path):
    configuration = {"tensor_parallel_size": 8,
                     "hf_overrides": {"basisserve_palu_factor_dir": "/factors/L31-8B-I-PALUG4-R96"}}
    config = dict(configuration=configuration, arm="palu_g4_sharded", graph="eager", experiment="decode",
                  batch=1, prompt_tokens=4096, generation_tokens=128)
    ranks = []
    for index in range(8):
        ranks.append(dict(deployment={
            "latent_bytes_per_token": [192 if index < 4 else 208] * 32,
            "group_traffic_bytes_since_load": dict(decode_steps=32 * 4, decode_weight_bytes=12 * 2**20,
                                                   decode_partial_bytes=3 * 2**20, prefill_shard_bytes=2**20)},
            gpu_cache_layer_bytes=[32 * 1024**2] * 32, gpu_cache_pool_bytes=1024**3,
            peak_allocated_bytes=2 * 1024**3, peak_reserved_bytes=3 * 1024**3,
            nvml_sampled_device_bytes=4 * 1024**3, model_parameter_storage_bytes=1024**3,
            model_buffer_storage_bytes=0))
    starts = [{**r, "deployment": {**r["deployment"], "group_traffic_bytes_since_load":
               dict.fromkeys(r["deployment"]["group_traffic_bytes_since_load"], 0)}} for r in ranks]
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "summary.json").write_text(json.dumps({"trials": [{"status": "complete"}], "medians": {}}))
    (tmp_path / "request_trials.jsonl").write_text('{}\n')
    (tmp_path / "memory_by_rank.jsonl").write_text('\n'.join(json.dumps(record) for record in [
        {"phase": "decode_ready", "ranks": starts}, {"phase": "decode_end", "ranks": ranks}]))
    row = read_cell(tmp_path)
    assert row["tp"] == 8
    assert row["kv_bytes_per_token_rank"] == 14336
    assert row["workload_value_gib_all_ranks"] == 4224 * 32 * (4 * 192 + 4 * 208) / 1024**3
    assert row["group_decode_weight_mib_per_step_per_rank"] == 3
    assert row["group_decode_partial_mib_per_step_per_rank"] == .75
    assert row["group_prefill_shard_mib_total_per_rank"] == 1
