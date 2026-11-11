"""Shared TP4 factor loading and folded V projection for C1 model adapters."""
from collections.abc import Iterable
from pathlib import Path
import torch
from torch import Tensor
from basisserve.core.tp4_c1_factors import HEAD_DIM, NUM_KV_HEADS, TP_SIZE, fold_value_weight, load_layer, load_manifest
from basisserve.vllm.prepared_c1_output import PreparedC1Output
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.utils import AutoWeightsLoader

class C1QKVParallelLinear(QKVParallelLinear):
    def __init__(self, encoder: Tensor, prefix: str, hidden_size: int, num_query_heads: int):
        super().__init__(hidden_size, HEAD_DIM, num_query_heads, NUM_KV_HEADS,
                         bias=False, prefix=prefix, v_head_size=encoder.shape[-1])
        self.register_buffer("encoder", encoder, persistent=False)
        self.value_loaded = False

    def _load_value(self, param, loaded_weight):
        assert param is self.weight and self.num_kv_heads == 2
        local = loaded_weight.narrow(0, self.tp_rank * 2 * HEAD_DIM, 2 * HEAD_DIM).to(self.encoder.device)
        compact = fold_value_weight(local, self.encoder)
        offset = (self.num_heads + self.num_kv_heads) * self.head_size
        param.data.narrow(0, offset, 2 * self.encoder.shape[-1]).copy_(compact)
        self.value_loaded = True

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None):
        assert loaded_shard_id in ("q", "k", "v")
        if loaded_shard_id == "v":
            self._load_value(param, loaded_weight)
        else:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        assert loaded_shard_id in ("q", "k", "v")
        if loaded_shard_id == "v":
            self._load_value(param, loaded_weight)
        else:
            super().weight_loader(param, loaded_weight, loaded_shard_id)


class TP4C1ModelMixin:

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        cfg = vllm_config.model_config.hf_config
        assert (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
                getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads), cfg.num_hidden_layers) == self.model_geometry
        assert not cfg.attention_bias
        group = get_tp_group()
        assert group.world_size == TP_SIZE and get_pp_group().world_size == 1
        assert vllm_config.parallel_config.decode_context_parallel_size == 1
        assert vllm_config.parallel_config.data_parallel_size == 1
        assert vllm_config.quant_config is None and vllm_config.lora_config is None
        assert vllm_config.model_config.dtype == torch.bfloat16
        assert vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
        assert vllm_config.speculative_config is None
        output_mode = getattr(cfg, "basisserve_output_mode", "ag")
        if output_mode != "ag":
            assert vllm_config.parallel_config.worker_cls == 'basisserve.vllm.kcuu_worker.KCuuWorker'
            assert not vllm_config.scheduler_config.async_scheduling
        root = Path(cfg.basisserve_c1_factor_dir).expanduser().resolve()
        manifest = load_manifest(root, cfg.basisserve_c1_result_sha256)
        fit = manifest["fit_config"]
        self.value_rank = fit["cache_rank_per_head"]
        assert (fit["hidden_size"], fit["num_query_heads"], fit["num_hidden_layers"]) == (
            cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.capture_first_logits = False
        self.first_logits = None
        device = next(self.parameters()).device
        self.c1_boundary = PreparedC1Output(
            group.device_group, device, cfg.num_attention_heads // TP_SIZE * self.value_rank,
            vllm_config.scheduler_config.max_num_batched_tokens,
            output_mode=output_mode,
            audit_routes=getattr(cfg, "basisserve_audit_routes", False),
        )
        for index, layer in enumerate(self.model.layers):
            encoder, decoder = load_layer(root, manifest, index, group.rank_in_group,
                                           device=device, dtype=torch.bfloat16)
            layer.self_attn = self.attention_class(layer.self_attn, encoder, decoder,
                                          self.c1_boundary, vllm_config)

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        skips = [f"model.layers.{layer}.self_attn.o_proj." for layer in range(self.config.num_hidden_layers)]
        if self.config.tie_word_embeddings:
            skips.append("lm_head.")
        filtered = ((name, tensor) for name, tensor in weights if not name.startswith(tuple(skips)))
        return AutoWeightsLoader(self).load_weights(filtered)

    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        if self.capture_first_logits and self.first_logits is None and logits is not None:
            self.first_logits = logits.detach().clone()
        return logits

    def c1_statistics(self):
        local_heads = self.model.layers[0].self_attn.num_heads
        return self.c1_boundary.statistics() | {
            "loaded_value_layers": sum(layer.self_attn.qkv_proj.value_loaded for layer in self.model.layers),
            "attention_backend": self.model.layers[0].self_attn.attn.attn_backend.get_name(),
            "attention_impl": type(self.model.layers[0].self_attn.attn.impl).__name__,
            "prefill_specialization": f"sm90_qk128_v{self.value_rank}",
            "decode_specialization": f"sm90_qk128_v{self.value_rank}_h{local_heads}_kv2",
            "decode_output_layout": "feature_major_allgather_slot",
            "key_head_size": HEAD_DIM,
            "value_head_size": self.value_rank,
        }
