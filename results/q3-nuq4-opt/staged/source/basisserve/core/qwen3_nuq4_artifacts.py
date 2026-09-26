"""Structure-only serving inputs from the frozen Qwen3-8B-Base PPL run."""

import json
from pathlib import Path

from safetensors.torch import load_file
import torch


class QwenNUQ4Artifacts:
    def __init__(self, prior, rank):
        assert rank in (64, 96)
        self.prior, self.rank = Path(prior).resolve(), rank
        self.protocol = json.loads((self.prior / "manifest.json").read_text())["protocol"]
        self.root = Path(self.protocol["checkpoint_root"]) / f"Q3-8B-C1-R{rank}"
        self.model = Path(self.protocol["model"])
        config = json.loads((self.model / "config.json").read_text())
        assert (config["model_type"], config["hidden_size"], config["num_hidden_layers"],
                config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]) == (
                    "qwen3", 4096, 36, 32, 8, 128)
        manifest = json.loads((self.root / "manifest.json").read_text())
        allocation = json.loads((self.root / "result.json").read_text())
        assert manifest["status"] == allocation["status"] == "complete"
        assert manifest["model"]["huggingface_repo"] == "Qwen/Qwen3-8B-Base"
        assert manifest["compression"]["equivalent_rank_target"] == rank
        self.schedule = allocation["selection"]["selected_schedule"]
        assert self.schedule == manifest["compression"]["layer_ranks"]
        assert len(self.schedule) == 36
        assert all(len(row) == 8 and all(16 <= r <= 128 and r % 16 == 0 for r in row)
                   for row in self.schedule)
        assert sum(map(sum, self.schedule)) == 36 * 8 * rank
        self.artifacts = allocation["selected_artifacts"]
        assert set(self.artifacts) == set(map(str, range(36)))
        self.codes = torch.load(self.prior / f"r{rank}/quantizers.pt", map_location="cpu", weights_only=False)
        assert set(self.codes) == {f"{i}.{kind}" for i in range(36) for kind in ("k", "v")}
        self.scales = json.loads((self.prior / f"r{rank}/kv4_fp8_scales.json").read_text())
        assert all(self.scales[f"{i}.decoder"]["input_scale"] > 0 for i in range(36))

    def layer(self, index, tp_rank, device):
        assert 0 <= index < 36 and 0 <= tp_rank < 8
        ranks = self.schedule[index]
        relative = Path(self.artifacts[str(index)]["file"])
        assert not relative.is_absolute() and ".." not in relative.parts
        factors = load_file(str(self.root / relative))
        assert set(factors) == {"value_coordinate_encoders", "head_output_decoders", "source_ranks"}
        assert factors["source_ranks"].tolist() == ranks
        enc, dec = factors["value_coordinate_encoders"], factors["head_output_decoders"]
        assert enc.shape == (8, 128, max(ranks)) and dec.shape == (32, max(ranks), 4096)
        assert torch.isfinite(enc).all() and torch.isfinite(dec).all()
        hi, lo, klut = self.codes[f"{index}.k"]
        assert hi.numel() == lo.numel() == 1024
        _, _, vlut = self.codes[f"{index}.v"]
        lower = lo.flatten()[tp_rank*128:(tp_rank+1)*128].to(device=device, dtype=torch.float32).contiguous()
        upper = hi.flatten()[tp_rank*128:(tp_rank+1)*128].to(device=device, dtype=torch.float32).contiguous()
        kcode = torch.as_tensor(klut[0], device=device, dtype=torch.float32).flatten().contiguous()
        vcode = torch.as_tensor(vlut[0], device=device, dtype=torch.float32).flatten().contiguous()
        assert kcode.shape == vcode.shape == (16,)
        assert torch.isfinite(lower).all() and torch.isfinite(upper).all() and (upper >= lower).all()
        assert torch.isfinite(kcode).all() and torch.isfinite(vcode).all()
        decoder = torch.cat([dec[q, :ranks[q//4]] for q in range(32)], dim=0)
        decoder = decoder.to(device=device, dtype=torch.bfloat16).contiguous()
        return dict(value_ranks=tuple(ranks), latent_widths=tuple(4*r for r in ranks),
            encoder=enc[tp_rank, :, :ranks[tp_rank]].to(device=device, dtype=torch.float32).contiguous(),
            decoder=decoder, k_lower=lower, k_upper=upper, k_lut=kcode, v_lut=vcode,
            a8_scale=torch.tensor(self.scales[f"{index}.decoder"]["input_scale"], device=device, dtype=torch.float32))
