"""Attention geometry recorded by calibration, independent of model size."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ValueGeometry:
    model_type: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    attention_layers: tuple[int, ...]

    def __post_init__(self) -> None:
        assert (
            min(
                self.num_layers,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.hidden_size,
            )
            > 0
        )
        assert self.num_heads % self.num_kv_heads == 0
        assert self.attention_layers
        assert self.attention_layers == tuple(sorted(set(self.attention_layers)))
        assert (
            0 <= self.attention_layers[0] <= self.attention_layers[-1] < self.num_layers
        )

    @property
    def attention_type(self) -> str:
        return "mha" if self.num_heads == self.num_kv_heads else "gqa"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "attention_layers": list(self.attention_layers)}

    @classmethod
    def from_snapshot(cls, directory: str | Path) -> "ValueGeometry":
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        model = manifest["model"]
        geometry = cls(
            model_type=model["model_type"],
            num_layers=int(model["num_hidden_layers"]),
            num_heads=int(model["num_attention_heads"]),
            num_kv_heads=int(model["num_key_value_heads"]),
            head_dim=int(model["head_dim"]),
            hidden_size=int(model["hidden_size"]),
            attention_layers=tuple(map(int, manifest["layers"])),
        )
        assert model["attention_type"] == geometry.attention_type
        return geometry

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValueGeometry":
        return cls(**{**value, "attention_layers": tuple(value["attention_layers"])})

    @classmethod
    def from_factors(cls, directory: str | Path) -> "ValueGeometry":
        records = sorted(Path(directory).glob("layer_*.json"))
        assert records, f"No fitted layer records in {directory}"
        record = json.loads(records[0].read_text())
        return cls.from_dict(record["fit_config"]["geometry"])
