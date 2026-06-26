from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    name: str = ""
    description: str = ""
    model: str = "configs/model/small.yaml"
    training: str = "configs/training/base.yaml"
    data: str = "configs/data/corpus.yaml"
    output_dir: str = "outputs/"
    seed: int = 24
    overrides: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            model=str(d.get("model", "configs/model/small.yaml")),
            training=str(d.get("training", "configs/training/base.yaml")),
            data=str(d.get("data", "configs/data/corpus.yaml")),
            output_dir=str(d.get("output_dir", "outputs/")),
            seed=int(d.get("seed", 24)),
            overrides=dict(d.get("overrides") or {}),
        )
