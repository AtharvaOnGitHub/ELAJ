"""Config loading utilities.

Entry point: load_experiment(path, root) returns all four config objects from a
single experiment YAML.  Every model component receives its config via these
dataclass instances — no hyperparameter is hardcoded elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import yaml

from endo_model.configs.data_config import DataConfig
from endo_model.configs.experiment_config import ExperimentConfig
from endo_model.configs.model_config import ModelConfig
from endo_model.configs.training_config import TrainingConfig

__all__ = [
    "load_experiment",
    "apply_override",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "ExperimentConfig",
]


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def apply_override(d: dict, dotpath: str, value: Any) -> None:
    """Set a nested key in *d* using a dot-separated path.

    Example: apply_override(d, "group_encoder.n_layers", 3)
    sets d["group_encoder"]["n_layers"] = 3, creating intermediate dicts as needed.
    """
    head, _, tail = dotpath.partition(".")
    if not tail:
        d[head] = value
    else:
        if head not in d or not isinstance(d[head], dict):
            d[head] = {}
        apply_override(d[head], tail, value)


def load_experiment(
    path: str,
    root: str | None = None,
) -> Tuple[ExperimentConfig, ModelConfig, TrainingConfig, DataConfig]:
    """Load an experiment YAML and all referenced sub-configs.

    Args:
        path: Path to the experiment YAML (absolute or relative to *root*).
        root: Project root directory. Defaults to the current working directory.
              Sub-config paths inside the experiment YAML are resolved relative
              to this root.

    Returns:
        (ExperimentConfig, ModelConfig, TrainingConfig, DataConfig)
    """
    root_path = Path(root) if root else Path.cwd()
    exp_path = Path(path) if Path(path).is_absolute() else root_path / path

    exp_dict = _load_yaml(exp_path)
    exp_cfg = ExperimentConfig.from_dict(exp_dict)

    model_dict = _load_yaml(root_path / exp_cfg.model)
    train_dict = _load_yaml(root_path / exp_cfg.training)
    data_dict = _load_yaml(root_path / exp_cfg.data)

    # Overrides use "config.field" dot-paths, e.g. "model.d_gauss: 64"
    for dotpath, value in exp_cfg.overrides.items():
        prefix, _, rest = dotpath.partition(".")
        if prefix == "model":
            apply_override(model_dict, rest, value)
        elif prefix == "training":
            apply_override(train_dict, rest, value)
        elif prefix == "data":
            apply_override(data_dict, rest, value)
        else:
            raise ValueError(
                f"Override key '{dotpath}' must start with 'model.', 'training.', or 'data.'."
            )

    return (
        exp_cfg,
        ModelConfig.from_dict(model_dict),
        TrainingConfig.from_dict(train_dict),
        DataConfig.from_dict(data_dict),
    )
