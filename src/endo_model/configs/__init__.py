"""Config loading utilities.

Entry point: load_experiment(path, root) returns all four config objects from a
single experiment YAML.  Every model component receives its config via these
dataclass instances — no hyperparameter is hardcoded elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import os

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


def _expand_env(value: Any) -> Any:
    """Expand ${VAR} in every string, so site paths live in site.env, not in configs."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return _expand_env(yaml.safe_load(f) or {})


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


_OVERRIDE_ROOTS = {"model": ModelConfig, "training": TrainingConfig, "data": DataConfig}

# Field names retired in September 2026, still accepted so a YAML written
# against the old names loads and lands on the right field.
_OVERRIDE_ALIASES = {
    "model.n_vMF": "model.n_circular",
    "model.loss.w_kl_vmf": "model.loss.w_kl_circ",
}


def check_override_path(dotpath: str) -> None:
    """Refuse an override whose path does not name a config field.

    Every `from_dict` reads with `d.get(key, default)`, so a misspelt key
    (`model.d_diseas: 16`, `training.batch_sixe: 128`) used to pass the
    prefix check, be written into the dict, and be read by nothing. The run
    then trained the default silently, under a YAML that said otherwise.
    The path is walked against the dataclass fields, nested configs
    included, and an unknown segment is an error naming the known ones.
    """
    import dataclasses
    import typing

    prefix, _, rest = dotpath.partition(".")
    if prefix not in _OVERRIDE_ROOTS:
        raise ValueError(
            f"Override key '{dotpath}' must start with 'model.', 'training.', or 'data.'."
        )
    cls = _OVERRIDE_ROOTS[prefix]
    walked = prefix
    for part in rest.split("."):
        if not dataclasses.is_dataclass(cls):
            raise ValueError(
                f"Override key '{dotpath}': '{walked}' is a plain value and "
                f"has no field '{part}'."
            )
        fields = {f.name: f for f in dataclasses.fields(cls)}
        if part not in fields:
            raise ValueError(
                f"Override key '{dotpath}': '{walked}' has no field '{part}'. "
                f"Known fields: {', '.join(sorted(fields))}."
            )
        hints = typing.get_type_hints(cls)
        cls = hints.get(part, fields[part].type)
        walked = f"{walked}.{part}"


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
        dotpath = _OVERRIDE_ALIASES.get(dotpath, dotpath)
        check_override_path(dotpath)
        prefix, _, rest = dotpath.partition(".")
        target = {"model": model_dict, "training": train_dict, "data": data_dict}[prefix]
        apply_override(target, rest, value)

    return (
        exp_cfg,
        ModelConfig.from_dict(model_dict),
        TrainingConfig.from_dict(train_dict),
        DataConfig.from_dict(data_dict),
    )
