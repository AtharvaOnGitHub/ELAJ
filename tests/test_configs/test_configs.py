"""Tests for config dataclasses and the YAML loader."""

import csv
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from endo_model.configs import apply_override, load_experiment
from endo_model.configs.data_config import (
    DataConfig,
    SplitConfig,
    StudyConfig,
    load_studies_csv,
)
from endo_model.configs.model_config import (
    AggregationEncoderConfig,
    CCEConfig,
    DABConfig,
    DecoderConfig,
    GroupEncoderConfig,
    ModelConfig,
)
from endo_model.configs.training_config import (
    KLAnnealingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)

CONFIGS_DIR = Path(__file__).parents[2] / "configs"
PROJECT_ROOT = CONFIGS_DIR.parent


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.name == "ELAJ_small"
    assert cfg.d_token == 32
    assert cfg.d_agg == 64
    assert cfg.d_gauss == 32
    assert cfg.n_vMF == 8
    assert cfg.d_dec == 64
    assert cfg.n_groups is None
    assert cfg.n_studies is None


def test_model_config_derived_d_z():
    cfg = ModelConfig(d_gauss=32, n_vMF=8)
    # Each vMF dim → (cos θ, sin θ) pair → 2 components
    assert cfg.d_z == 32 + 2 * 8   # 48


def test_model_config_derived_d_gene_rep():
    cfg = ModelConfig(d_token=32, d_agg=64)
    assert cfg.d_gene_rep == 32 + 64   # 96


def test_model_config_d_z_with_different_n_vmf():
    cfg = ModelConfig(d_gauss=16, n_vMF=4)
    assert cfg.d_z == 16 + 2 * 4   # 24  (not 16 + 4 = 20)


def test_model_config_from_dict_partial():
    d = {"d_token": 64, "d_agg": 128, "group_encoder": {"n_layers": 3}}
    cfg = ModelConfig.from_dict(d)
    assert cfg.d_token == 64
    assert cfg.d_agg == 128
    assert cfg.group_encoder.n_layers == 3
    assert cfg.group_encoder.n_heads == 4         # default preserved
    assert cfg.aggregation_encoder.n_layers == 2  # default preserved


def test_model_config_cce_defaults():
    cfg = ModelConfig()
    assert cfg.cce.enabled is False
    assert cfg.cce.temperature == 0.07
    assert cfg.cce.lambda_cce == 0.1


def test_model_config_cce_from_dict():
    d = {"cce": {"enabled": True, "temperature": 0.1, "lambda_cce": 0.5}}
    cfg = ModelConfig.from_dict(d)
    assert cfg.cce.enabled is True
    assert cfg.cce.temperature == 0.1
    assert cfg.cce.lambda_cce == 0.5


def test_model_config_dab_defaults():
    cfg = ModelConfig()
    assert cfg.dab.lambda_dab == 1.0
    assert cfg.dab.hidden_dim == 64


def test_model_config_dab_from_dict():
    d = {"dab": {"lambda_dab": 0.5, "hidden_dim": 128}}
    cfg = ModelConfig.from_dict(d)
    assert cfg.dab.lambda_dab == 0.5
    assert cfg.dab.hidden_dim == 128


def test_group_encoder_config_from_dict_empty():
    cfg = GroupEncoderConfig.from_dict({})
    assert cfg.n_layers == 2
    assert cfg.n_heads == 4
    assert cfg.dropout == 0.1


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.batch_size == 256
    assert cfg.max_epochs == 100
    assert cfg.grad_clip_norm == 1.0
    assert cfg.resume_from_checkpoint is None


def test_training_config_kl_annealing_defaults():
    cfg = TrainingConfig()
    assert cfg.kl_annealing.beta_max == 1.0
    assert cfg.kl_annealing.warmup_fraction == 0.20


def test_training_config_from_dict():
    d = {
        "batch_size": 64,
        "max_epochs": 50,
        "kl_annealing": {"beta_max": 0.5, "warmup_fraction": 0.1},
        "resume_from_checkpoint": "/outputs/epoch_10.pt",
    }
    cfg = TrainingConfig.from_dict(d)
    assert cfg.batch_size == 64
    assert cfg.max_epochs == 50
    assert cfg.kl_annealing.beta_max == 0.5
    assert cfg.kl_annealing.warmup_fraction == 0.1
    assert cfg.resume_from_checkpoint == "/outputs/epoch_10.pt"


def test_training_config_no_cce_field():
    # CCE config now lives in ModelConfig — TrainingConfig should not have it
    cfg = TrainingConfig()
    assert not hasattr(cfg, "objectives")
    assert not hasattr(cfg, "cce_enabled")


def test_optimizer_betas_coerced_to_tuple():
    d = {"betas": [0.8, 0.95]}
    cfg = OptimizerConfig.from_dict(d)
    assert isinstance(cfg.betas, tuple)
    assert cfg.betas == (0.8, 0.95)


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


def test_split_config_default_seed():
    cfg = SplitConfig()
    assert cfg.seed == 24


def test_data_config_from_dict_inline():
    d = {
        "vocabulary_path": "data/vocab.json",
        "studies": [
            {
                "id": 0, "name": "study_a", "path": "data/a.h5ad",
                "tissue_types": ["eutopic"], "has_cycle_phase": False,
            }
        ],
        "splits": {"val_fraction": 0.2},
    }
    cfg = DataConfig.from_dict(d)
    assert len(cfg.studies) == 1
    assert cfg.studies[0].name == "study_a"
    assert cfg.splits.val_fraction == 0.2
    assert cfg.splits.test_fraction == 0.15   # default preserved
    assert cfg.studies_csv is None


def test_data_config_studies_csv_field():
    cfg = DataConfig.from_dict({"studies_csv": "data/processed/studies.csv"})
    assert cfg.studies_csv == "data/processed/studies.csv"
    assert cfg.studies == []   # inline list is empty; CSV is the source


def test_data_config_resolve_studies_inline(tmp_path):
    cfg = DataConfig.from_dict({
        "studies": [{"id": 0, "name": "s0", "path": "x.h5ad",
                     "tissue_types": ["eutopic"], "has_cycle_phase": False}]
    })
    studies = cfg.resolve_studies(root=str(tmp_path))
    assert len(studies) == 1
    assert studies[0].id == 0


def test_data_config_resolve_studies_from_csv(tmp_path):
    csv_path = tmp_path / "studies.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "name", "path", "tissue_types", "has_cycle_phase"]
        )
        writer.writeheader()
        writer.writerow({
            "id": 0, "name": "study_a", "path": "data/a.h5ad",
            "tissue_types": "eutopic|ectopic", "has_cycle_phase": "false",
        })
        writer.writerow({
            "id": 1, "name": "study_b", "path": "data/b.h5ad",
            "tissue_types": "control", "has_cycle_phase": "true",
        })

    cfg = DataConfig.from_dict({"studies_csv": "studies.csv"})
    studies = cfg.resolve_studies(root=str(tmp_path))
    assert len(studies) == 2
    assert studies[0].id == 0
    assert studies[1].id == 1
    assert studies[0].tissue_types == ["eutopic", "ectopic"]
    assert studies[1].has_cycle_phase is True


def test_load_studies_csv_missing_file():
    with pytest.raises(FileNotFoundError, match="Studies CSV not found"):
        load_studies_csv("/nonexistent/path/studies.csv")


def test_load_studies_csv_missing_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name"])
        writer.writeheader()
        writer.writerow({"id": 0, "name": "s0"})

    with pytest.raises(ValueError, match="missing required columns"):
        load_studies_csv(str(csv_path))


def test_load_studies_csv_sorted_by_id(tmp_path):
    csv_path = tmp_path / "studies.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "name", "path", "tissue_types", "has_cycle_phase"]
        )
        writer.writeheader()
        # Write in reverse order
        for i in [2, 0, 1]:
            writer.writerow({
                "id": i, "name": f"s{i}", "path": f"data/{i}.h5ad",
                "tissue_types": "eutopic", "has_cycle_phase": "false",
            })

    studies = load_studies_csv(str(csv_path))
    assert [s.id for s in studies] == [0, 1, 2]


def test_data_config_max_cells_none():
    cfg = DataConfig.from_dict({"max_cells_per_study": None})
    assert cfg.max_cells_per_study is None


def test_data_config_max_cells_integer():
    cfg = DataConfig.from_dict({"max_cells_per_study": 1000})
    assert cfg.max_cells_per_study == 1000


# ---------------------------------------------------------------------------
# apply_override
# ---------------------------------------------------------------------------


def test_apply_override_shallow():
    d = {"a": 1, "b": 2}
    apply_override(d, "a", 99)
    assert d["a"] == 99
    assert d["b"] == 2


def test_apply_override_nested():
    d = {"group_encoder": {"n_layers": 2, "dropout": 0.1}}
    apply_override(d, "group_encoder.n_layers", 4)
    assert d["group_encoder"]["n_layers"] == 4
    assert d["group_encoder"]["dropout"] == 0.1


def test_apply_override_creates_intermediate_dicts():
    d: dict = {}
    apply_override(d, "a.b.c", 42)
    assert d["a"]["b"]["c"] == 42


def test_apply_override_nested_cce():
    d: dict = {}
    apply_override(d, "cce.enabled", True)
    assert d["cce"]["enabled"] is True


# ---------------------------------------------------------------------------
# Full YAML round-trips via load_experiment
# ---------------------------------------------------------------------------


def test_load_full_model():
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(
        str(CONFIGS_DIR / "experiment" / "full_model.yaml"),
        root=str(PROJECT_ROOT),
    )
    assert exp_cfg.name == "full_model"
    assert model_cfg.name == "ELAJ_small"
    assert model_cfg.d_token == 32
    assert model_cfg.cce.enabled is False
    assert train_cfg.batch_size == 256
    assert train_cfg.resume_from_checkpoint is None
    assert exp_cfg.seed == 24


def test_load_full_model_cce_override():
    """Toggle CCE on via experiment override without touching model YAML."""
    override_exp = {
        "name": "cce_test",
        "model": "configs/model/ELAJ_small.yaml",
        "training": "configs/training/base.yaml",
        "data": "configs/data/corpus.yaml",
        "output_dir": "outputs/test/",
        "seed": 24,
        "overrides": {"model.cce.enabled": True},
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(override_exp, f)
        tmp_path = f.name
    try:
        _, model_cfg, _, _ = load_experiment(tmp_path, root=str(PROJECT_ROOT))
        assert model_cfg.cce.enabled is True
        assert model_cfg.cce.temperature == 0.07  # other CCE params unchanged
    finally:
        os.unlink(tmp_path)


def test_load_experiment_resume_from_checkpoint():
    override_exp = {
        "name": "resume_test",
        "model": "configs/model/ELAJ_small.yaml",
        "training": "configs/training/base.yaml",
        "data": "configs/data/corpus.yaml",
        "output_dir": "outputs/test/",
        "seed": 24,
        "overrides": {"training.resume_from_checkpoint": "outputs/epoch_5.pt"},
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(override_exp, f)
        tmp_path = f.name
    try:
        _, _, train_cfg, _ = load_experiment(tmp_path, root=str(PROJECT_ROOT))
        assert train_cfg.resume_from_checkpoint == "outputs/epoch_5.pt"
    finally:
        os.unlink(tmp_path)


def test_load_experiment_bad_override_prefix():
    bad_exp = {
        "name": "bad",
        "model": "configs/model/ELAJ_small.yaml",
        "training": "configs/training/base.yaml",
        "data": "configs/data/corpus.yaml",
        "overrides": {"unknown.field": 1},
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(bad_exp, f)
        tmp_path = f.name
    try:
        with pytest.raises(ValueError, match="must start with"):
            load_experiment(tmp_path, root=str(PROJECT_ROOT))
    finally:
        os.unlink(tmp_path)


def test_default_seed_is_24():
    """Seed watermark: default seed must be 24 in all configs that have one."""
    from endo_model.configs.data_config import SplitConfig
    from endo_model.configs.experiment_config import ExperimentConfig
    assert SplitConfig().seed == 24
    assert ExperimentConfig().seed == 24
