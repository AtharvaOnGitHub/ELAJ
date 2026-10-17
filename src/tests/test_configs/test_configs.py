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
    assert cfg.n_circular == 8
    assert cfg.d_dec == 64
    assert cfg.n_groups is None
    assert cfg.n_studies is None


def test_model_config_derived_d_z():
    cfg = ModelConfig(d_gauss=32, n_circular=8)
    # Each circular dim → (cos θ, sin θ) pair → 2 components
    assert cfg.d_z == 32 + 2 * 8   # 48


def test_model_config_derived_d_gene_rep():
    """Gene identity only — d_agg must NOT appear, that was the bypass."""
    cfg = ModelConfig(d_token=32, d_agg=64)
    assert cfg.d_gene_rep == 32


def test_model_config_d_z_with_different_n_circular():
    cfg = ModelConfig(d_gauss=16, n_circular=4)
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
    """The October 2026 shape: wider, deeper, normalised, no dropout.

    The old default was one 64-unit layer with dropout 0.1, and every arm
    from r1 to m4 beat that classifier without becoming study-invariant.
    """
    cfg = ModelConfig()
    assert cfg.dab.lambda_dab == 1.0
    assert cfg.dab.hidden_dim == 256
    assert cfg.dab.n_layers == 3
    assert cfg.dab.normalise is True
    assert cfg.dab.dropout == 0.0


def test_model_config_dab_from_dict():
    d = {"dab": {"lambda_dab": 0.5, "hidden_dim": 128, "n_layers": 2,
                 "normalise": False, "dropout": 0.1}}
    cfg = ModelConfig.from_dict(d)
    assert cfg.dab.lambda_dab == 0.5
    assert cfg.dab.hidden_dim == 128
    assert cfg.dab.n_layers == 2
    assert cfg.dab.normalise is False
    assert cfg.dab.dropout == 0.1


def test_model_config_ecs_defaults_are_off():
    """ECS must not arrive switched on: it changes the objective.

    `enabled` False and `w_ecs` 0 are two independent guards, and train.py
    requires both before the term reaches the loss.
    """
    cfg = ModelConfig()
    assert cfg.ecs.enabled is False
    assert cfg.loss.w_ecs == 0.0
    assert cfg.ecs.threshold == 0.8
    assert cfg.ecs.buffer_batches == 0


def test_model_config_ecs_from_dict():
    d = {"ecs": {"enabled": True, "threshold": 0.7, "buffer_batches": 8},
         "loss": {"w_ecs": 0.25}}
    cfg = ModelConfig.from_dict(d)
    assert cfg.ecs.enabled is True
    assert cfg.ecs.threshold == 0.7
    assert cfg.ecs.buffer_batches == 8
    assert cfg.loss.w_ecs == 0.25


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


# ── Override paths are checked against the dataclass fields (findings D4) ──

def test_override_paths_resolve_to_real_fields():
    from endo_model.configs import check_override_path

    for ok in ("model.d_gauss", "model.group_encoder.n_layers",
               "model.loss.w_dab", "model.cce.enabled", "model.d_disease",
               "training.kl_annealing.beta_max", "training.diagnostics.log_every",
               "training.optimizer.lr", "data.include_disease_statuses"):
        check_override_path(ok)


def test_misspelt_override_is_refused_and_names_the_known_fields():
    from endo_model.configs import check_override_path

    with pytest.raises(ValueError, match="no field 'd_diseas'"):
        check_override_path("model.d_diseas")
    with pytest.raises(ValueError, match="batch_sixe.*Known fields.*batch_size"):
        check_override_path("training.batch_sixe")
    with pytest.raises(ValueError, match="plain value"):
        check_override_path("model.d_gauss.extra")
    with pytest.raises(ValueError, match="must start with"):
        check_override_path("mdl.d_gauss")


def test_load_experiment_refuses_a_misspelt_override(tmp_path):
    (tmp_path / "m.yaml").write_text("d_gauss: 8\n")
    (tmp_path / "t.yaml").write_text("max_epochs: 1\n")
    (tmp_path / "d.yaml").write_text("vocabulary_path: v.json\n")
    (tmp_path / "e.yaml").write_text(
        "name: x\nmodel: m.yaml\ntraining: t.yaml\ndata: d.yaml\n"
        "overrides:\n  model.d_gauss: 4\n  training.max_epoch: 3\n"
    )
    with pytest.raises(ValueError, match="max_epoch"):
        load_experiment("e.yaml", root=str(tmp_path))


def test_every_experiment_yaml_in_the_repo_loads():
    """The check is strict, so every arm ever written has to pass it."""
    import glob

    root = Path(__file__).resolve().parents[2]
    configs = sorted(glob.glob(str(root / "configs" / "experiment" / "*.yaml")))
    assert configs
    for path in configs:
        load_experiment(path, root=str(root))


# The retired spellings are built by concatenation so a repo-wide rename
# cannot rewrite the test into asserting nothing.
OLD_N = "n_" + "vMF"
OLD_W = "w_kl_" + "vmf"


def test_retired_names_still_load():
    """The circular branch's config keys were renamed in September 2026; old
    snapshots and YAMLs written against the old names must land on the new
    fields."""
    from endo_model.configs.model_config import LossConfig

    assert ModelConfig.from_dict({OLD_N: 3}).n_circular == 3
    assert ModelConfig.from_dict({"n_circular": 2, OLD_N: 3}).n_circular == 2
    assert LossConfig.from_dict({OLD_W: 0.3}).w_kl_circ == 0.3


def test_retired_override_names_are_translated(tmp_path):
    (tmp_path / "m.yaml").write_text("d_gauss: 8\n")
    (tmp_path / "t.yaml").write_text("max_epochs: 1\n")
    (tmp_path / "d.yaml").write_text("vocabulary_path: v.json\n")
    (tmp_path / "e.yaml").write_text(
        "name: x\nmodel: m.yaml\ntraining: t.yaml\ndata: d.yaml\n"
        f"overrides:\n  model.{OLD_N}: 3\n  model.loss.{OLD_W}: 0.2\n"
    )
    _, model_cfg, _, _ = load_experiment("e.yaml", root=str(tmp_path))
    assert model_cfg.n_circular == 3 and model_cfg.loss.w_kl_circ == 0.2


# ── p1: the 2x2 factorial wave ────────────────────────────────────────────────
# The four arms differ in exactly two override keys and agree on every other
# one. That is not a tidiness preference: it is the condition under which the
# square has main effects and an interaction to read at all. A later edit that
# touches one arm's base and not the other three turns the factorial back into
# four unattributable runs, which is the thing LEVERS.md was written about.
# ONE WAVE, FIVE ARMS, AND THEY ALL CARRY THE SAME WAVE NUMBER. The series
# convention here is <letter><wave>_<arm>: r1 had six arms, m2 five, m6 five,
# and the number increments only when a wave is superseded -- m1 -> m2 after
# the reporting defects were fixed, m2 -> m3 after d_study. p2 would mean a
# genuine fault was found in p1 and corrected, not a different configuration.
P1_ARMS = {
    "p1_dab7_kl1": (7.0, 1.0),
    "p1_dab7_kl30": (7.0, 30.0),
    "p1_dab168_kl1": (168.0, 1.0),
    "p1_dab336_kl30": (336.0, 30.0),
    # The cell m5_combined would have filled, run rather than assumed: m5
    # carried a study_knn checkpoint guard this wave does not have.
    "p1_dab168_kl30": (168.0, 30.0),
}
P1_KNOBS = ("model.dab.lambda_dab", "model.loss.w_kl_gauss")


def _p1_overrides():
    import yaml

    root = Path(__file__).resolve().parents[2]
    out = {}
    for name in P1_ARMS:
        path = root / "configs" / "experiment" / f"{name}.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["name"] == name, f"{path} names itself {raw['name']!r}"
        out[name] = raw
    return out


def test_p1_knobs_take_the_four_intended_values():
    """m5_combined fills the square's fourth cell at (168, 30), so these four
    must be the other three plus the 336 extension -- no duplicates, no
    accidental third value on either axis."""
    raws = _p1_overrides()
    got = {
        name: tuple(float(raw["overrides"][k]) for k in P1_KNOBS)
        for name, raw in raws.items()
    }
    assert got == P1_ARMS, got


def test_p1_shares_one_base_outside_the_two_knobs():
    raws = _p1_overrides()
    bases = {
        name: {k: v for k, v in raw["overrides"].items()
               if k not in P1_KNOBS}
        for name, raw in raws.items()
    }
    reference, expected = next(iter(bases.items()))
    for name, base in bases.items():
        differing = {
            k for k in set(base) | set(expected)
            if base.get(k, "<absent>") != expected.get(k, "<absent>")
        }
        assert not differing, (
            f"{name} differs from {reference} outside the two knobs on "
            f"{sorted(differing)}; the 2x2 cannot be attributed."
        )


def test_p1_selects_checkpoints_on_nb_alone():
    """No guard, deliberately: `study_knn` is one of the two axes the frontier
    is plotted on, so guarding on it would select the points by their position
    on an axis we then report as a result. It is also arm-dependent in effect,
    which is what stopped m5 at nine of twelve epochs."""
    for name, raw in _p1_overrides().items():
        over = raw["overrides"]
        assert over.get("training.stop_metric") == "nb", name
        assert "training.guard_metric" not in over, name
        assert "training.guard_max" not in over, name


def test_p1_shares_one_horizon_and_one_seed():
    """`max_epochs` sets T_max on CosineAnnealingLR and is the denominator of
    both the KL warm-up and the DAB ramp, so it is an optimisation variable
    and not just a budget. Its VALUE is a decision recorded in
    RETRAIN_EXPECTATIONS; what this asserts is that the arms agree on it,
    that none of them falls through to base.yaml's 50, and that the seed is
    not a fifth variable. (A divergence would also trip the base-identity
    test above; this one names the cause.)"""
    raws = _p1_overrides()
    horizons = {name: raw["overrides"].get("training.max_epochs")
                for name, raw in raws.items()}
    assert all(h is not None for h in horizons.values()), horizons
    assert len(set(horizons.values())) == 1, horizons
    assert {int(raw["seed"]) for raw in raws.values()} == {24}
