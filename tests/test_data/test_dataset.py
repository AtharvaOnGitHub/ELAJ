"""Tests for dataset.py — EndometriosisDataset with synthetic .h5ad fixtures.

These tests require both torch and anndata.  They are skipped gracefully
if either is not installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping dataset tests")
anndata = pytest.importorskip("anndata", reason="anndata not installed; skipping dataset tests")
scipy_sparse = pytest.importorskip("scipy.sparse", reason="scipy not installed; skipping dataset tests")

from endo_model.data.collate import make_collate_fn
from endo_model.data.constants import DNS_SENTINEL
from endo_model.data.dataset import EndometriosisDataset
from endo_model.data.samplers import PerStudyBatchSampler
from endo_model.data.vocabulary import GeneVocabulary, generate_synthetic_vocabulary


# ---------------------------------------------------------------------------
# Synthetic AnnData factory
# ---------------------------------------------------------------------------


def make_synthetic_h5ad(
    tmp_path,
    filename: str,
    gene_names: list[str],
    n_cells: int,
    n_patients: int,
    study_id: int,
    seed: int = 24,
) -> str:
    """Write a synthetic .h5ad and return the path."""
    rng = np.random.default_rng(seed)

    # Counts: integer matrix (cells × genes)
    x = rng.negative_binomial(n=5, p=0.5, size=(n_cells, len(gene_names))).astype(np.float32)
    # Make 30% of entries zero (sparse-like)
    zero_mask = rng.random(x.shape) < 0.3
    x[zero_mask] = 0.0

    obs_data = {
        "patient_id": [f"P{study_id}_{i % n_patients:03d}" for i in range(n_cells)],
        "disease_status": rng.choice(
            ["eutopic", "ectopic", "control"], size=n_cells
        ).tolist(),
        "age": rng.uniform(20, 60, size=n_cells).tolist(),
        "tissue_compartment": rng.integers(0, 3, size=n_cells).tolist(),
        "tissue_organ": rng.integers(0, 5, size=n_cells).tolist(),
        "tissue_type": rng.integers(0, 6, size=n_cells).tolist(),
        "tissue_microsite": rng.integers(0, 4, size=n_cells).tolist(),
    }

    import pandas as pd
    adata = anndata.AnnData(
        X=scipy_sparse.csr_matrix(x),
        obs=pd.DataFrame(obs_data),
        var=pd.DataFrame(index=gene_names),
    )

    path = str(tmp_path / filename)
    adata.write_h5ad(path)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vocab_and_raw():
    raw = generate_synthetic_vocabulary(n_genes=100, n_groups=5, seed=24)
    return GeneVocabulary.from_dict(raw), raw


@pytest.fixture()
def two_study_setup(tmp_path, vocab_and_raw):
    vocab, raw = vocab_and_raw
    all_genes = list(raw.keys())

    # Study 0: all 100 genes
    path0 = make_synthetic_h5ad(
        tmp_path, "study0.h5ad",
        gene_names=all_genes,
        n_cells=60, n_patients=6, study_id=0, seed=0,
    )

    # Study 1: only first 60 genes (40 are DNS)
    path1 = make_synthetic_h5ad(
        tmp_path, "study1.h5ad",
        gene_names=all_genes[:60],
        n_cells=40, n_patients=5, study_id=1, seed=1,
    )

    study_configs = [
        {"id": 0, "name": "study0", "path": path0},
        {"id": 1, "name": "study1", "path": path1},
    ]
    return vocab, study_configs


# ---------------------------------------------------------------------------
# Basic dataset construction
# ---------------------------------------------------------------------------


def test_dataset_train_nonempty(two_study_setup):
    vocab, study_configs = two_study_setup
    ds = EndometriosisDataset(study_configs, vocab, split="train", seed=24)
    assert len(ds) > 0


def test_dataset_splits_cover_all_cells(two_study_setup):
    vocab, study_configs = two_study_setup
    n_train = len(EndometriosisDataset(study_configs, vocab, split="train", seed=24))
    n_val = len(EndometriosisDataset(study_configs, vocab, split="val", seed=24))
    n_test = len(EndometriosisDataset(study_configs, vocab, split="test", seed=24))
    assert n_train + n_val + n_test == 60 + 40


def test_dataset_splits_nonempty(two_study_setup):
    vocab, study_configs = two_study_setup
    for split in ("train", "val", "test"):
        ds = EndometriosisDataset(study_configs, vocab, split=split, seed=24)
        assert len(ds) > 0, f"split '{split}' is empty"


def test_invalid_split_raises(two_study_setup):
    vocab, study_configs = two_study_setup
    with pytest.raises(ValueError, match="split must be"):
        EndometriosisDataset(study_configs, vocab, split="predict", seed=24)


# ---------------------------------------------------------------------------
# __getitem__ structure
# ---------------------------------------------------------------------------


def test_getitem_keys(two_study_setup):
    vocab, study_configs = two_study_setup
    ds = EndometriosisDataset(study_configs, vocab, split="train", seed=24)
    item = ds[0]
    assert set(item.keys()) == {"counts", "study_id", "tissue_levels", "age", "disease_status", "patient_id"}


def test_getitem_counts_shape(two_study_setup):
    vocab, study_configs = two_study_setup
    ds = EndometriosisDataset(study_configs, vocab, split="train", seed=24)
    item = ds[0]
    assert item["counts"].shape == (vocab.vocab_size,)


def test_getitem_counts_dtype(two_study_setup):
    vocab, study_configs = two_study_setup
    ds = EndometriosisDataset(study_configs, vocab, split="train", seed=24)
    assert ds[0]["counts"].dtype == np.float32


def test_getitem_tissue_levels_shape(two_study_setup):
    vocab, study_configs = two_study_setup
    ds = EndometriosisDataset(study_configs, vocab, split="train", seed=24)
    assert ds[0]["tissue_levels"].shape == (4,)


# ---------------------------------------------------------------------------
# DNS logic
# ---------------------------------------------------------------------------


def test_study1_has_dns_for_missing_genes(tmp_path, vocab_and_raw):
    """Study 1 only has 60 genes — the other 40 must be DNS in every cell."""
    vocab, raw = vocab_and_raw
    all_genes = list(raw.keys())
    missing_genes = all_genes[60:]

    path = make_synthetic_h5ad(
        tmp_path, "partial.h5ad",
        gene_names=all_genes[:60],
        n_cells=20, n_patients=4, study_id=0, seed=99,
    )
    ds = EndometriosisDataset(
        [{"id": 0, "name": "partial", "path": path}],
        vocab, split="train",
        val_fraction=0.0, test_fraction=0.0, seed=24,
    )

    missing_global_idxs = [vocab.gene_to_index(g) for g in missing_genes]
    for i in range(len(ds)):
        counts = ds[i]["counts"]
        assert all(
            counts[idx] == float(DNS_SENTINEL) for idx in missing_global_idxs
        ), f"Cell {i} has non-DNS value for a missing gene"


def test_study0_no_dns(tmp_path, vocab_and_raw):
    """Study 0 has all vocabulary genes — no DNS expected."""
    vocab, raw = vocab_and_raw
    all_genes = list(raw.keys())

    path = make_synthetic_h5ad(
        tmp_path, "full.h5ad",
        gene_names=all_genes,
        n_cells=20, n_patients=4, study_id=0, seed=42,
    )
    ds = EndometriosisDataset(
        [{"id": 0, "name": "full", "path": path}],
        vocab, split="train",
        val_fraction=0.0, test_fraction=0.0, seed=24,
    )

    for i in range(len(ds)):
        counts = ds[i]["counts"]
        assert float(DNS_SENTINEL) not in counts, f"Cell {i} has unexpected DNS"


# ---------------------------------------------------------------------------
# study_ids property (for PerStudyBatchSampler)
# ---------------------------------------------------------------------------


def test_study_ids_property(two_study_setup):
    vocab, study_configs = two_study_setup
    ds = EndometriosisDataset(study_configs, vocab, split="train", seed=24)
    sids = ds.study_ids
    assert len(sids) == len(ds)
    assert all(isinstance(s, int) for s in sids)
    assert set(sids).issubset({0, 1})


# ---------------------------------------------------------------------------
# Integration: Dataset → Sampler → Collate
# ---------------------------------------------------------------------------


def test_dataloader_integration(two_study_setup):
    """Full pipeline: dataset → sampler → collate → batch."""
    from torch.utils.data import DataLoader

    vocab, study_configs = two_study_setup
    ds = EndometriosisDataset(study_configs, vocab, split="train", seed=24)
    sampler = PerStudyBatchSampler(ds.study_ids, batch_size=8, drop_last=True)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate_fn(vocab))

    batch = next(iter(loader))
    assert batch["counts"].shape[0] == 8
    assert batch["counts"].shape[1] == vocab.vocab_size

    # DSBN invariant: all cells in batch from same study
    assert batch["study_id"].unique().numel() == 1


# ---------------------------------------------------------------------------
# Missing obs column
# ---------------------------------------------------------------------------


def test_missing_patient_id_raises(tmp_path, vocab_and_raw):
    vocab, raw = vocab_and_raw
    all_genes = list(raw.keys())

    import pandas as pd
    x = np.zeros((5, len(all_genes)), dtype=np.float32)
    adata = anndata.AnnData(
        X=scipy_sparse.csr_matrix(x),
        obs=pd.DataFrame({"disease_status": ["eutopic"] * 5}),
        var=pd.DataFrame(index=all_genes),
    )
    path = str(tmp_path / "no_patient.h5ad")
    adata.write_h5ad(path)

    with pytest.raises(KeyError, match="patient_id"):
        EndometriosisDataset(
            [{"id": 0, "name": "no_pid", "path": path}],
            vocab, split="train",
        )


# ---------------------------------------------------------------------------
# max_cells_per_study
# ---------------------------------------------------------------------------


def test_max_cells_per_study(tmp_path, vocab_and_raw):
    vocab, raw = vocab_and_raw
    all_genes = list(raw.keys())

    path = make_synthetic_h5ad(
        tmp_path, "big_study.h5ad",
        gene_names=all_genes,
        n_cells=80, n_patients=8, study_id=0, seed=7,
    )
    ds = EndometriosisDataset(
        [{"id": 0, "name": "big", "path": path}],
        vocab, split="train",
        val_fraction=0.0, test_fraction=0.0,
        max_cells_per_study=20, seed=24,
    )
    assert len(ds) <= 20
