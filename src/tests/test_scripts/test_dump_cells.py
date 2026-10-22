"""The per-cell dump, checked without a GPU.

It runs five minutes into a twelve-hour job and everything downstream joins
its two files by row position, so a silent misalignment would not surface
until the analysis was already wrong. `_dump_cells` takes arrays and a dict,
so it is tested here rather than discovered on the cluster.

The first run of this file found a real defect: validate_latent.py imports
numpy per function rather than at module scope, and `_dump_cells` used `np`
without importing it. That would have been a NameError five minutes into the
job.
"""

from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.validate_latent import _dump_cells            # noqa: E402

N, D = 2000, 16
FAMILIES = ("stromal", "epithelial", "immune", "endothelial")
STUDIES = ["marecko_2024", "duempelmann_2026", "li_2025"]


def corpus(seed: int = 0):
    rng = np.random.default_rng(seed)
    meta = {
        "patient_id": np.array([f"d{i % 40:02d}" for i in range(N)],
                               dtype=object),
        "cell family": np.array([FAMILIES[i % 4] for i in range(N)],
                                dtype=object),
        # None for unlabelled, which most labels are for most studies
        "menstrual phase": np.array(
            [None if i % 7 else "secretory_mid" for i in range(N)],
            dtype=object),
        # float nan, which is how a numeric obs column spells "missing"
        "Endometriosis_stage": np.array(
            [float("nan") if i % 3 else 2.0 for i in range(N)], dtype=object),
        "pct_counts_mt": rng.normal(5, 2, N),
    }
    return (rng.normal(0, 1, (N, D)).astype(np.float32), meta,
            np.array([i % 3 for i in range(N)]),
            np.array([("train", "val", "test")[i % 3] for i in range(N)],
                     dtype=object))


def dump(tmp_path, circular_width: int = 0):
    mu, meta, study_id, split_of = corpus()
    rng = np.random.default_rng(1)
    ang = rng.normal(0, 1, (N, circular_width)).astype(np.float32)
    kap = rng.gamma(2, 1, (N, circular_width))
    _dump_cells(tmp_path, mu, ang, kap, meta, study_id, STUDIES, split_of)
    return mu, meta


def read_meta(tmp_path):
    with gzip.open(tmp_path / "metadata.csv.gz", "rt") as f:
        return list(csv.DictReader(f))


def test_latent_round_trips_unchanged(tmp_path):
    mu, _ = dump(tmp_path)
    back = np.load(tmp_path / "latent.npy")
    assert back.dtype == np.float32
    assert np.array_equal(back, mu)


def test_one_metadata_row_per_cell_in_the_same_order(tmp_path):
    """Everything downstream joins the two files by row position."""
    mu, meta = dump(tmp_path)
    rows = read_meta(tmp_path)
    assert len(rows) == len(mu)
    for i in (0, 1, 123, 999, N - 1):
        assert rows[i]["patient_id"] == meta["patient_id"][i]
        assert rows[i]["cell family"] == meta["cell family"][i]


def test_study_and_split_are_added(tmp_path):
    """Neither is in `meta` -- they are the loader's bookkeeping -- and both
    are wanted for colouring a figure."""
    dump(tmp_path)
    rows = read_meta(tmp_path)
    assert rows[0]["study"] == STUDIES[0]
    assert rows[1]["study"] == STUDIES[1]
    assert rows[0]["split"] == "train"
    assert rows[1]["split"] == "val"
    assert {r["study"] for r in rows} == set(STUDIES)


def test_missing_values_are_blank_not_the_word_nan(tmp_path):
    """A float column's missing values are nan; written literally they would
    read back as the string "nan" and count as a level of their own."""
    dump(tmp_path)
    rows = read_meta(tmp_path)
    assert not any(r["Endometriosis_stage"].strip().lower() == "nan"
                   for r in rows)
    assert not any(r["menstrual phase"].strip().lower() == "none"
                   for r in rows)
    # and the values that ARE present survived. The fixture sets a stage on
    # every third cell (nan if i % 3), so the count is ceil(N / 3).
    assert sum(1 for r in rows if r["Endometriosis_stage"]) == (N + 2) // 3


def test_a_gaussian_only_arm_writes_two_files(tmp_path):
    """Every p1 arm has n_circular 0, so there is no circular half to write."""
    dump(tmp_path, circular_width=0)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "latent.npy", "metadata.csv.gz"]


def test_a_circular_arm_writes_its_half_too(tmp_path):
    dump(tmp_path, circular_width=2)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["kappa.npy", "latent.npy", "latent_circular.npy",
                     "metadata.csv.gz"]
    assert np.load(tmp_path / "latent_circular.npy").shape == (N, 2)


def test_it_creates_its_own_directory(tmp_path):
    nested = tmp_path / "a" / "b"
    mu, _ = dump(nested)
    assert (nested / "latent.npy").is_file()


@pytest.mark.parametrize("column", ["study", "split", "patient_id",
                                    "cell family", "menstrual phase"])
def test_every_colouring_column_is_present(column, tmp_path):
    dump(tmp_path)
    assert column in read_meta(tmp_path)[0]
