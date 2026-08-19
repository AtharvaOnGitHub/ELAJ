"""The platform lookup must cover the corpus and never renumber silently."""

import csv
from pathlib import Path

import pytest

from endo_model.data.study_platform import CODE_COLUMNS, UNKNOWN, StudyPlatform

CSV = Path(__file__).resolve().parents[2] / "configs/data/study_platform.csv"
STUDIES_CSV = Path(__file__).resolve().parents[3] / (
    "data_collation/data/processed/studies.csv"
)


@pytest.fixture(scope="module")
def platform() -> StudyPlatform:
    return StudyPlatform(CSV)


def test_every_configured_study_has_a_row(platform):
    """A study in studies.csv with no platform row would train with its
    technical context silently absent."""
    if not STUDIES_CSV.exists():
        pytest.skip("studies.csv not present in this checkout")
    names = [r["name"] for r in csv.DictReader(STUDIES_CSV.open())]
    missing = [n for n in names if platform.record_for(n) is None]
    assert not missing, f"no platform row for: {missing}"


def test_an_unlisted_study_reads_as_unknown_rather_than_raising(platform):
    assert platform.codes_for("not_a_study") == (UNKNOWN,) * len(CODE_COLUMNS)


def test_codes_are_non_negative_and_sizes_admit_them(platform):
    sizes = platform.sizes()
    for record in (platform.record_for(s) for s in ("shih_2022", "li_2025")):
        for code, size in zip(record.codes, sizes):
            assert 0 <= code < size


def test_zero_is_reserved_for_unknown(platform):
    """garcia_alonso has no confirmed platform, so it must read as unknown —
    not as whichever category happened to be numbered zero."""
    record = platform.record_for("garcia_alonso_2021_a")
    assert record is not None
    assert not record.is_known


def test_the_two_studies_that_differ_from_every_other_are_encoded_differently():
    """li_2025 is microwell, not droplet; duempelmann_2026 is 5', not 3'.

    If either collapsed onto the common droplet-3' code the covariate would be
    worse than useless — it would assert that the most technically distinct
    studies in the corpus match the rest.
    """
    p = StudyPlatform(CSV)
    common = p.record_for("tan_2022")
    assert p.record_for("li_2025").codes[0] != common.codes[0]        # assay
    assert p.record_for("duempelmann_2026").codes[1] != common.codes[1]  # chemistry


def test_marecko_material_is_marked_mixed():
    """35 scRNA-seq and 68 snRNA-seq assays share one study id, so the material
    code must say so rather than claim one of them."""
    record = StudyPlatform(CSV).record_for("marecko_2024")
    assert "mixed" in record.material


def test_fonseca_version_is_an_explicit_unknown_not_a_guess():
    record = StudyPlatform(CSV).record_for("fonseca_2023")
    assert "unstated" in record.chemistry
