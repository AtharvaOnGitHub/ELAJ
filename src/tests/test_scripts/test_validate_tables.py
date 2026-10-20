"""Does the full-corpus pass's table-building survive the corpus's real labels?

WHY THIS EXISTS. geometry_tables() and _union_severity() are the newest code
in a twelve-hour GPU job, and nothing else tests them. A value spelling they
do not expect, or a `meta` key that is absent for a split, would surface hours
into a cluster run and take a scarce GPU slot with it. Neither function needs
a model, a checkpoint or a corpus -- an array and a dict of object arrays is
the whole interface -- so they are tested here instead.

EVERY LABEL VALUE BELOW IS THE REAL ONE, transcribed from label_survey.csv
(20 October 2026): marecko's `Secretory Early-Mid` with its capitals and
hyphen, garcia-alonso's `early-mid-secretory` with the adjectives reversed,
marecko's numeric rASRM, duempelmann's `CTL`. Synthetic values would test the
code against my expectations rather than against the corpus.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("sklearn")

from scripts.validate_latent import (                       # noqa: E402
    SEVERITY_FINE, _union_severity, geometry_tables,
)

# ── the corpus's own spellings, per label_survey.csv ────────────────────────
MARECKO_PHASE = ["Proliferative", "Secretory Early-Mid", "Secretory Early",
                 "Secretory Mid", "Hormones", "Proliferative Late",
                 "Hormones Menstrual", "Secretory Late"]
GARCIA_PHASE = ["early-mid-secretory", "proliferative", "early-secretory",
                "late-secretory", "mid-secretory"]
DUEMP_PHASE = ["proliferative", "periovulatory", "secretory_mid",
               "secretory_early", "secretory_late"]


def every_phase_source_value() -> list[str]:
    """One source spelling per level MENSTRUAL_PHASE can produce.

    DERIVED, NOT TRANSCRIBED, and that distinction caught a real hole. The
    hand-written lists above come from label_survey.csv's display, which
    truncates each study at eight values with a "+3 more" -- and one of
    marecko's unshown three is the value that harmonises to `menstrual`. So a
    corpus built from those lists never produces that level, and a deliberately
    broken grouping that forgot `menstrual` passed every test in this file.

    Reading the harmoniser instead makes the corpus complete by construction
    and keeps it complete as the vocabulary changes.
    """
    from endo_model.evaluation.label_harmonisation import MENSTRUAL_PHASE

    wanted = {v for v in MENSTRUAL_PHASE.mapping.values() if v is not None}
    chosen, seen = [], set()
    for source, target in MENSTRUAL_PHASE.mapping.items():
        if target in wanted and target not in seen:
            chosen.append(source)
            seen.add(target)
    assert seen == wanted, sorted(wanted - seen)
    return chosen
MARECKO_STAGE = ["0", "1", "2", "3", "4"]
DUEMP_GRADE = ["CTL", "mild", "severe"]
FAMILIES = ["epithelial", "stromal", "endothelial", "immune"]


def corpus(n_per_donor: int = 25, dims: int = 16, seed: int = 5):
    """A meta dict shaped like the real one: most labels in one or two studies."""
    rng = np.random.default_rng(seed)
    rows = []
    # marecko donors: Stage (menstrual), Endometriosis_stage, lineage.
    # Its phase values cover EVERY level the harmoniser can produce, plus the
    # study's own hormone-treated spellings, which the harmoniser drops on
    # purpose and which must therefore not show up as a grouping orphan.
    marecko_phase = every_phase_source_value() + ["Hormones",
                                                  "Hormones Menstrual"]
    for i, stage in enumerate(MARECKO_STAGE):
        for rep in range(max(4, len(marecko_phase))):
            rows.append({"study": "marecko_2024",
                         "donor": f"mar{i}_{rep}",
                         "Stage": marecko_phase[rep % len(marecko_phase)],
                         "Endometriosis_stage": stage,
                         "disease_status": "control" if stage == "0" else "eutopic"})
    # duempelmann donors: MenstrualCyclePhase, EndometriosisGrade, AnnotationMain
    for i, grade in enumerate(DUEMP_GRADE):
        for rep in range(5):
            rows.append({"study": "duempelmann_2026", "donor": f"due{i}_{rep}",
                         "MenstrualCyclePhase": DUEMP_PHASE[(i + rep) % len(DUEMP_PHASE)],
                         "EndometriosisGrade": grade,
                         "disease_status": "control" if grade == "CTL" else "eutopic"})
    # garcia-alonso donors: Stage only, adjectives reversed
    for rep in range(4):
        rows.append({"study": "garcia_alonso_2021_a", "donor": f"gar{rep}",
                     "Stage": GARCIA_PHASE[rep % len(GARCIA_PHASE)]})

    columns = ["Stage", "MenstrualCyclePhase", "MenstrualCyclePhase_main",
               "Endometriosis_stage", "EndometriosisGrade", "disease_status",
               "AnnotationMain", "lineage", "phase", "AnnotationRefined"]
    meta = {c: [] for c in columns}
    meta["patient_id"] = []
    latent, study = [], []
    for r in rows:
        centre = rng.normal(0, 2.0, dims)
        latent.append(rng.normal(centre, 0.8, (n_per_donor, dims)))
        for j in range(n_per_donor):
            meta["patient_id"].append(r["donor"])
            study.append(r["study"])
            for c in columns:
                if c in r:
                    meta[c].append(r[c])
                elif c == "lineage" and r["study"] == "marecko_2024":
                    meta[c].append(FAMILIES[j % 4].capitalize()
                                   if FAMILIES[j % 4] != "stromal"
                                   else "Mesenchymal")
                elif c == "AnnotationMain" and r["study"] == "duempelmann_2026":
                    meta[c].append(FAMILIES[j % 4])
                elif c == "phase" and r["study"] == "marecko_2024":
                    meta[c].append(("G1", "G1", "G1", "S", "G2M")[j % 5])
                elif c == "MenstrualCyclePhase_main" and r["study"] == "duempelmann_2026":
                    meta[c].append(("proliferative", "periovulatory",
                                    "secretory")[j % 3])
                else:
                    meta[c].append(None)
    meta = {c: np.array(v, dtype=object) for c, v in meta.items()}
    return np.concatenate(latent), meta, np.array(study, dtype=object)


def with_harmonised(meta):
    """Add the two harmonised columns and the severity union, as main() does."""
    from endo_model.evaluation.label_harmonisation import (
        LINEAGE, LINEAGE_SOURCE_COLUMNS, MENSTRUAL_PHASE,
        PHASE_SOURCE_COLUMNS, merge_columns,
    )

    meta["cell family"] = merge_columns(meta, LINEAGE_SOURCE_COLUMNS, LINEAGE)
    meta["menstrual phase"] = merge_columns(meta, PHASE_SOURCE_COLUMNS,
                                            MENSTRUAL_PHASE)
    meta[SEVERITY_FINE] = _union_severity(meta)
    return meta


# ── _union_severity ─────────────────────────────────────────────────────────

def test_union_severity_prefixes_each_studys_own_vocabulary():
    _, meta, _ = corpus()
    out = _union_severity(meta)
    present = {v for v in out if v is not None}
    assert present == {"stage_0", "stage_1", "stage_2", "stage_3", "stage_4",
                       "grade_ctl", "grade_mild", "grade_severe"}


def test_union_severity_normalises_a_float_spelled_rASRM_level():
    """An obs column read as float gives "2.0"; as int, "2". Same level."""
    meta = {"patient_id": np.array(["a", "b"], dtype=object),
            "Endometriosis_stage": np.array(["2.0", "2"], dtype=object),
            "EndometriosisGrade": np.array([None, None], dtype=object)}
    assert list(_union_severity(meta)) == ["stage_2", "stage_2"]


def test_union_severity_leaves_unlabelled_cells_alone():
    meta = {"patient_id": np.array(["a", "b", "c", "d"], dtype=object),
            "Endometriosis_stage": np.array([None, "nan", "", "  "], dtype=object),
            "EndometriosisGrade": np.array([None, None, None, None], dtype=object)}
    assert list(_union_severity(meta)) == [None, None, None, None]


def test_union_severity_survives_a_missing_column_entirely():
    """A split where no study carries one of the two columns."""
    meta = {"patient_id": np.array(["a", "b"], dtype=object),
            "EndometriosisGrade": np.array(["CTL", "severe"], dtype=object)}
    assert list(_union_severity(meta)) == ["grade_ctl", "grade_severe"]


# ── geometry_tables ─────────────────────────────────────────────────────────

def test_the_test_corpus_produces_every_harmonised_phase_level():
    """Guards the hole the mutation check found.

    If this corpus stops covering a level, a grouping that forgets that level
    passes `test_no_grouping_orphans...` while being broken. So the coverage
    is asserted rather than assumed.
    """
    from endo_model.evaluation.label_harmonisation import MENSTRUAL_PHASE

    _, meta, _ = corpus()
    meta = with_harmonised(meta)
    present = {v for v in meta["menstrual phase"] if v is not None}
    expected = {v for v in MENSTRUAL_PHASE.mapping.values() if v is not None}
    assert present == expected, sorted(expected - present)


def test_every_expected_table_is_produced_and_non_empty():
    latent, meta, _ = corpus()
    tables = geometry_tables(latent, with_harmonised(meta), k=15)
    for name in ("level_statistics", "centroid_distances", "neighbour_counts",
                 "donor_adjacency", "donor_levels", "grouped_distances"):
        assert tables[name], f"{name} is empty"


def test_level_statistics_account_for_every_labelled_cell():
    """n must sum, per label, to the cells carrying that label. Nothing lost."""
    latent, meta, _ = corpus()
    meta = with_harmonised(meta)
    tables = geometry_tables(latent, meta, k=15)
    from endo_model.evaluation.latent_validation import _is_labelled

    per_label: dict[str, int] = {}
    for r in tables["level_statistics"]:
        per_label[r["label"]] = per_label.get(r["label"], 0) + r["n"]
    for label, total in per_label.items():
        expected = int(sum(_is_labelled(v) for v in meta[label]))
        assert total == expected, (label, total, expected)


def test_no_grouping_orphans_a_level_the_real_vocabularies_produce():
    """The failure this whole test file is for: a drifted grouping.

    An orphan means a declared grouping says nothing about a level the corpus
    actually holds, so that level's cells drop out of the row silently. With
    marecko's, duempelmann's and garcia-alonso's real spellings all present,
    any orphan here is a genuine gap.
    """
    latent, meta, _ = corpus()
    tables = geometry_tables(latent, with_harmonised(meta), k=15)
    orphaned = {(r["label"], r["grouping"], r["orphan_levels"])
                for r in tables["grouped_distances"] if r["orphan_levels"]}
    assert not orphaned, orphaned


def test_the_severity_variants_all_appear_with_the_right_group_counts():
    latent, meta, _ = corpus()
    tables = geometry_tables(latent, with_harmonised(meta), k=15)
    seen = {r["grouping"] for r in tables["grouped_distances"]}
    for name in ("severity_merged", "severity_cases_only", "severity_by_study",
                 "case_vs_control"):
        assert name in seen, name
    groups = {r["grouping"]: {r2["from"] for r2 in tables["grouped_distances"]
                              if r2["grouping"] == r["grouping"]}
              | {r2["to"] for r2 in tables["grouped_distances"]
                 if r2["grouping"] == r["grouping"]}
              for r in tables["grouped_distances"]}
    assert groups["case_vs_control"] == {"case", "control"}
    assert groups["severity_cases_only"] == {"mild", "severe"}
    assert groups["severity_merged"] == {"control", "mild", "severe"}


def test_all_five_phase_variants_appear():
    latent, meta, _ = corpus()
    tables = geometry_tables(latent, with_harmonised(meta), k=15)
    seen = {r["grouping"] for r in tables["grouped_distances"]}
    for name in ("phase_3_menstrual_own", "phase_3_menstrual_prolif",
                 "phase_3_menstrual_dropped", "phase_2_boundary_dropped",
                 "phase_2_menstrual_prolif"):
        assert name in seen, name


def test_donor_adjacency_rows_reference_donors_with_a_recorded_level():
    """Every donor in the matrix must appear in donor_levels for that label.

    merge_labels joins the two by donor name, so a donor present in one and
    absent from the other silently loses its relations from the cross-donor
    statistic.
    """
    latent, meta, _ = corpus()
    tables = geometry_tables(latent, with_harmonised(meta), k=15)
    levels: dict[str, set] = {}
    for r in tables["donor_levels"]:
        levels.setdefault(r["label"], set()).add(r["donor"])
    for r in tables["donor_adjacency"]:
        assert r["from_donor"] in levels[r["label"]], r
        assert r["to_donor"] in levels[r["label"]], r


def test_every_row_writes_cleanly_through_the_real_csv_fieldnames(tmp_path):
    """The field lists in main() must cover what geometry_tables emits.

    A key the DictWriter has no column for raises; extrasaction="ignore" hides
    that, so this checks the other direction -- that nothing a reader needs is
    being silently dropped on the way to disk.
    """
    latent, meta, _ = corpus()
    tables = geometry_tables(latent, with_harmonised(meta), k=15)
    fields = {
        "level_statistics": ["label", "donor", "level", "n", "sumsq"]
                            + [f"dim{d}" for d in range(latent.shape[1])],
        "centroid_distances": ["label", "from", "to", "distance",
                               "distance_over_scale", "n_from", "n_to",
                               "scatter_from", "scatter_to"],
        "neighbour_counts": ["label", "own", "neighbour", "count"],
        "donor_adjacency": ["label", "from_donor", "to_donor", "count"],
        "donor_levels": ["label", "donor", "level", "n_cells"],
        "grouped_distances": ["label", "grouping", "from", "to", "distance",
                              "over_scatter", "n_from", "n_to",
                              "knn_agreement", "knn_excess",
                              "excluded_levels", "orphan_levels"],
    }
    for name, columns in fields.items():
        extra = set().union(*(set(r) for r in tables[name])) - set(columns)
        assert not extra, f"{name} emits {sorted(extra)} with no column"
        dest = tmp_path / f"{name}.csv"
        with open(dest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            w.writerows(tables[name])
        assert len(list(csv.DictReader(open(dest)))) == len(tables[name])


def test_a_label_absent_from_meta_is_skipped_not_fatal():
    """A split where a whole study is missing takes columns with it."""
    latent, meta, _ = corpus()
    meta = with_harmonised(meta)
    for gone in ("AnnotationRefined", "EndometriosisGrade", "phase"):
        meta.pop(gone, None)
    tables = geometry_tables(latent, meta, k=15)
    assert tables["level_statistics"]
    assert not any(r["label"] == "AnnotationRefined"
                   for r in tables["level_statistics"])
