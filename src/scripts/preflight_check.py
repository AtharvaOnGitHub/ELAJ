#!/usr/bin/env python3
"""Pre-training preflight: verify the corpus and vocabulary are training-ready.

Reads only — nothing is written and no model is constructed.  Run this on the
HPC before the first training job and paste the output for review.

    python src/scripts/preflight_check.py --root /path/to/project

Checks performed:
  1. vocabulary.json  — size, group count, group size distribution
  2. studies.csv      — required columns, contiguous ids, resolvable paths
  3. per-study h5ad   — cell/gene counts, overlap with the vocabulary,
                        patient_id presence, disease_status label values
  4. patient_id       — uniqueness *across* studies, not merely within one
  5. stored counts    — NaN / Inf / negative values, and the value range
  6. obs values       — tissue codes within their embedding tables (both ends),
                        age finite

Checks 5 and 6 exist because both failure modes are invisible at their source.
DSBN normalises over the flattened (B x G) axis, so one NaN anywhere in a batch
makes every activation NaN; and an out-of-range embedding index reads unallocated
memory on CUDA instead of raising.  Both surface as a NaN in the latent sampler,
modules away from the cause.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from endo_model.configs.data_config import load_studies_csv  # noqa: E402
from endo_model.data.constants import (  # noqa: E402
    DISEASE_STATUS_TO_INT,
    TISSUE_LEVEL_SIZES,
)
from endo_model.data.vocabulary import GeneVocabulary  # noqa: E402


def check_vocabulary(path: Path) -> GeneVocabulary | None:
    """Report vocabulary size and the realised group-size distribution."""
    print(f"\n[1] vocabulary: {path}")
    if not path.exists():
        print("    MISSING — run scripts/build_vocabulary.py first.")
        return None

    vocab = GeneVocabulary(str(path))
    sizes = np.array(list(vocab.group_sizes.values()))

    print(f"    D (genes)      : {vocab.vocab_size}")
    print(f"    K (groups)     : {len(vocab.group_names)}")
    print(f"    group size     : min={sizes.min()} median={int(np.median(sizes))} "
          f"mean={sizes.mean():.1f} max={sizes.max()}")
    for q in (10, 25, 50, 75, 90):
        print(f"      p{q:<3d}        : {np.percentile(sizes, q):.0f}")
    print(f"    d_eff = d * K  : {len(vocab.group_names)} * d_token")
    return vocab


def check_studies_csv(path: Path, root: Path) -> list:
    """Report study count, id contiguity, and whether each h5ad resolves."""
    print(f"\n[2] studies.csv: {path}")
    if not path.exists():
        print("    MISSING — the preprocessing pipeline has not produced it.")
        return []

    studies = load_studies_csv(str(path))
    print(f"    studies        : {len(studies)}")

    ids = [s.id for s in studies]
    expected = list(range(len(studies)))
    if ids == expected:
        print(f"    ids            : contiguous 0..{len(studies) - 1}  OK")
    else:
        print(f"    ids            : NOT contiguous — {ids}")
        print("                     n_studies indexes DSBN/study embeddings; fix "
              "the CSV or use data.exclude_studies")

    print(f"\n    {'name':<28} {'id':>3}  {'path resolves':<14} tissue_types")
    for s in studies:
        resolved = Path(s.path)
        if not resolved.is_absolute():
            resolved = root / s.path
        mark = "yes" if resolved.exists() else "NO  <-- fix"
        print(f"    {s.name:<28} {s.id:>3}  {mark:<14} {'|'.join(s.tissue_types)}")
    return studies


def check_study_h5ad(
    study, root: Path, vocab: GeneVocabulary | None
) -> set[str]:
    """Report one study's shape, vocabulary overlap, and obs label values.

    Returns the study's patient_id set, so the caller can check for collisions
    across studies.
    """
    path = Path(study.path)
    if not path.is_absolute():
        path = root / study.path
    if not path.exists():
        print(f"    {study.name:<28} SKIPPED (path does not resolve)")
        return set()

    import anndata

    adata = anndata.read_h5ad(path, backed="r")
    n_cells, n_genes = adata.shape

    in_vocab = (
        sum(1 for g in adata.var_names if g in vocab) if vocab is not None else 0
    )
    of_vocab = 100.0 * in_vocab / vocab.vocab_size if vocab is not None else 0.0

    has_pid = "patient_id" in adata.obs.columns
    patients = (
        set(adata.obs["patient_id"].astype(str)) if has_pid else set()
    )

    print(f"    {study.name:<28} cells={n_cells:>8}  genes={n_genes:>6}  "
          f"in_vocab={in_vocab:>6} ({of_vocab:4.1f}% of D)  "
          f"patients={len(patients):>4}"
          f"{'' if has_pid else '   MISSING patient_id'}")

    if "disease_status" in adata.obs.columns:
        counts = Counter(
            str(v).strip().lower() for v in adata.obs["disease_status"]
        )
        unknown = {k: v for k, v in counts.items() if k not in DISEASE_STATUS_TO_INT}
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"        disease_status: {summary}")
        if unknown:
            print(f"        UNRECOGNISED  : {sorted(unknown)}  -> silently 'other'")
        declared = {t for t in study.tissue_types if t in DISEASE_STATUS_TO_INT}
        if declared and not declared & set(counts):
            print(f"        MISMATCH      : studies.csv declares {sorted(declared)} "
                  f"but no cell carries any of them")
    else:
        print("        disease_status: MISSING -> every cell becomes 'other'")

    adata.file.close()
    return patients


def _scan_values(dset, chunk: int = 20_000_000) -> dict:
    """Reduce a 1-D HDF5 float dataset to NaN/Inf/negative counts and a range.

    Read in chunks: the stored values of the largest study are several GB, and
    the whole point of this check is to run before committing a GPU job.
    """
    n = int(dset.shape[0])
    stats = {"n": n, "nan": 0, "inf": 0, "neg": 0,
             "min": float("inf"), "max": float("-inf")}

    for start in range(0, n, chunk):
        block = np.asarray(dset[start : start + chunk], dtype=np.float64)
        is_nan = np.isnan(block)
        # ~isfinite covers NaN and Inf both, so subtract the NaNs back out.
        is_inf = ~np.isfinite(block) & ~is_nan
        stats["nan"] += int(is_nan.sum())
        stats["inf"] += int(is_inf.sum())

        finite = block[np.isfinite(block)]
        if finite.size:
            stats["neg"] += int((finite < 0).sum())
            stats["min"] = min(stats["min"], float(finite.min()))
            stats["max"] = max(stats["max"], float(finite.max()))

    return stats


def check_count_matrix(study, root: Path) -> bool:
    """Verify one study's stored counts contain nothing the model cannot consume.

    NaN and Inf are the dangerous cases and they are not local: DSBN applies
    BatchNorm over the flattened (B x G) axis, so one bad value anywhere in a
    batch makes the batch mean and variance NaN and every activation after it
    NaN too.  The failure then surfaces in the latent sampler, several modules
    away from its cause.

    Returns True if the study is clean.
    """
    path = Path(study.path)
    if not path.is_absolute():
        path = root / study.path
    if not path.exists():
        print(f"    {study.name:<28} SKIPPED (path does not resolve)")
        return True

    import h5py

    with h5py.File(path, "r") as f:
        x = f["X"]
        # AnnData writes sparse X as a group (data/indices/indptr) and dense X
        # as a plain dataset.
        values = x["data"] if isinstance(x, h5py.Group) else x
        stats = _scan_values(values)

    bad = stats["nan"] or stats["inf"] or stats["neg"]
    flag = "  <-- PROBLEM" if bad else ""
    print(
        f"    {study.name:<28} stored={stats['n']:>12,}  "
        f"nan={stats['nan']:>6}  inf={stats['inf']:>6}  neg={stats['neg']:>6}  "
        f"range=[{stats['min']:.4g}, {stats['max']:.4g}]{flag}"
    )
    return not bad


def check_obs_values(study, root: Path) -> bool:
    """Verify the obs columns the model consumes are inside what it can accept.

    Two failure modes, both invisible where they originate:

    * A tissue code outside its embedding table.  Checked at **both** ends —
      "unknown" is often encoded -1, and a negative index reads unallocated
      memory on CUDA rather than raising as it would on CPU.
    * A non-finite age.  AgeEncoder guards with torch.isnan, which is False for
      inf, and the value then reaches torch.sin — sin(inf) is NaN.

    Returns True if the study is clean.
    """
    import anndata
    import pandas as pd

    columns = ("tissue_compartment", "tissue_organ", "tissue_type", "tissue_microsite")

    path = Path(study.path)
    if not path.is_absolute():
        path = root / study.path
    if not path.exists():
        return True

    adata = anndata.read_h5ad(path, backed="r")
    problems: list[str] = []

    for level, (col, size) in enumerate(zip(columns, TISSUE_LEVEL_SIZES)):
        if col not in adata.obs.columns:
            continue
        coerced = pd.to_numeric(adata.obs[col], errors="coerce")
        if coerced.isna().all():
            continue  # treated as absent by the Dataset; not a range problem
        top, bottom = coerced.max(), coerced.min()
        if top >= size:
            problems.append(f"{col} max={top:g} >= table size {size} (level {level})")
        if bottom < 0:
            problems.append(f"{col} min={bottom:g} < 0 (level {level})")

    if "age" in adata.obs.columns:
        age = pd.to_numeric(adata.obs["age"], errors="coerce")
        # NaN is the documented "unknown" marker and is handled; inf is not.
        n_inf = int(np.isinf(age.to_numpy(dtype=np.float64)).sum())
        if n_inf:
            problems.append(f"age has {n_inf} infinite value(s) -> sin(inf) = NaN")

    adata.file.close()

    if problems:
        print(f"    {study.name:<28} " + "; ".join(problems) + "   <-- PROBLEM")
        return False
    return True


def check_patient_id_collisions(per_study: dict[str, set[str]]) -> None:
    """Verify patient_id is unique across studies, not merely within one.

    The split pools every patient_id into one global set.  Two studies reusing
    a label ("P1", "patient_1") would be treated as the same person: their
    cells are forced into the same split, and the split sizes are wrong.
    """
    print("\n[4] patient_id uniqueness across studies")
    total = sum(len(p) for p in per_study.values())
    union = set().union(*per_study.values()) if per_study else set()

    print(f"    sum of per-study patients : {total}")
    print(f"    globally unique ids       : {len(union)}")

    if len(union) == total:
        print("    no collisions  OK")
        return

    print(f"    COLLISION — {total - len(union)} id(s) shared between studies")
    names = list(per_study)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = per_study[a] & per_study[b]
            if shared:
                sample = sorted(shared)[:5]
                more = f" (+{len(shared) - 5} more)" if len(shared) > 5 else ""
                print(f"      {a} & {b}: {sample}{more}")
    print("    Prefix each study's patient_id with the study name before "
          "training, or the split will merge distinct people.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root for relative paths.")
    parser.add_argument(
        "--vocabulary", default="data/processed/vocabulary.json",
        help="Vocabulary path, relative to --root.",
    )
    parser.add_argument(
        "--studies-csv", default="data/processed/studies.csv",
        help="Studies CSV path, relative to --root.",
    )
    parser.add_argument(
        "--skip-h5ad", action="store_true",
        help="Skip check 3 (reads every .h5ad header; slow on a cold filesystem).",
    )
    parser.add_argument(
        "--skip-counts", action="store_true",
        help="Skip check 5 (streams every stored count value; several GB of "
             "sequential read, but the only way to catch NaN/Inf before a job).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"ELAJ preflight — root: {root}")

    vocab = check_vocabulary(root / args.vocabulary)
    studies = check_studies_csv(root / args.studies_csv, root)

    if studies and not args.skip_h5ad:
        print("\n[3] per-study .h5ad")
        per_study = {
            s.name: check_study_h5ad(s, root, vocab) for s in studies
        }
        check_patient_id_collisions({k: v for k, v in per_study.items() if v})
    elif args.skip_h5ad:
        print("\n[3] per-study .h5ad — skipped (--skip-h5ad)")

    clean = True
    if studies and not args.skip_counts:
        print("\n[5] stored count values (NaN / Inf / negative / range)")
        for s in studies:
            clean &= check_count_matrix(s, root)

        print("\n[6] obs values the model consumes (tissue codes, age)")
        obs_clean = True
        for s in studies:
            obs_clean &= check_obs_values(s, root)
        if obs_clean:
            print("    all within range  OK")
        clean &= obs_clean
    elif args.skip_counts:
        print("\n[5] stored count values — skipped (--skip-counts)")

    print("\nDone." if clean else "\nDone — PROBLEMS found above.")


if __name__ == "__main__":
    main()
