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
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from endo_model.configs.data_config import load_studies_csv  # noqa: E402
from endo_model.data.constants import DISEASE_STATUS_TO_INT  # noqa: E402
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

    print("\nDone.")


if __name__ == "__main__":
    main()
