"""What each study's h5ad says about how its cells were made, per sample.

    python -m scripts.describe_obs_platform --config configs/experiment/c_ref.yaml \
        --root . --studies garcia_alonso_2021_a garcia_alonso_2021_b garcia_alonso_2021_c

CELLxGENE h5ads carry `assay` (an EFO term such as "10x 3' v3"),
`suspension_type` (cell or nucleus), and sometimes `tissue_type` (tissue or
organoid) per cell. configs/data/study_platform.csv has garcia_alonso_2021_a
and _c at code 0 (unknown) in every column because those files were not in
the local checkout when the table was written; the cluster has them. This
prints, per study and per `source_sample_id`, the distinct combinations of
those columns with cell counts, in the shape sample_platform.csv wants, so
the rows can be written from the data rather than guessed.

Reads obs only; never loads X.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("elaj.obs_platform")

COLUMNS = ("assay", "assay_ontology_term_id", "suspension_type", "tissue_type",
           "tissue", "development_stage", "donor_id")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--studies", nargs="+", required=True,
                   help="Study names as in studies.csv.")
    p.add_argument("--sample-column", default="source_sample_id")
    return p.parse_args()


def read_obs(path: str):
    """The obs frame alone. anndata reads it without touching X."""
    import h5py
    from anndata.experimental import read_elem

    with h5py.File(path, "r") as f:
        return read_elem(f["obs"])


def describe(name: str, obs, sample_column: str) -> None:
    present = [c for c in COLUMNS if c in obs.columns]
    missing = [c for c in COLUMNS if c not in obs.columns]
    print(f"\n== {name}: {len(obs)} cells; columns present: {present}; "
          f"absent: {missing}")
    if sample_column not in obs.columns:
        print(f"   no {sample_column!r} column; per-study totals only")
        keys = [c for c in ("assay", "suspension_type", "tissue_type") if c in obs.columns]
        if keys:
            print(obs.groupby(keys, observed=True).size().to_string())
        return
    keys = [sample_column] + [c for c in ("assay", "suspension_type", "tissue_type")
                              if c in obs.columns]
    table = obs.groupby(keys, observed=True).size().rename("n_cells").reset_index()
    print(table.to_string(index=False))
    print(f"   {table[sample_column].nunique()} samples; combinations per sample: "
          f"{table.groupby(sample_column).size().max()} at most")
    if "assay" in obs.columns:
        print("   assay totals:", obs["assay"].value_counts().to_dict())
    if "suspension_type" in obs.columns:
        print("   suspension totals:", obs["suspension_type"].value_counts().to_dict())


def main() -> None:
    args = parse_args()
    from endo_model.configs import load_experiment

    _, _, _, data_cfg = load_experiment(args.config, root=args.root)
    by_name = {s.name: s for s in data_cfg.resolve_studies(args.root)}
    for name in args.studies:
        if name not in by_name:
            logger.error("no study %r in the corpus; have %s", name, sorted(by_name))
            continue
        path = str(Path(args.root) / by_name[name].path)
        describe(name, read_obs(path), args.sample_column)


if __name__ == "__main__":
    main()
