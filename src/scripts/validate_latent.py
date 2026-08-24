"""Does the latent carry anything the model was never given?

    python -m scripts.validate_latent --root . --run r8_direction \
        --runs-dir outputs/r8 --cycle-period 28

Loads a checkpoint, embeds the validation split ONCE, and runs every test off
that single embedding. No training. Runs against checkpoints already on disk,
so results arrive without waiting for a run to finish.

See docs/walkthrough/12-validation.md for what is being asked and why most of
the obvious labels are disqualified -- `age` is a model input, `tissue` and
`platform` are decoder inputs, and `disease_status` is exactly determined by
tissue in this corpus.

Every result is reported WITH its cell count. The corpus has no shared metadata
schema: most labels exist for one or two studies, so "n" is not a footnote here,
it is half the result.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.validate")


# Obs columns requested from the Dataset. A study lacking one reads None for
# its cells; nothing here reaches the model. Grouped by the test each serves.
CIRCULAR_CONTINUOUS = ["CycleDay"]                       # periodic — the headline
LINEAR_CONTINUOUS = ["ProgesteroneSerum", "epithelial.pseudotime"]
BIOLOGY_LABELS = [
    "MenstrualCyclePhase_main", "MenstrualCyclePhase", "phase",
    "AnnotationMain", "AnnotationRefined", "celltype", "lineage",
    "EndometriosisGrade", "Endometriosis_stage",
]
# Negative controls. These SHOULD show nothing: library size is handed to the
# decoder explicitly and mitochondrial fraction is cell stress, so a latent
# organised by them is organised by technical noise.
QC_CONTROLS = ["pct_counts_mt", "total_counts", "n_genes_by_counts",
               "doublet_score", "scDblFinder.score", "scrublet_score"]
ALL_COLUMNS = (CIRCULAR_CONTINUOUS + LINEAR_CONTINUOUS
               + BIOLOGY_LABELS + QC_CONTROLS + ["patient_id"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Experiment name, e.g. r8_direction.")
    p.add_argument("--root", default=None)
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--runs-dir", default=None,
                   help="Look for the run directory under here instead of "
                        "beside the path in its config.")
    p.add_argument("--k", type=int, default=30, help="Neighbours per cell.")
    p.add_argument("--dims", type=int, default=0,
                   help="Project the latent to this many PCs first. 0 uses it "
                        "as-is, which is right here: every test compares "
                        "labels within ONE run, so there is no second latent "
                        "width to reconcile with.")
    p.add_argument("--cycle-period", type=float, default=28.0,
                   help="Days in a full menstrual cycle, for the periodic "
                        "correlation. Supplied rather than inferred: a wrong "
                        "period lowers the coefficient for a reason that has "
                        "nothing to do with the model.")
    p.add_argument("--out", default=None,
                   help="Where to write validation.csv. Defaults to the run "
                        "directory.")
    return p.parse_args()


def collect(model, loader, device, order_len: int):
    """Embed the split, in the loader's own order.

    Returns mu_gauss, mu_angle and study_id as arrays whose row i is the i-th
    cell the loader yielded. That ordering is what lets the metadata be joined:
    see `sampler_order`.
    """
    import numpy as np
    import torch

    # The shared helper, not a local reimplementation. `group_indices` and
    # `group_dns_mask` are dicts OF tensors, so a `hasattr(v, "to")` test
    # silently leaves them on the CPU and the first embedding lookup dies with
    # "expected all tensors to be on the same device". _to_device recurses one
    # level, which is exactly what the Batch schema needs.
    from endo_model.training.trainer import _to_device

    mu_gauss, mu_angle, study = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            out = model.encode(batch)
            mu_gauss.append(out["mu_gauss"].float().cpu().numpy())
            mu_angle.append(out["mu_angle"].float().cpu().numpy())
            study.append(batch["study_id"].cpu().numpy())
    if not mu_gauss:
        return None
    got = np.concatenate(mu_gauss)
    if len(got) != order_len:
        # Loud rather than silent: a mismatch means the metadata join below
        # would pair each cell with another cell's labels, and every number on
        # the page would be wrong in a way that looks plausible.
        raise RuntimeError(
            f"collected {len(got)} embeddings but the sampler yielded "
            f"{order_len} indices — the metadata join would be misaligned"
        )
    return got, np.concatenate(mu_angle), np.concatenate(study)


def sampler_order(loader) -> list[int]:
    """The dataset indices the loader will yield, in order.

    PerStudyBatchSampler is deterministic with shuffle=False, and it drops each
    study's incomplete tail, so this is NOT simply range(len(dataset)). Reading
    it out explicitly is what aligns `extra_obs` with the embeddings.
    """
    return [i for batch in loader.batch_sampler for i in batch]


def main() -> None:
    import numpy as np
    import torch

    args = parse_args()
    root = args.root or str(Path.cwd())

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.latent_validation import (
        circular_linear_corr, format_agreement, label_knn_agreement,
        periodic_corr,
    )
    from endo_model.evaluation.purity_report import best_checkpoint
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager
    from scripts.report_purity import build_loader, resolve_run_dir

    config = Path(root) / args.config_dir / f"{args.run}.yaml"
    if not config.exists():
        logger.error("no config at %s", config)
        sys.exit(1)

    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(str(config), root=root)
    run_dir = resolve_run_dir(root, exp_cfg.output_dir, args.runs_dir)
    if not run_dir.is_dir():
        logger.error("no directory at %s — point --runs-dir at its parent", run_dir)
        sys.exit(1)

    vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))
    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
        for s in data_cfg.resolve_studies(root)
    ]
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)
    names = {s["id"]: s["name"] for s in study_configs}

    logger.info("Building the validation split with %d metadata columns",
                len(ALL_COLUMNS))
    loader = build_loader(root, exp_cfg, train_cfg, data_cfg, vocab,
                          study_configs, extra_obs_columns=ALL_COLUMNS)
    order = sampler_order(loader)
    extra = loader.dataset.extra_obs

    ckpt = best_checkpoint(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    CheckpointManager.load(str(ckpt), model, None, map_location=device)

    logger.info("Embedding %d cells", len(order))
    collected = collect(model, loader, device, len(order))
    if collected is None:
        logger.error("validation loader was empty")
        sys.exit(1)
    mu_gauss, mu_angle, study_id = collected

    meta = {c: extra[c][order] for c in ALL_COLUMNS}
    rows: list[dict] = []

    def record(test, label, scope, n, value, note=""):
        rows.append({"test": test, "label": label, "scope": scope,
                     "n_cells": n, "value": round(float(value), 4), "note": note})

    # ── 1. the headline: does a circular dimension track cycle day? ──────────
    logger.info("── 1. cycle day against the circular latent")
    for col in CIRCULAR_CONTINUOUS:
        day = _as_float(meta[col])
        ok = np.isfinite(day)
        if ok.sum() < 30:
            logger.info("   %s: only %d labelled cells — skipped", col, int(ok.sum()))
            continue
        for d in range(mu_angle.shape[1]):
            r = periodic_corr(mu_angle[ok, d], day[ok], args.cycle_period)
            logger.info("   %s vs circular dim %d: r = %.3f  (n=%d)",
                        col, d, r, int(ok.sum()))
            record("periodic", col, f"circular dim {d}", int(ok.sum()), r)
        best = max(periodic_corr(mu_angle[ok, d], day[ok], args.cycle_period)
                   for d in range(mu_angle.shape[1]))
        logger.info("   best circular dimension: r = %.3f", best)

    # ── 2. continuous covariates, against BOTH branches ─────────────────────
    logger.info("── 2. continuous covariates against the latent")
    for col in LINEAR_CONTINUOUS + QC_CONTROLS:
        x = _as_float(meta[col])
        ok = np.isfinite(x)
        if ok.sum() < 30:
            continue
        n = int(ok.sum())
        rg = max(abs(np.corrcoef(mu_gauss[ok, d], x[ok])[0, 1])
                 for d in range(mu_gauss.shape[1]))
        # Against the circular branch too: serum progesterone rises and falls
        # with the cycle, so it is a periodic quantity and the Gaussian
        # correlation alone would miss it sitting on the torus.
        rc = max(circular_linear_corr(mu_angle[ok, d], x[ok])
                 for d in range(mu_angle.shape[1]))
        kind = "NEGATIVE CONTROL" if col in QC_CONTROLS else "covariate"
        logger.info("   %-24s gaussian |r|=%.3f  circular r=%.3f  (n=%d)  %s",
                    col, rg, rc, n, kind)
        record("linear-corr", col, "gaussian latent", n, rg, kind)
        record("circular-linear", col, "circular latent", n, rc, kind)
        _log_contributors(meta[col], study_id, names, col)

    # ── 3. categorical labels: kNN agreement ────────────────────────────────
    logger.info("── 3. biological labels — kNN agreement (high = latent groups it)")
    dims = args.dims or None
    for col in BIOLOGY_LABELS:
        res = label_knn_agreement(mu_gauss, meta[col], k=args.k, n_components=dims)
        if res["n_cells"] < args.k + 1:
            continue
        logger.info("   %s", format_agreement(col, res))
        record("knn-agreement", col, "all labelled cells",
               res["n_cells"], res["excess"], f"{res['n_labels']} labels")
        _log_contributors(meta[col], study_id, names, col)

    # ── 4. patient identity — the only held-out label in every study ─────────
    logger.info("── 4. patient identity (high = memorised the individual)")
    res = label_knn_agreement(mu_gauss, meta["patient_id"], k=args.k,
                              n_components=dims)
    logger.info("   %s", format_agreement("patient_id", res))
    record("knn-agreement", "patient_id", "all cells", res["n_cells"],
           res["excess"], "high is BAD — individual identity, not biology")

    out = Path(args.out) if args.out else run_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test", "label", "scope",
                                          "n_cells", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    logger.info("Written to %s", path)


def _log_contributors(values, study_id, names, col: str) -> None:
    """Which studies actually supplied this label, and how many cells each.

    The corpus has no shared metadata schema, so a headline "n=48,000" is
    usually one study. Naming it is what stops a single-study result being read
    as a corpus-wide one -- see docs/walkthrough/12-validation.md.
    """
    import numpy as np

    from endo_model.evaluation.latent_validation import _is_labelled

    ok = np.array([_is_labelled(v) for v in values])
    if not ok.any():
        return
    ids, counts = np.unique(np.asarray(study_id)[ok], return_counts=True)
    parts = [f"{names.get(int(i), i)} {int(c):,}" for i, c in zip(ids, counts)]
    logger.info("        from: %s", "; ".join(parts))


def _as_float(values):
    """Object array -> float array, with anything uncoercible as NaN."""
    import numpy as np
    out = np.full(len(values), np.nan, dtype=float)
    for i, v in enumerate(values):
        try:
            if v is not None:
                out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


if __name__ == "__main__":
    main()
