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
    p.add_argument("--permutations", type=int, default=200,
                   help="Permutations for the cross-donor null. duempelmann "
                        "contributes 9 validation donors, and at that count "
                        "the raw excess is noise -- a donor-only latent "
                        "produced +0.59 on synthetic data. 0 disables.")
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
        circular_coordinates, circular_linear_corr, cross_group_agreement,
        format_agreement, label_knn_agreement, periodic_corr,
    )
    from endo_model.evaluation.scib_metrics import compute_scib_metrics
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

    # ── 3. categorical labels, on EACH HALF OF THE LATENT ───────────────────
    # The first run tested the Gaussian half only and reported the result as
    # "the latent groups cell type". Which half does the work is exactly the
    # open question about the circular branch, so every label now runs on both
    # and on the concatenation the decoder actually receives.
    circ = circular_coordinates(mu_angle)
    branches = {
        "gaussian": mu_gauss,
        "circular": circ,
        "combined": np.concatenate([mu_gauss, circ], axis=-1),
    }
    logger.info("── 3. biological labels — kNN agreement, per latent half")
    logger.info("   (branch widths: gaussian %d, circular %d, combined %d)",
                mu_gauss.shape[1], circ.shape[1], branches["combined"].shape[1])
    dims = args.dims or None
    for col in BIOLOGY_LABELS + ["patient_id"]:
        first = True
        for bname, lat in branches.items():
            res = label_knn_agreement(lat, meta[col], k=args.k, n_components=dims)
            if res["n_cells"] < args.k + 1:
                break
            logger.info("   %-9s %s", bname, format_agreement(col, res))
            note = ("high is BAD — individual identity" if col == "patient_id"
                    else f"{res['n_labels']} labels")
            record("knn-agreement", col, bname, res["n_cells"], res["excess"], note)
            if first:
                _log_contributors(meta[col], study_id, names, col)
                first = False

    # ── 4. does the label survive when same-donor neighbours are removed? ────
    # Phase (32.1%), grade (25.9%) and donor identity (27.6%) all scored the
    # same in the first run, which is what a latent clustering by donor looks
    # like: each donor carries one phase and one grade, so both come for free.
    # Excluding a cell's own donor removes that shortcut.
    logger.info("── 4. cross-donor: same test, same-donor neighbours discarded")
    for col in ["MenstrualCyclePhase_main", "EndometriosisGrade",
                "AnnotationMain", "phase", "Endometriosis_stage"]:
        for bname, lat in branches.items():
            res = cross_group_agreement(lat, meta[col], meta["patient_id"],
                                        k=max(args.k, 100),
                                        n_permutations=args.permutations)
            if not np.isfinite(res.get("agreement", float("nan"))):
                break
            p_val = res.get("p_value", float("nan"))
            verdict = ("" if not np.isfinite(p_val)
                       else "  SIGNIFICANT" if p_val <= 0.05
                       else "  not significant")
            logger.info(
                "   %-9s %-26s n=%6d  excess=%6.1f%%  null95=%.2f  p=%.3f  "
                "(%.0f of %d neighbours kept, %d donors)%s",
                bname, col, res["n_cells"], 100 * res["excess"],
                res.get("null_p95", float("nan")), p_val,
                res["neighbours_kept"], max(args.k, 100),
                res.get("n_groups", 0), verdict,
            )
            record("cross-donor", col, bname, res["n_cells"], res["excess"],
                   f"p={p_val:.3f}; null95={res.get('null_p95', float('nan')):.2f}; "
                   f"{res['neighbours_kept']:.0f} kept; "
                   f"{res.get('n_groups', 0)} donors")

    # ── 5. NMI and ARI — a differently-built check on the same question ──────
    # kNN agreement asks "are a cell's neighbours like it". NMI and ARI cluster
    # first and then compare to the labels, so they fail in different ways.
    # Agreement between the two is far stronger than either alone, and this is
    # what the scIB benchmark (Luecken et al. 2022) reports.
    logger.info("── 5. NMI / ARI on cell type (scIB-style, clustering-based)")
    for col in ["AnnotationMain", "AnnotationRefined", "celltype"]:
        vals = meta[col]
        ok = np.array([v is not None and str(v).strip().lower()
                       not in ("", "nan", "none", "na", "unknown") for v in vals])
        if ok.sum() < 100:
            continue
        _, bio = np.unique(np.asarray(vals)[ok].astype(str), return_inverse=True)
        for bname, lat in branches.items():
            try:
                m = compute_scib_metrics(lat[ok], study_id[ok], bio)
            except Exception as exc:                      # noqa: BLE001
                logger.warning("   %s %s: %s", bname, col, exc)
                continue
            shown = {k2: v for k2, v in m.items() if np.isfinite(v)}
            logger.info("   %-9s %-20s n=%6d  %s", bname, col, int(ok.sum()),
                        "  ".join(f"{k2}={v:.3f}" for k2, v in shown.items()))
            for k2, v in shown.items():
                record(f"scib-{k2}", col, bname, int(ok.sum()), v, "")

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
