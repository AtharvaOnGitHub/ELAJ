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
    "MenstrualCyclePhase_main", "MenstrualCyclePhase", "Stage",
    "AnnotationMain", "AnnotationRefined", "celltype", "lineage",
    "EndometriosisGrade", "Endometriosis_stage",
    # `phase` is marecko's CELL CYCLE column -- G1, G2M, S. It is kept as a
    # NEGATIVE control, not as menstrual phase, which is what it was read as
    # for two waves. marecko's menstrual phase is in `Stage`.
    "phase",
]
# Merged across studies from the columns above; see label_harmonisation.
HARMONISED = ["cell family", "menstrual phase"]
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
    p.add_argument("--splits", nargs="+", default=["val"],
                   choices=["train", "val", "test"],
                   help="Which splits to embed. More donors is the ONLY thing "
                        "that lifts the cross-donor test off its floor: at six "
                        "donors the smallest reachable p is 0.10, at eleven it "
                        "is 0.002. 'val test' is the natural choice -- both are "
                        "held out, and the test split has never been used for "
                        "anything, so nothing has been tuned against it. Adding "
                        "'train' brings far more donors but the model has seen "
                        "those cells; the phase LABEL was never an input in "
                        "either case, so what it risks is memorised placement "
                        "rather than a leaked label.")
    p.add_argument("--permutations", type=int, default=200,
                   help="Permutations for the cross-donor null. duempelmann "
                        "contributes 9 validation donors, and at that count "
                        "the raw excess is noise -- a donor-only latent "
                        "produced +0.59 on synthetic data. 0 disables.")
    p.add_argument("--out", default=None,
                   help="Where validation.csv AND the figures go. Defaults to "
                        "the run directory. Set it per split combination: two "
                        "runs sharing an output directory overwrite each "
                        "other's pictures, which is silent and total.")
    p.add_argument("--umap-cells", type=int, default=150000,
                   help="Points the UMAP figures draw. The full corpus is "
                        "~800,000, which costs most of an hour to embed and "
                        "renders as a solid block. The sample size and seed go "
                        "on the figure.")
    p.add_argument("--baseline-cells", type=int, default=25000,
                   help="Cells kept for the raw-counts baseline. This is what "
                        "separates 'the model destroyed the signal' from 'the "
                        "signal is not in the data' -- see "
                        "evaluation/expression_baseline.py. 0 disables it. "
                        "Costs cells x vocab x 4 bytes in memory.")
    p.add_argument("--topology-cells", type=int, default=2000,
                   help="Cells sampled for the persistent-homology figure. "
                        "The computation is roughly cubic, so this is the one "
                        "number that decides whether that figure takes a "
                        "minute or an hour: 900 points runs in seconds, 3000 "
                        "in tens of minutes. 1500-2000 recovers a torus "
                        "reliably. 0 skips the topology.")
    p.add_argument("--no-figures", action="store_true",
                   help="Measure and write the CSV only.")
    return p.parse_args()


def collect(model, loader, device, order_len: int, reservoir=None,
            row_offset: int = 0):
    """Embed the split, in the loader's own order.

    Returns mu_gauss, mu_angle, kappa and study_id as arrays whose row i is the
    i-th cell the loader yielded. That ordering is what lets the metadata be
    joined: see `sampler_order`.

    `reservoir`, when given, is offered each batch's raw counts for the
    expression baseline. It is filled during this pass rather than in a second
    one because the counts are already in memory here and re-reading the corpus
    to get them would double the run.
    """
    import numpy as np
    import torch

    # The shared helper, not a local reimplementation. `group_indices` and
    # `group_dns_mask` are dicts OF tensors, so a `hasattr(v, "to")` test
    # silently leaves them on the CPU and the first embedding lookup dies with
    # "expected all tensors to be on the same device". _to_device recurses one
    # level, which is exactly what the Batch schema needs.
    from endo_model.training.trainer import _to_device

    mu_gauss, mu_angle, kappa, study = [], [], [], []
    # Collected here because they are BATCH TENSOR fields, not obs columns.
    # extra_obs cannot reach them, so the control panel's "given to the model"
    # block silently had nothing to draw and the figure lost a third of its
    # argument without any error.
    tissue, platform = [], []
    seen = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if reservoir is not None:
                counts = batch["counts"].float().cpu().numpy()
                # Tags are GLOBAL row indices, so the sampled cells can be
                # paired with their labels afterwards. The reservoir keeps a
                # random subset, not a prefix, so slicing would misalign it.
                reservoir.add(counts,
                              np.arange(row_offset + seen,
                                        row_offset + seen + len(counts)))
                seen += len(counts)
            # Level 2 is tissue_type and code 0 is the assay: the two the
            # decoder is conditioned on, which is what makes them controls.
            tissue.append(batch["tissue_levels"][:, 2].cpu().numpy())
            platform.append(batch["platform_codes"][:, 0].cpu().numpy())
            batch = _to_device(batch, device)
            out = model.encode(batch)
            mu_gauss.append(out["mu_gauss"].float().cpu().numpy())
            mu_angle.append(out["mu_angle"].float().cpu().numpy())
            kappa.append(out["kappa"].float().cpu().numpy())
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
    return (got, np.concatenate(mu_angle), np.concatenate(kappa),
            np.concatenate(study), np.concatenate(tissue),
            np.concatenate(platform))


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

    ckpt = best_checkpoint(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    CheckpointManager.load(str(ckpt), model, None, map_location=device)

    # One embedding pass per split, concatenated. Each split is a separate
    # Dataset, so its own sampler order is what aligns its metadata; mixing the
    # orders would pair cells with other cells' labels.
    reservoir = None
    if args.baseline_cells > 0:
        from endo_model.evaluation.expression_baseline import ReservoirSampler
        reservoir = ReservoirSampler(args.baseline_cells, seed=exp_cfg.seed)

    parts, metas, split_tags = [], [], []
    for split in args.splits:
        logger.info("Building the %s split with %d metadata columns",
                    split, len(ALL_COLUMNS))
        loader = build_loader(root, exp_cfg, train_cfg, data_cfg, vocab,
                              study_configs, extra_obs_columns=ALL_COLUMNS,
                              split=split)
        order = sampler_order(loader)
        extra = loader.dataset.extra_obs
        logger.info("Embedding %d cells from %s", len(order), split)
        # row_offset keeps the reservoir's tags pointing at rows of the
        # CONCATENATED arrays rather than at rows of this split.
        got = collect(model, loader, device, len(order), reservoir,
                      row_offset=sum(len(p_[0]) for p_ in parts))
        if got is None:
            logger.warning("  %s split was empty — skipped", split)
            continue
        parts.append(got)
        metas.append({c: extra[c][order] for c in ALL_COLUMNS})
        split_tags.append(np.full(len(got[0]), split, dtype=object))

    if not parts:
        logger.error("every requested split was empty")
        sys.exit(1)

    mu_gauss = np.concatenate([p_[0] for p_ in parts])
    mu_angle = np.concatenate([p_[1] for p_ in parts])
    kappa = np.concatenate([p_[2] for p_ in parts])
    study_id = np.concatenate([p_[3] for p_ in parts])
    split_of = np.concatenate(split_tags)
    meta = {c: np.concatenate([m[c] for m in metas]) for c in ALL_COLUMNS}

    # The two decoder covariates, named rather than left as integer codes.
    # These are the "given to the model" block of the control panel: the
    # decoder is conditioned on both, which is exactly why a latent that still
    # encodes them has not done what the conditioning was meant to allow.
    tissue_names = {}
    if data_cfg.tissue_level_names_csv:
        from endo_model.data.tissue_level_names import TissueLevelNames
        tissue_names = TissueLevelNames(
            Path(root) / data_cfg.tissue_level_names_csv
        ).for_index(2)
    meta["tissue"] = np.array(
        [tissue_names.get(int(v), f"tissue {int(v)}")
         for v in np.concatenate([p_[4] for p_ in parts])], dtype=object)
    # Named, not coded. "assay 1" on a bar chart tells a reader nothing, and
    # the codes are an implementation detail of the conditioning tensor rather
    # than something a figure should expose. Names come from the assay_family
    # column of study_platform.csv, which is where the codes are defined.
    _ASSAY_NAMES = {0: "unknown", 1: "droplet 3\u2032",
                    2: "droplet 5\u2032", 3: "microwell"}
    meta["platform"] = np.array(
        [_ASSAY_NAMES.get(int(v), f"assay {int(v)}")
         for v in np.concatenate([p_[5] for p_ in parts])],
        dtype=object)
    logger.info("Total: %d cells across split(s) %s",
                len(mu_gauss), ", ".join(args.splits))

    baseline, baseline_rows = None, None
    if reservoir is not None and len(reservoir) > args.k + 1:
        from endo_model.evaluation.expression_baseline import counts_pca
        logger.info("Raw-counts baseline: %d of %d cells sampled, %d genes",
                    len(reservoir), reservoir.n_seen, reservoir.rows.shape[1])
        baseline = counts_pca(reservoir.rows,
                              n_components=mu_gauss.shape[1] + 2 * mu_angle.shape[1],
                              seed=exp_cfg.seed)
        baseline_rows = reservoir.tags
        del reservoir

    # Two derived columns, merged from whichever studies supply them. Added to
    # `meta` so every section below picks them up without a special case.
    #
    # This is what takes the corpus from one study to two: duempelmann's
    # AnnotationMain and marecko's lineage are the same four cell families, and
    # marecko's menstrual phase is in `Stage` -- its `phase` column holds
    # G1/S/G2M, which is CELL CYCLE and now serves as a negative control.
    from endo_model.evaluation.label_harmonisation import (
        LINEAGE, LINEAGE_SOURCE_COLUMNS, MENSTRUAL_PHASE, PHASE_SOURCE_COLUMNS,
        merge_columns,
    )

    meta["cell family"] = merge_columns(meta, LINEAGE_SOURCE_COLUMNS, LINEAGE)
    meta["menstrual phase"] = merge_columns(meta, PHASE_SOURCE_COLUMNS,
                                            MENSTRUAL_PHASE)
    for name, harmoniser, columns in (
        ("cell family", LINEAGE, LINEAGE_SOURCE_COLUMNS),
        ("menstrual phase", MENSTRUAL_PHASE, PHASE_SOURCE_COLUMNS),
    ):
        _log_harmonised(meta, name, harmoniser, columns, study_id, names)

    # The number that decides whether the cross-donor test can answer anything.
    donors = {str(v) for v, ph in zip(meta["patient_id"],
                                      meta["menstrual phase"])
              if v is not None and ph is not None
              and str(ph).strip().lower() not in ("", "nan", "none", "unknown")}
    logger.info("Donors carrying a menstrual phase label: %d", len(donors))
    rows: list[dict] = []
    cross_donor_results: dict[str, dict] = {}

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
        best = _best_over_dims(periodic_corr(mu_angle[ok, d], day[ok], args.cycle_period)
                               for d in range(mu_angle.shape[1]))
        logger.info("   best circular dimension: r = %.3f", best)

    # ── 1b. the OTHER cycle, and the one this corpus can resolve ────────────
    # Menstrual cycle day is a DONOR-level variable here: each patient was
    # biopsied once, so all of their cells share one cycle day and a per-cell
    # encoder has no within-sample variation to represent. The cell cycle is
    # not like that -- G1 -> S -> G2M -> G1 runs inside every proliferating
    # cell, varies within a single sample, and returns to where it started.
    # It has only ever been tested here as three flat categories, which throws
    # away the ordering and the wrap that make it a test of a circular latent.
    from endo_model.evaluation.label_harmonisation import (
        CELL_CYCLE, CELL_CYCLE_PERIOD_HOURS, CELL_CYCLE_SOURCE_COLUMNS,
        cell_cycle_angle,
    )

    logger.info("── 1b. cell-cycle phase against the circular latent")
    cc_names = merge_columns(meta, CELL_CYCLE_SOURCE_COLUMNS, CELL_CYCLE)
    meta["cell cycle phase"] = cc_names
    if cc_names.size:
        _log_harmonised(meta, "cell cycle phase", CELL_CYCLE,
                        CELL_CYCLE_SOURCE_COLUMNS, study_id, names)
        cc_hours = cell_cycle_angle(cc_names) * CELL_CYCLE_PERIOD_HOURS / (2 * np.pi)
        ok = np.isfinite(cc_hours)
        if ok.sum() >= 30:
            for d in range(mu_angle.shape[1]):
                r = periodic_corr(mu_angle[ok, d], cc_hours[ok],
                                  CELL_CYCLE_PERIOD_HOURS)
                logger.info("   cell cycle vs circular dim %d: r = %.3f  (n=%d)",
                            d, r, int(ok.sum()))
                record("periodic", "cell cycle phase", f"circular dim {d}",
                       int(ok.sum()), r)
            # Against the Gaussian half too. The circular branch's claim is
            # that it represents periodic structure BETTER, and that is a
            # comparison rather than an absolute.
            rg = _best_over_dims(circular_linear_corr(mu_angle[ok, d], cc_hours[ok])
                                 for d in range(mu_angle.shape[1]))
            logger.info("   circular-linear (best circular dim): r = %.3f", rg)
        else:
            logger.info("   only %d labelled cells — skipped", int(ok.sum()))

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
        rc = _best_over_dims(circular_linear_corr(mu_angle[ok, d], x[ok])
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
    if circ.shape[1] == 0:
        # A Gaussian-only model has no circular half to score, and its
        # combined latent IS its Gaussian one; scoring both would print the
        # same numbers twice under different names.
        del branches["circular"]
    logger.info("── 3. biological labels — kNN agreement, per latent half")
    logger.info("   (branch widths: gaussian %d, circular %d, combined %d)",
                mu_gauss.shape[1], circ.shape[1], branches["combined"].shape[1])
    dims = args.dims or None
    # The harmonised columns first: they span two studies, so they carry the
    # most cells and the only cross-study evidence on this page.
    for col in HARMONISED + BIOLOGY_LABELS + ["patient_id"]:
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
    for col in HARMONISED + ["EndometriosisGrade", "Endometriosis_stage",
                             "phase"]:
        cross_donor_results.setdefault(col, {})
        for bname, lat in branches.items():
            res = cross_group_agreement(lat, meta[col], meta["patient_id"],
                                        k=max(args.k, 100),
                                        n_permutations=args.permutations)
            if not np.isfinite(res.get("agreement", float("nan"))):
                break
            p_val = res.get("p_value", float("nan"))
            floor = res.get("p_floor", float("nan"))
            if not np.isfinite(p_val):
                verdict = ""
            elif p_val <= 0.05:
                verdict = "  SIGNIFICANT"
            elif np.isfinite(floor) and floor > 0.05:
                # Significance was never reachable: with few donors there are
                # too few ways to deal the labels out for the permutation test
                # to produce a small p. Not the same as "no signal".
                verdict = f"  UNDETECTABLE (p could not fall below {floor:.2f})"
            else:
                verdict = "  not significant"
            logger.info(
                "   %-9s %-26s n=%6d  excess=%6.1f%%  null95=%.2f  p=%.3f  "
                "(%.0f of %d neighbours kept, %d donors)%s",
                bname, col, res["n_cells"], 100 * res["excess"],
                res.get("null_p95", float("nan")), p_val,
                res["neighbours_kept"], max(args.k, 100),
                res.get("n_groups", 0), verdict,
            )
            cross_donor_results[col][bname] = res
            record("cross-donor", col, bname, res["n_cells"], res["excess"],
                   f"p={p_val:.3f}; floor={floor:.3f}; "
                   f"null95={res.get('null_p95', float('nan')):.2f}; "
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

    # ── 6. donor-level view, and a phase-coloured picture ───────────────────
    # Descriptive, not statistical: with few donors this carries exactly the
    # same information as section 4 and hits the same floor. Worth having
    # because "the proliferative donors sit here" is reportable even when the
    # significance test cannot run.
    _describe_donor_layout(meta, mu_gauss, circ, names, study_id, run_dir,
                           args.splits)

    out = Path(args.out) if args.out else run_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test", "label", "scope",
                                          "n_cells", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    logger.info("Written to %s", path)

    if args.no_figures:
        return

    # ── 7. the figures ───────────────────────────────────────────────────────
    # Into `out`, not into the run directory: two runs over different splits
    # would otherwise overwrite each other's pictures with no warning.
    from endo_model.evaluation.latent_bundle import LatentBundle
    from endo_model.viz import apply_theme
    from endo_model.viz.validation_figures import ValidationFigureWriter

    logger.info("── 7. figures")
    apply_theme()
    bundle = LatentBundle(
        mu_gauss=mu_gauss, mu_angle=mu_angle, kappa=kappa,
        study_id=study_id, split=split_of, meta=meta, study_names=names,
        baseline=baseline, baseline_rows=baseline_rows,
    )
    writer = ValidationFigureWriter(
        bundle, out / "figures", k=args.k, seed=exp_cfg.seed,
        umap_cells=args.umap_cells, cycle_period=args.cycle_period,
        topology_cells=args.topology_cells,
    )
    writer.write_all()
    # Drawn from the cross-donor measurements already made above, rather than
    # recomputing a permutation null that took most of this run's time.
    try:
        from endo_model.viz import save_figure
        save_figure(writer.cross_donor(cross_donor_results),
                    out / "figures" / "05_cross_donor")
        logger.info("  05_cross_donor")
    except Exception as exc:                                   # noqa: BLE001
        logger.warning("  05_cross_donor skipped (%s: %s)",
                       type(exc).__name__, exc)


def _describe_donor_layout(meta, mu_gauss, circ, names, study_id, run_dir,
                           splits) -> None:
    """Where each donor sits, and a UMAP coloured by phase.

    Two things the significance test cannot give: a description of the layout
    that stands even when p is uninformative, and a picture. The picture is
    illustration, not evidence -- UMAP flattens the latent to two dimensions and
    should not decide anything. With few donors it also shows mostly DONORS
    rather than phases, which is worth remembering while looking at it.
    """
    import numpy as np

    from endo_model.evaluation.latent_validation import _is_labelled

    phase = meta["MenstrualCyclePhase_main"]
    pid = meta["patient_id"]
    ok = np.array([_is_labelled(v) for v in phase])
    if ok.sum() < 50:
        return

    logger.info("── 6. donor-level layout (descriptive)")
    donors = {}
    for d in np.unique(np.asarray(pid)[ok].astype(str)):
        rows = np.flatnonzero(ok & (np.asarray(pid).astype(str) == d))
        donors[d] = (str(phase[rows[0]]), len(rows), mu_gauss[rows].mean(axis=0))
    for ph in sorted({v[0] for v in donors.values()}):
        members = [(d, n) for d, (p_, n, _) in donors.items() if p_ == ph]
        logger.info("   %-16s %d donors, %s cells",
                    ph, len(members), f"{sum(n for _, n in members):,}")

    # Between-phase separation of donor centroids, against within-phase spread.
    # >1 means donors of one phase sit closer to each other than to the other
    # phase. Descriptive only -- no p-value is possible at this donor count.
    groups = {}
    for d, (ph, _, c) in donors.items():
        groups.setdefault(ph, []).append(c)
    if len(groups) == 2:
        (a, b) = [np.stack(v) for v in groups.values()]
        within = float(np.mean([np.linalg.norm(x - y)
                                for g in (a, b) for x in g for y in g
                                if not np.allclose(x, y)]) or 0.0)
        between = float(np.mean([np.linalg.norm(x - y) for x in a for y in b]))
        logger.info("   donor centroids: between-phase %.3f, within-phase %.3f, "
                    "ratio %.2f  (>1 = phases separate; %d donors, no p-value "
                    "possible at this count)",
                    between, within, between / within if within else float("nan"),
                    len(donors))

    try:
        _plot_phase_umap(mu_gauss[ok], circ[ok], np.asarray(phase)[ok],
                         run_dir, splits)
    except Exception as exc:                                   # noqa: BLE001
        logger.warning("   phase UMAP skipped (%s: %s)", type(exc).__name__, exc)


def _plot_phase_umap(gauss, circ, phase, run_dir, splits) -> None:
    """Both halves of the latent, coloured by menstrual phase."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from endo_model.viz.latent_panels import safe_umap

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, lat, title in ((axes[0], gauss, "Gaussian half"),
                           (axes[1], circ, "circular half")):
        coords = safe_umap(lat, seed=24)
        if coords is None:      # umap absent, or too few cells to embed
            ax.text(0.5, 0.5, "UMAP unavailable", ha="center", va="center")
            ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
            continue
        for ph in np.unique(phase):
            m = phase == ph
            ax.scatter(coords[m, 0], coords[m, 1], s=2, alpha=0.4,
                       label=f"{ph} ({m.sum():,})")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=6, fontsize=8, loc="best")
    n_donors = "descriptive only — with few donors this shows donors, not phases"
    fig.suptitle(f"Menstrual phase over the latent — split(s): "
                 f"{', '.join(splits)}\n{n_donors}")
    fig.tight_layout()
    out = Path(run_dir) / "figures" / "phase_umap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("   phase UMAP written to %s", out)


def _log_harmonised(meta, name: str, harmoniser, columns, study_id,
                    names) -> None:
    """What the merge produced, and what it threw away.

    The dropped values matter as much as the kept ones. A study whose phase
    column is full of unanticipated spellings would silently contribute nothing
    and look identical, in a cell count, to a study with no phase column at all.
    """
    import numpy as np

    from endo_model.evaluation.latent_validation import _is_labelled

    values = meta[name]
    ok = np.array([_is_labelled(v) for v in values])
    kept, counts = np.unique(np.asarray(values)[ok].astype(str),
                             return_counts=True)
    logger.info("Harmonised %-16s %6d cells, %d values: %s",
                name, int(ok.sum()), len(kept),
                ", ".join(f"{k} ({c:,})" for k, c in
                          sorted(zip(kept, counts), key=lambda p: -p[1])))
    _log_contributors(values, study_id, names, name)

    for column in columns:
        if column not in meta:
            continue
        missed = harmoniser.unmapped(meta[column])
        if missed:
            logger.warning(
                "   %s: %d values in %s matched no rule and were dropped: %s",
                name, sum(missed.values()), column,
                ", ".join(f"{k} ({v:,})" for k, v in
                          sorted(missed.items(), key=lambda p: -p[1])[:6]),
            )


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


def _best_over_dims(values) -> float:
    """The largest of a per-dimension statistic, or NaN when there are no
    dimensions: a Gaussian-only model has no circular dimension to be best."""
    values = list(values)
    return max(values) if values else float("nan")


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
