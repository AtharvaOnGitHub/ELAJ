"""Does the study separation in the UMAPs come from a BatchNorm mismatch?

The question
-----------
`DSBN` (endo_model/model/embeddings/dsbn.py) is one shared `BatchNorm1d`
followed by a per-study scale and shift. BatchNorm behaves differently in the
two modes PyTorch gives it:

  training mode — normalise using the statistics of the batch in front of it
  eval mode     — normalise using `running_mean` / `running_var`, one set of
                  numbers accumulated over the whole run

Batches in this project hold exactly one study. So in training mode each study
is normalised by its own statistics, and the per-study scale and shift are
learned against that. In eval mode every study is instead normalised by one
corpus-wide average — and then corrected by a scale and shift calibrated for
something else.

Every figure in every run so far is drawn in eval mode.

The test
--------
Collect the same validation cells twice from the same checkpoint: once in eval
mode, once with only the BatchNorm layers put back into training mode. Because
batches are single-study, the second pass reproduces training-time
normalisation exactly.

Then compare, on both:

  kNN study purity — the share of a cell's nearest neighbours that come from
                     its own study, which is what "are the studies mixed?"
                     actually means
  a UMAP           — the picture, side by side

If purity drops substantially in training mode, the separation we have been
reading off these figures is largely this bug. If it does not move, the bug is
real but small, and the separation is the model's.

Usage
-----
    python -m scripts.diagnose_bn_mode --config configs/experiment/r3_nodab.yaml \\
        --checkpoint outputs/r3_nodab/checkpoints/<best>.pt --root .
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.bn_mode")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument(
        "--checkpoint", default="auto",
        help="Path to a .pt, or 'auto' to pick the retained checkpoint whose "
             "epoch has the lowest val_nb in the run's metrics.csv.",
    )
    p.add_argument("--root", default=None)
    p.add_argument("--k", type=int, default=30, help="Neighbours for purity.")
    p.add_argument("--out", default=None, help="Figure directory.")
    return p.parse_args()


def resolve_checkpoint(out_dir: Path) -> Path:
    """The retained checkpoint at the best-val_nb epoch.

    CheckpointManager keeps the best three and encodes the epoch in the
    filename as `epoch_XXXX_step_YYYYYYY.pt`. metrics.csv holds val_nb per
    epoch, so the two together identify the one to load without the caller
    having to read a directory listing and do the join by hand.
    """
    import csv

    ckpt_dir = out_dir / "checkpoints"
    available = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not available:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")

    rows = list(csv.DictReader((out_dir / "metrics.csv").open()))
    best = min(rows, key=lambda r: float(r["val_nb"]))
    want = f"epoch_{int(best['epoch']):04d}_"

    for path in available:
        if path.name.startswith(want):
            logger.info(
                "Best val_nb %.5f at epoch %s -> %s",
                float(best["val_nb"]), best["epoch"], path.name,
            )
            return path

    # The best epoch's checkpoint can have been evicted: retention keeps three
    # and ranks by val_nb, so this should not happen, but a resumed run starts
    # with an empty in-memory history and can prune inconsistently.
    fallback = available[-1]
    logger.warning(
        "No checkpoint for best epoch %s (kept: %s). Using %s instead.",
        best["epoch"], ", ".join(p.name for p in available), fallback.name,
    )
    return fallback


def set_batchnorm_training(model, training: bool) -> int:
    """Put every BatchNorm layer into the given mode, leaving the rest alone.

    Dropout must stay off and the latent branches must stay deterministic, so
    `model.train()` is not what is wanted — that would change three things at
    once and the comparison would mean nothing.
    """
    import torch.nn as nn

    n = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train(training)
            n += 1
    return n


def main() -> None:
    import torch

    args = parse_args()
    root = args.root or str(Path.cwd())

    from endo_model.configs import load_experiment
    from endo_model.data.dataset import EndometriosisDataset
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.latent_collection import collect_latents
    from endo_model.evaluation.study_mixing import describe, expected_purity, knn_purity
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager
    from endo_model.training.reporting import EpochReporter

    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(args.config, root=root)
    vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))
    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
        for s in data_cfg.resolve_studies(root)
    ]
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)

    val = EndometriosisDataset(
        split="val",
        study_configs=study_configs,
        vocab=vocab,
        val_fraction=data_cfg.splits.val_fraction,
        test_fraction=data_cfg.splits.test_fraction,
        seed=data_cfg.splits.seed,
        max_cells_per_study=data_cfg.max_cells_per_study,
        include_disease_statuses=data_cfg.include_disease_statuses,
        tissue_hierarchy_path=(
            str(Path(root) / data_cfg.tissue_hierarchy_csv)
            if data_cfg.tissue_hierarchy_csv else None
        ),
        tissue_level_names_path=(
            str(Path(root) / data_cfg.tissue_level_names_csv)
            if data_cfg.tissue_level_names_csv else None
        ),
        study_platform_path=(
            str(Path(root) / data_cfg.study_platform_csv)
            if data_cfg.study_platform_csv else None
        ),
        sample_platform_path=(
            str(Path(root) / data_cfg.sample_platform_csv)
            if data_cfg.sample_platform_csv else None
        ),
    )

    from torch.utils.data import DataLoader
    from endo_model.data.collate import make_collate_fn
    from endo_model.data.samplers import PerStudyBatchSampler

    loader = DataLoader(
        val,
        batch_sampler=PerStudyBatchSampler(
            val.study_ids, train_cfg.batch_size, shuffle=False, seed=exp_cfg.seed
        ),
        collate_fn=make_collate_fn(vocab),
        num_workers=0,
    )

    run_dir = Path(root) / exp_cfg.output_dir
    ckpt = (
        resolve_checkpoint(run_dir) if args.checkpoint == "auto"
        else Path(args.checkpoint)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    CheckpointManager.load(str(ckpt), model, None, map_location=device)
    logger.info("Loaded %s", ckpt)

    out_dir = Path(args.out or (Path(root) / exp_cfg.output_dir / "figures"))
    reporter = EpochReporter(
        out_dir=out_dir.parent, n_studies=len(study_configs),
        id_to_name={s["id"]: s["name"] for s in study_configs},
        seed=exp_cfg.seed,
    )

    results = {}
    for label, bn_training in (("eval", False), ("train", True)):
        model.eval()
        n_bn = set_batchnorm_training(model, bn_training)
        logger.info(
            "Pass '%s': %d BatchNorm layer(s) in %s mode",
            label, n_bn, "training" if bn_training else "eval",
        )
        latents = collect_latents(model, loader, device, reporter.new_collector())
        if latents is None:
            logger.error("Validation loader yielded no batches.")
            sys.exit(1)
        results[label] = latents
        logger.info("  %s", describe(latents["mu_gauss"], latents["study_id"], args.k))

    # ── The comparison ───────────────────────────────────────────────────────
    floor = expected_purity(results["eval"]["study_id"])
    pe = knn_purity(results["eval"]["mu_gauss"], results["eval"]["study_id"], args.k)
    pt = knn_purity(results["train"]["mu_gauss"], results["train"]["study_id"], args.k)

    logger.info("")
    logger.info("=" * 68)
    logger.info("  perfectly mixed would be   %.3f", floor)
    logger.info("  eval-mode BatchNorm        %.3f   (what every figure shows)", pe)
    logger.info("  training-mode BatchNorm    %.3f   (what the model trained under)", pt)
    logger.info("  difference                 %+.3f", pt - pe)
    share = (pe - pt) / (pe - floor) if pe > floor else float("nan")
    logger.info("")
    logger.info(
        "  %.0f%% of the eval-mode separation above the mixed floor disappears "
        "when BatchNorm is put back into training mode.", 100 * share,
    )
    logger.info("=" * 68)

    from endo_model.viz.latent_panels import latent_umap, safe_umap
    from endo_model.viz import apply_theme, save_figure
    import matplotlib.pyplot as plt

    apply_theme()
    for label, latents in results.items():
        coords = safe_umap(latents["mu_gauss"], seed=exp_cfg.seed)
        if coords is None:
            continue
        fig = latent_umap(
            coords,
            {"by study": (latents["study_id"], reporter.id_to_name, None)},
            epoch=-1,
        )
        fig.suptitle(f"BatchNorm in {label} mode — kNN study purity "
                     f"{knn_purity(latents['mu_gauss'], latents['study_id'], args.k):.3f}")
        save_figure(fig, out_dir / f"bn_mode_{label}")
        plt.close(fig)
    logger.info("Figures written to %s", out_dir)


if __name__ == "__main__":
    main()
