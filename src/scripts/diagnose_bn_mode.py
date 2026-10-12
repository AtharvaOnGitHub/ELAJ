"""Does the study separation in the UMAPs come from a BatchNorm mismatch?

The question
-----------
`DSBN` (endo_model/model/embeddings/dsbn.py) is a normalisation followed by a
per-study scale and shift, and a normalisation with running statistics behaves
differently in the two modes PyTorch gives it:

  training mode — normalise using the statistics of the batch in front of it
  eval mode     — normalise using the accumulated `running_mean` /
                  `running_var`

Every figure in every run is drawn in eval mode, so if the two disagree, the
figures are not showing what the model trained under.

WHAT THE ANSWER WAS, AND WHY IT MOVED. When DSBN held one shared
`BatchNorm1d`, eval mode pooled every study into a single average that batch
counts made essentially the largest studies', and the cost ran from +0.0052
for the study holding 35.5% of validation to +0.2770 for the one holding 1.0%.
`PerStudyBatchNorm` closed that by giving each study its own statistics, so on
a current checkpoint this script is asking whether anything is LEFT, not
whether the original bug is there.

It is still worth asking, because the per-study estimates are an exponential
moving average at momentum 0.1 updated once per batch of that study, and a
study with few batches keeps an estimate still part-way from its
initialisation. `num_batches_tracked` says which studies those are.

A NOTE ON WHAT IS MATCHED. `PerStudyBatchNorm` is not a `_BatchNorm` subclass,
and for a while this script's selector was an isinstance check against
`_BatchNorm` alone. It matched nothing, toggled nothing, and reported +0.000.
See `normalisation_layers`, which now refuses to run on a model where it finds
no such layer.

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


def normalisation_layers(model) -> list:
    """Every layer whose behaviour differs between training and eval mode.

    `PerStudyBatchNorm` IS ONE OF THEM AND IS NOT A `_BatchNorm`. It keeps its
    statistics as one buffer indexed by study rather than as N child
    `BatchNorm1d` modules, which is the whole point of the class, so
    `isinstance(m, nn.modules.batchnorm._BatchNorm)` does not match it. On
    12 October 2026 this function was that isinstance check alone, it matched
    nothing on m6_drain, and the script reported a difference of +0.000 — not
    the finding it reads as, but a computation compared with itself.

    `nn.LayerNorm` is deliberately absent: it holds no running statistics and
    behaves identically in both modes, so toggling it would change nothing and
    inflate the count this function is trusted for.
    """
    import torch.nn as nn

    from endo_model.model.embeddings.per_study_batch_norm import PerStudyBatchNorm

    kinds = (nn.modules.batchnorm._BatchNorm, PerStudyBatchNorm)
    return [m for m in model.modules() if isinstance(m, kinds)]


def set_batchnorm_training(model, training: bool) -> int:
    """Put every normalisation layer into the given mode, leaving the rest alone.

    Dropout must stay off and the latent branches must stay deterministic, so
    `model.train()` is not what is wanted — that would change three things at
    once and the comparison would mean nothing.

    Raises:
        RuntimeError: if the model holds no such layer. The entire question
            this script asks is what those layers do between modes; with none
            found there is nothing to ask, and printing +0.000 would answer it
            falsely.
    """
    layers = normalisation_layers(model)
    if not layers:
        raise RuntimeError(
            "no BatchNorm or PerStudyBatchNorm layer in this model, so there "
            "is nothing to toggle and the two passes would be identical. "
            "Either the architecture normalises some other way, in which case "
            "normalisation_layers() needs to learn about it, or this config "
            "has no normalisation at all and the question does not apply."
        )
    for module in layers:
        module.train(training)
    return len(layers)


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
