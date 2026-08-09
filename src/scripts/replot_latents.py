"""Redraw a finished run's latent figures from its checkpoint.

    python -m scripts.replot_latents --config configs/experiment/s1_circ_low.yaml \
        --checkpoint outputs/s1_circ_low/checkpoints/epoch_0016_step_0049147.pt

Runs the validation pass far enough to fill the latent collector, then draws
every latent panel — including ones the original run could not.

Two reasons a finished run's figures can be incomplete:

  1. **Tissue was all zeros.** The four numeric hierarchy columns do not exist
     in any study, so `by tissue compartment` and `by organ` were single-valued
     for every run before configs/data/tissue_hierarchy.csv existed. Replotting
     picks up the UBERON mapping and the panels become real.
  2. **Only the Gaussian branch was plotted.** The circular UMAP was added after
     s1_circ_low ran, and that run is the first with six live circular
     dimensions — so it is exactly the run whose circular panel is worth seeing.

Also prints kappa percentiles per circular dimension. The figure reports a
median and a nats figure computed from a small-kappa approximation that is wrong
above kappa ~ 1; the percentiles here are the raw numbers, with the KL computed
properly from power_spherical_kl_uniform.

Writes into <output_dir>/figures_replot/ so nothing from the original run is
overwritten.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("elaj.replot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="The run's experiment YAML.")
    p.add_argument("--checkpoint", required=True, help="Checkpoint .pt to load.")
    p.add_argument("--root", default=".", help="Project root.")
    p.add_argument(
        "--epoch", type=int, default=None,
        help="Epoch number for the figure titles. Defaults to the checkpoint's.",
    )
    return p.parse_args()


def report_kappa(kappa) -> None:
    """Percentiles and true KL per circular dimension.

    The median alone hides the shape. A dimension whose cells are split between
    kappa ~ 1e-4 and kappa ~ 1000 has a median that describes neither half, and
    that is the distribution a partially-revived branch actually has.
    """
    import numpy as np
    import torch

    from endo_model.model.latent.power_spherical import power_spherical_kl_uniform

    logger.info("Per-dimension kappa (percentiles) and true KL to uniform:")
    logger.info(
        "%4s %10s %10s %10s %10s %12s", "dim", "p10", "p50", "p90", "max", "KL (nats)"
    )
    for d in range(kappa.shape[1]):
        col = kappa[:, d]
        col = col[np.isfinite(col)]
        if not col.size:
            continue
        kl = float(
            power_spherical_kl_uniform(torch.from_numpy(col).float(), dim=2).mean()
        )
        p10, p50, p90 = np.percentile(col, [10, 50, 90])
        logger.info(
            "%4d %10.3e %10.3e %10.3e %10.3e %12.4f", d, p10, p50, p90, col.max(), kl
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    import torch

    from endo_model.configs import load_experiment
    from endo_model.data.collate import make_collate_fn
    from endo_model.data.dataset import EndometriosisDataset
    from endo_model.data.samplers import PerStudyBatchSampler
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.precision import PrecisionPolicy
    from endo_model.training.reporting import EpochReporter
    from endo_model.viz.latent_panels import circular_latent
    from torch.utils.data import DataLoader

    root = args.root
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(args.config, root=root)
    vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))

    studies = [
        {"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
        for s in data_cfg.resolve_studies(root)
    ]
    model_cfg.n_studies = len(studies)
    model_cfg.n_groups = len(vocab.group_names)

    val_dataset = EndometriosisDataset(
        study_configs=studies, vocab=vocab, split="val",
        val_fraction=data_cfg.splits.val_fraction,
        test_fraction=data_cfg.splits.test_fraction,
        seed=data_cfg.splits.seed,
        include_disease_statuses=data_cfg.include_disease_statuses,
        tissue_hierarchy_path=(
            str(Path(root) / data_cfg.tissue_hierarchy_csv)
            if data_cfg.tissue_hierarchy_csv else None
        ),
    )
    loader = DataLoader(
        val_dataset,
        batch_sampler=PerStudyBatchSampler(
            val_dataset.study_ids, train_cfg.batch_size, shuffle=False,
            seed=exp_cfg.seed,
        ),
        collate_fn=make_collate_fn(vocab),
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    epoch = args.epoch if args.epoch is not None else int(payload.get("epoch", 0))
    logger.info("Loaded %s (epoch %d)", args.checkpoint, epoch)

    out_dir = Path(root) / exp_cfg.output_dir
    reporter = EpochReporter(
        out_dir=out_dir / "figures_replot",
        n_studies=len(set(val_dataset.study_ids)),
        id_to_name={s["id"]: s["name"] for s in studies},
        latent_every=1,
        max_latent_cells=train_cfg.diagnostics.max_latent_cells,
        seed=exp_cfg.seed,
    )
    reporter.fig_dir = out_dir / "figures_replot"
    collector = reporter.new_collector()
    precision = PrecisionPolicy(train_cfg.precision)

    # Stop as soon as the collector is full. A full validation pass would be
    # 163k cells to fill a 20k subsample.
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()
            }
            with precision.autocast(device):
                output = model(batch)
            collector.add(output, batch)
            if collector.full:
                break

    latents = collector.stacked()
    if latents is None:
        logger.error("Collected no latents — is the validation split empty?")
        return 1
    logger.info("Collected %d cells", latents["mu_gauss"].shape[0])

    report_kappa(latents["kappa"])
    # Only the latent panels — the loss curves belong to the original run
    # and its metrics.csv already has them.
    reporter._draw(
        f"latent_circular_epoch{epoch:03d}",
        lambda: circular_latent(latents['mu_angle'], latents['kappa'], epoch),
    )
    reporter._draw_umap(epoch, latents)
    logger.info("Figures written to %s", out_dir / "figures_replot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
