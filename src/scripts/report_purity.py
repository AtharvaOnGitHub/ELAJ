"""kNN study purity for one or more finished runs, from their best checkpoints.

    python -m scripts.report_purity --root . \
        --run r4_tiny_latent --run r4_thin_groups --run r4_deep_cell

Each run is named by its experiment config; the script finds that run's output
directory, loads the checkpoint at its best val_nb epoch, embeds the validation
split, and measures how well studies mix. No training, no UMAP.

Several runs go in one invocation on purpose. Rebuilding the validation Dataset
is the expensive part — every .h5ad opened — and runs that share a corpus share
that work, so eight arms in one job cost far less than eight jobs.

Writes purity.csv per run and one combined table to the output directory.
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
logger = logging.getLogger("elaj.purity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True,
                   help="Experiment name, e.g. r4_tiny_latent. Repeatable.")
    p.add_argument("--root", default=None)
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--k", type=int, default=30, help="Neighbours per cell.")
    p.add_argument("--dims", type=int, default=8,
                   help="Project every latent to this many PCs before the "
                        "neighbour search, so runs of different latent width "
                        "are comparable. 0 uses each latent as-is.")
    p.add_argument("--out", default="outputs/purity",
                   help="Where the combined table is written.")
    p.add_argument("--runs-dir", default=None,
                   help="Look for each run's directory under here instead of "
                        "beside the path in its config. The run keeps its own "
                        "folder name, so a wave archived into outputs/r5/ is "
                        "reached with --runs-dir outputs/r5.")
    return p.parse_args()


def resolve_run_dir(root: str, output_dir: str, runs_dir: str | None) -> Path:
    """Where this run's outputs actually live.

    A config records where a run WROTE its outputs, which stops being true the
    moment a finished wave is filed into a subdirectory. `runs_dir` replaces
    the parent and keeps the leaf, so outputs/r5_cce/ under
    --runs-dir outputs/r5 becomes outputs/r5/r5_cce.
    """
    recorded = Path(root) / output_dir
    if runs_dir is None:
        return recorded
    return Path(root) / runs_dir / recorded.name


def corpus_key(data_cfg) -> tuple:
    """What makes two runs share a validation Dataset.

    Only these fields change which cells are in the split, so runs agreeing on
    all of them can reuse one Dataset. Model shape does not appear — that
    changes the encoder, not the data.
    """
    return (
        tuple(sorted(data_cfg.include_disease_statuses)),
        data_cfg.splits.seed,
        data_cfg.splits.val_fraction,
        data_cfg.splits.test_fraction,
        data_cfg.max_cells_per_study,
    )


def build_loader(root, exp_cfg, train_cfg, data_cfg, vocab, study_configs,
                 extra_obs_columns=None, split="val"):
    from torch.utils.data import DataLoader

    from endo_model.data.collate import make_collate_fn
    from endo_model.data.dataset import EndometriosisDataset
    from endo_model.data.samplers import PerStudyBatchSampler

    val = EndometriosisDataset(
        split=split,
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
        # Evaluation-only per-cell metadata. Never reaches the model; see
        # EndometriosisDataset.extra_obs.
        extra_obs_columns=extra_obs_columns,
    )
    return DataLoader(
        val,
        batch_sampler=PerStudyBatchSampler(
            val.study_ids, train_cfg.batch_size, shuffle=False, seed=exp_cfg.seed
        ),
        collate_fn=make_collate_fn(vocab),
        num_workers=0,
    )


def main() -> None:
    import torch

    args = parse_args()
    root = args.root or str(Path.cwd())

    from endo_model.configs import load_experiment
    from endo_model.data.tissue_level_names import TissueLevelNames
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.latent_collection import collect_latents
    from endo_model.evaluation.purity_report import (
        best_checkpoint, format_table, measure, write_csv,
    )
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager
    from endo_model.training.reporting import EpochReporter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders: dict[tuple, object] = {}
    results = []

    for name in args.run:
        logger.info("── %s", name)
        config = Path(root) / args.config_dir / f"{name}.yaml"
        if not config.exists():
            logger.error("  no config at %s — skipped", config)
            continue

        exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(str(config), root=root)
        run_dir = resolve_run_dir(root, exp_cfg.output_dir, args.runs_dir)
        # Two different failures, and conflating them sent a search for a
        # crashed run when the outputs had simply been filed elsewhere.
        if not run_dir.is_dir():
            logger.error(
                "  no directory at %s — the run may have been moved; "
                "point --runs-dir at its parent; skipped", run_dir,
            )
            continue
        if not (run_dir / "metrics.csv").exists():
            # Not fatal any more. best_checkpoint falls back to the val_nb
            # stored in each checkpoint, which is exactly the case a run killed
            # at the walltime leaves behind — and that is when this script is
            # most needed.
            logger.warning(
                "  %s has no metrics.csv — the run did not finish; reading the "
                "best epoch from the checkpoints instead", run_dir,
            )

        vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))
        study_configs = [
            {"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
            for s in data_cfg.resolve_studies(root)
        ]
        model_cfg.n_studies = len(study_configs)
        model_cfg.n_groups = len(vocab.group_names)

        key = corpus_key(data_cfg)
        if key not in loaders:
            logger.info("  building the validation split for this corpus")
            loaders[key] = build_loader(
                root, exp_cfg, train_cfg, data_cfg, vocab, study_configs
            )
        else:
            logger.info("  reusing the validation split from an earlier run")
        loader = loaders[key]

        try:
            ckpt = best_checkpoint(run_dir)
        except FileNotFoundError as exc:
            logger.error("  %s — skipped", exc)
            continue

        model = EndoFoundationModel(vocab, model_cfg).to(device)
        CheckpointManager.load(str(ckpt), model, None, map_location=device)
        model.eval()

        reporter = EpochReporter(
            out_dir=run_dir, n_studies=len(study_configs), seed=exp_cfg.seed
        )
        latents = collect_latents(model, loader, device, reporter.new_collector())
        if latents is None:
            logger.error("  validation loader was empty — skipped")
            continue

        names = {}
        if data_cfg.tissue_level_names_csv:
            names = TissueLevelNames(
                Path(root) / data_cfg.tissue_level_names_csv
            ).for_index(2)          # tissue_type

        found = measure(
            run=name,
            latent=latents["mu_gauss"],
            study_id=latents["study_id"],
            tissue_type=latents["tissue"][:, 2],
            tissue_names=names,
            k=args.k,
            n_components=args.dims or None,
        )
        write_csv(found, run_dir / "purity.csv")
        results.extend(found)
        for r in found:
            logger.info("  %-20s purity %.3f  (mixed %.3f, excess %.1f%%)",
                        r.scope, r.purity, r.floor, 100 * r.excess)

        # The same per-study table FinalReporter writes at the end of training.
        # Repeated here so a run that never reached its final report -- killed
        # at the walltime, most likely -- can still produce it from the best
        # checkpoint, with no retraining.
        _report_per_study(
            latents, run_dir, {s["id"]: s["name"] for s in study_configs},
            k=args.k, n_components=args.dims or None,
        )
        _report_angle_range(latents, model_cfg)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        logger.error("Nothing measured.")
        sys.exit(1)

    out = Path(root) / args.out / "purity.csv"
    write_csv(results, out)
    logger.info("\n%s", format_table(results))
    logger.info("Written to %s and to each run's purity.csv", out)


def _report_angle_range(latents, model_cfg) -> None:
    """Does the circular mean wrap? The question wave R8 turns on.

    Under the "angle" parameterisation the head is one unbounded number per
    dimension, fed to cos/sin, which repeat every 2*pi. If those numbers stay
    inside [-pi, pi] there is no wrapping, the head is behaving, and swapping
    the parameterisation would be aimed at a problem that does not exist. If
    they run well past it, cells are aliasing onto shared angles and gradients
    from similar cells are cancelling.

    Costs nothing: mu_angle is already collected for the circular figures.
    """
    import math

    import numpy as np

    angle = latents.get("mu_angle")
    if angle is None or angle.size == 0:
        return

    absmax = float(np.abs(angle).max())
    outside = float((np.abs(angle) > math.pi).mean())
    verdict = (
        "wrapping — cells alias onto shared angles"
        if outside > 0.01 else
        "no wrapping — the head stays inside one turn"
    )
    logger.info(
        "  circular mean (%s): |angle| max %.2f rad = %.2f turns, "
        "%.1f%% of values outside [-pi, pi] -> %s",
        getattr(model_cfg, "circular_parameterisation", "angle"),
        absmax, absmax / (2 * math.pi), 100 * outside, verdict,
    )


def _report_per_study(latents, run_dir, study_names, k, n_components):
    """Per-study purity for one run: logged, and written beside its purity.csv.

    Wrapped, like the copy in FinalReporter: a diagnostic table must not be the
    reason a recovery job over several runs fails partway through.
    """
    from endo_model.evaluation.study_mixing import (
        format_per_study, per_study_purity,
    )

    try:
        rows = per_study_purity(
            latents["mu_gauss"], latents["study_id"], k=k,
            n_components=n_components,
        )
        if not rows:
            return
        logger.info("  per study:\n%s", format_per_study(rows, study_names))
        path = run_dir / "purity_per_study.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["study_id", "study", "n_cells",
                               "purity", "floor", "excess"],
            )
            writer.writeheader()
            for r in sorted(rows, key=lambda r: r["excess"], reverse=True):
                writer.writerow({**r, "study": study_names.get(r["study_id"], "")})
        logger.info("  per-study purity written to %s", path)
    except Exception as exc:  # noqa: BLE001 — diagnostic must not be fatal
        logger.warning("  per-study purity skipped (%s: %s)",
                       type(exc).__name__, exc)


if __name__ == "__main__":
    main()
