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


def build_loader(root, exp_cfg, train_cfg, data_cfg, vocab, study_configs):
    from torch.utils.data import DataLoader

    from endo_model.data.collate import make_collate_fn
    from endo_model.data.dataset import EndometriosisDataset
    from endo_model.data.samplers import PerStudyBatchSampler

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
            logger.error("  %s has no metrics.csv — run did not finish; skipped", run_dir)
            continue

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


if __name__ == "__main__":
    main()
