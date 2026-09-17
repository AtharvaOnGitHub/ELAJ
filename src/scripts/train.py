"""train.py — entry point for a single ELAJ training run.

Usage:
    python scripts/train.py --config configs/experiment/full_model.yaml
    python scripts/train.py --config configs/experiment/full_model.yaml --root /path/to/project

Outputs (written to experiment_config.output_dir/):
    checkpoints/          best checkpoints (CheckpointManager)
    metrics.csv           per-epoch train/val loss breakdown
    run_config.json       snapshot of all resolved configs
"""

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

# Ensure src/ is on sys.path so endo_model is importable when the script is run
# directly.  Python puts the *script's* directory on the path, not the cwd, so
# cd-ing into src/ before running scripts/train.py is not enough.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ELAJ foundation model.")
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML (absolute or relative to --root).",
    )
    parser.add_argument(
        "--root", default=None,
        help="Project root directory (default: current working directory).",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Checkpoint to continue from, relative to --root. Overrides "
             "training.resume_from_checkpoint in the YAML. 'latest' picks the "
             "newest checkpoint in the run's own checkpoints directory.",
    )
    return parser.parse_args()


def resolve_resume(root: str, out_dir: Path, requested: str | None) -> Path | None:
    """Which checkpoint to continue from, or None for a fresh run.

    'latest' is the newest file by epoch in <run>/checkpoints, which is what
    a walltime-killed run wants: the manager keeps the best three by
    val_nb, and the newest of those is the furthest point training reached
    that is worth continuing from.
    """
    if not requested:
        return None
    if requested != "latest":
        return Path(root) / requested
    found = sorted((out_dir / "checkpoints").glob("epoch_*.pt"))
    if not found:
        raise FileNotFoundError(f"--resume-from latest: no checkpoints in "
                                f"{out_dir / 'checkpoints'}")
    return found[-1]


def main() -> None:
    import torch

    args = parse_args()
    root = args.root or str(Path.cwd())

    # ── Load all configs ──────────────────────────────────────────────────────
    from endo_model.configs import load_experiment
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(args.config, root=root)
    logger.info("Experiment: %s", exp_cfg.name or args.config)

    # ── Seed everything ───────────────────────────────────────────────────────
    from endo_model.utils.seeding import set_seed
    set_seed(exp_cfg.seed)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(root) / exp_cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"

    # ── Vocabulary ────────────────────────────────────────────────────────────
    from endo_model.data.vocabulary import GeneVocabulary
    vocab_path = Path(root) / data_cfg.vocabulary_path
    logger.info("Loading vocabulary from %s", vocab_path)
    vocab = GeneVocabulary(str(vocab_path))
    logger.info("Vocabulary: %d genes across %d groups", vocab.vocab_size, len(vocab.group_names))

    # ── Study configs ─────────────────────────────────────────────────────────
    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
        for s in data_cfg.resolve_studies(root)
    ]
    if not study_configs:
        logger.error(
            "No studies configured. Set 'studies' or 'studies_csv' in %s",
            data_cfg.vocabulary_path,
        )
        sys.exit(1)

    # Patch model config with runtime-resolved values
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)

    # ── Datasets ──────────────────────────────────────────────────────────────
    from endo_model.data.dataset import EndometriosisDataset
    split_kwargs = dict(
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
    # One read of the corpus for both splits (findings C3: it used to be
    # read once per split, two full passes before the first step).
    datasets = EndometriosisDataset.load_splits(("train", "val"), **split_kwargs)
    train_dataset, val_dataset = datasets["train"], datasets["val"]
    del datasets
    logger.info(
        "Dataset: %d train cells, %d val cells",
        len(train_dataset), len(val_dataset),
    )

    # ── Device ────────────────────────────────────────────────────────────────
    # Resolved before the DataLoaders so loader settings can depend on it.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    from endo_model.data.collate import make_collate_fn
    from endo_model.data.samplers import PerStudyBatchSampler
    from endo_model.utils.resources import available_cpus, dataloader_workers

    collate = make_collate_fn(vocab)
    train_sampler = PerStudyBatchSampler(
        train_dataset.study_ids, train_cfg.batch_size,
        shuffle=True, seed=exp_cfg.seed,
    )
    val_sampler = PerStudyBatchSampler(
        val_dataset.study_ids, train_cfg.batch_size,
        shuffle=False, seed=exp_cfg.seed,
    )

    # StudyPanel keeps counts sparse and rebuilds a dense (vocab_size,) vector
    # per cell in __getitem__, so collation is CPU-heavy.  With num_workers=0
    # that work serialises against the training step and the GPU waits for it.
    n_workers = dataloader_workers(train_cfg.num_workers)
    logger.info(
        "DataLoader workers: %d (of %d cores available)", n_workers, available_cpus()
    )
    loader_kwargs = dict(
        collate_fn=collate,
        num_workers=n_workers,
        # pin_memory stays off: collate_fn builds group_indices with .expand(),
        # a stride-0 view whose rows all alias one row of memory, and pinning
        # rejects that ("more than one element of the written-to tensor refers
        # to a single memory location").  Making it contiguous would cost
        # B x vocab_size x 8 = ~16 MB of duplicated indices per batch, more than
        # the real payload, so the fix is to drop the redundant batch dimension
        # from the schema rather than to materialise it.  See docs/walkthrough.
        pin_memory=False,
        # Workers hold the dataset, whose per-study panels are the expensive
        # part to re-fork; keeping them alive avoids paying that every epoch.
        persistent_workers=n_workers > 0,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    from endo_model.model.endo_model import EndoFoundationModel

    model = EndoFoundationModel(vocab, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ── Composite loss ────────────────────────────────────────────────────────
    from endo_model.model.objectives.composite import CompositeLoss
    composite_loss = CompositeLoss(
        n_studies=model_cfg.n_studies,
        d_gauss=model_cfg.d_gauss,
        n_circular=model_cfg.n_circular,
        w_cce=model_cfg.cce.lambda_cce if model_cfg.cce.enabled else 0.0,
        lambda_dab_max=model_cfg.dab.lambda_dab,
        cce_temperature=model_cfg.cce.temperature,
        cce_block_normalise=model_cfg.cce.block_normalise,
        cce_circular_weight=model_cfg.cce.circular_weight,
        dab_hidden_dim=model_cfg.dab.hidden_dim,
        w_nb=model_cfg.loss.w_nb,
        w_kl_gauss=model_cfg.loss.w_kl_gauss,
        w_kl_circ=model_cfg.loss.w_kl_circ,
        w_dab=model_cfg.loss.w_dab,
        free_bits=model_cfg.loss.free_bits,
        beta_max=train_cfg.kl_annealing.beta_max,
        warmup_fraction=train_cfg.kl_annealing.warmup_fraction,
    ).to(device)
    logger.info(
        "Loss: w_nb=%.3g w_kl_gauss=%.3g w_kl_circ=%.3g w_dab=%.3g w_cce=%.3g "
        "free_bits=%.3g beta_max=%.3g warmup=%.3g lambda_dab=%.3g",
        model_cfg.loss.w_nb, model_cfg.loss.w_kl_gauss, model_cfg.loss.w_kl_circ,
        model_cfg.loss.w_dab,
        model_cfg.cce.lambda_cce if model_cfg.cce.enabled else 0.0,
        model_cfg.loss.free_bits, train_cfg.kl_annealing.beta_max,
        train_cfg.kl_annealing.warmup_fraction, model_cfg.dab.lambda_dab,
    )
    if model_cfg.cce.enabled:
        # Stated in the log because the embedding layout decides whether the
        # circular branch survives CCE, and the two arms of wave R6 differ in
        # nothing else visible at this point.
        logger.info(
            "CCE: temperature=%.3g block_normalise=%s circular_weight=%.3g",
            model_cfg.cce.temperature, model_cfg.cce.block_normalise,
            model_cfg.cce.circular_weight,
        )

    # ── Optimizer and scheduler, by their config names ───────────────────────
    from endo_model.training.factories import build_optimizer, build_scheduler

    all_params = list(model.parameters()) + list(composite_loss.parameters())
    optimizer = build_optimizer(all_params, train_cfg.optimizer)
    scheduler = build_scheduler(train_cfg.scheduler.name, optimizer,
                                train_cfg.max_epochs)
    logger.info("Optimizer %s (lr=%.3g), scheduler %s over %d epochs",
                train_cfg.optimizer.name, train_cfg.optimizer.lr,
                train_cfg.scheduler.name, train_cfg.max_epochs)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    # Weights and optimizer here; everything else (global step, LR schedule,
    # early stopping, history) once the Trainer exists, in Trainer.restore.
    resume_payload = None
    resume_path = resolve_resume(
        root, out_dir, args.resume_from or train_cfg.resume_from_checkpoint
    )
    if resume_path is not None:
        from endo_model.training.checkpointing import CheckpointManager
        logger.info("Resuming from %s", resume_path)
        resume_payload = CheckpointManager.load(
            str(resume_path), model, optimizer, map_location=str(device)
        )

    # ── Baseline ──────────────────────────────────────────────────────────────
    # Fitted before training, once, over the training split. Without it a
    # reconstruction loss is uninterpretable: run 49817807 reported nb = 0.43
    # for eleven epochs and nothing in the log could say whether that beat
    # predicting a constant. It did not.
    from endo_model.evaluation.baseline import MeanProfileBaseline

    logger.info("Fitting mean-profile baseline over the training split ...")
    baseline = MeanProfileBaseline(
        n_studies=model_cfg.n_studies, vocab_size=vocab.vocab_size
    ).fit(train_loader)
    baseline_scores = baseline.score(val_loader)
    baseline.save(out_dir / "baseline.json", baseline_scores)
    logger.info(
        "Baseline (study mean profile): nb=%.4f all genes, nb=%.4f over %d HVGs",
        baseline_scores.nb_all_genes, baseline_scores.nb_hvg, baseline_scores.n_hvg,
    )

    # The model scored on the same gene set as the baseline's HVG number, so
    # the two curves on that panel are comparable rather than merely adjacent.
    from endo_model.evaluation.metrics import HvgReconstructionMetric
    hvg_metric = HvgReconstructionMetric(baseline.hvg_mask)

    # Reported at the end of every validation pass rather than left in
    # metrics.csv.  Two complete runs finished before anyone noticed neither had
    # ever beaten this number — see evaluation/baseline_gate.py.
    from endo_model.evaluation.baseline_gate import BaselineGate
    baseline_gate = BaselineGate(
        baseline_all=baseline_scores.nb_all_genes,
        baseline_hvg=baseline_scores.nb_hvg or None,
    )

    # ── Figures ───────────────────────────────────────────────────────────────
    from endo_model.training.reporting import EpochReporter

    # Studies that actually contribute cells, not studies configured.  The
    # adversary's chance line is ln(n), and a study the disease filter emptied
    # is one the classifier can never be asked about — including it moves the
    # line and makes a correctly-at-chance adversary read as below chance.
    # With 12 configured and 8 populated that is ln(12)=2.485 against
    # ln(8)=2.079, and run kl_check settled at 2.084.
    n_live_studies = len(set(train_dataset.study_ids))
    if n_live_studies != model_cfg.n_studies:
        logger.info(
            "Adversary chance line: ln(%d) = %.3f — %d of %d studies contribute "
            "cells after filtering",
            n_live_studies, math.log(max(n_live_studies, 1)),
            n_live_studies, model_cfg.n_studies,
        )

    reporter = EpochReporter(
        out_dir=out_dir,
        n_studies=n_live_studies,
        id_to_name={s["id"]: s["name"] for s in study_configs},
        baseline_all=baseline_scores.nb_all_genes,
        baseline_hvg=baseline_scores.nb_hvg,
        latent_every=train_cfg.diagnostics.latent_every,
        max_latent_cells=train_cfg.diagnostics.max_latent_cells,
        seed=exp_cfg.seed,
        tissue_names=_tissue_level_names(root, data_cfg),
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    # Watches every step for non-finite values.  The normal path costs one
    # extra scalar check per step; the expensive per-module instrumentation is
    # installed only to replay a step that has already failed, which is what
    # turns "a distribution rejected a NaN" into "this module made it".
    from endo_model.training.diagnostics import NonFiniteFault, TrainingMonitor
    diag_cfg = train_cfg.diagnostics
    monitor = TrainingMonitor(
        model=model,
        extra_modules=[composite_loss],
        log_every=diag_cfg.log_every,
        detect_anomaly=diag_cfg.detect_anomaly,
        dump_dir=(out_dir / "faults") if diag_cfg.dump_faults else None,
        id_to_name={s["id"]: s["name"] for s in study_configs},
        skip_nonfinite_steps=diag_cfg.skip_nonfinite_steps,
        max_consecutive_skips=diag_cfg.max_consecutive_skips,
    )

    # ── Precision ─────────────────────────────────────────────────────────────
    from endo_model.precision import PrecisionPolicy
    precision = PrecisionPolicy(train_cfg.precision)
    logger.info("Precision: %s", precision.describe())

    # ── Trainer ───────────────────────────────────────────────────────────────
    from endo_model.training.trainer import Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        composite_loss=composite_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        config=train_cfg,
        checkpoint_dir=str(ckpt_dir),
        device=device,
        cce_enabled=model_cfg.cce.enabled,
        cce_every=model_cfg.cce.every,
        monitor=monitor,
        reporter=reporter,
        extra_val_metrics=hvg_metric,
        precision=precision,
        baseline_gate=baseline_gate,
        # Rewrite metrics.csv every epoch, so a walltime kill leaves the
        # history behind rather than only the checkpoints.
        on_epoch_metrics=lambda history: _write_metrics_csv(
            out_dir / "metrics.csv", history
        ),
    )
    if resume_payload is not None:
        trainer.restore(resume_payload)

    # ── Save run config snapshot ──────────────────────────────────────────────
    _save_run_config(out_dir, exp_cfg, model_cfg, train_cfg, data_cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training — max_epochs=%d", train_cfg.max_epochs)
    try:
        with monitor:
            history = trainer.fit()
    except NonFiniteFault:
        # The report is already logged and written; re-raising it verbatim
        # would bury it under a traceback pointing at this line, which is not
        # where anything went wrong.
        logger.error(
            "Training stopped on a non-finite value. Report and failing batch "
            "are in %s", out_dir / "faults",
        )
        sys.exit(1)

    # ── Write metrics CSV ─────────────────────────────────────────────────────
    _write_metrics_csv(out_dir / "metrics.csv", history)
    logger.info("%s", baseline_gate.summarise())
    if monitor.skipped_steps:
        logger.warning(
            "%d update(s) were skipped for non-finite gradients over the run.",
            monitor.skipped_steps,
        )
    # ── Final figures from the best checkpoint ───────────────────────────────
    # The per-epoch figures land on `latent_every`, which is almost never the
    # epoch early stopping actually chose: r3_capacity's best was 27 and its
    # last figure is 28, r3_loose's best was 21 and its last is 24. Redraw them
    # once from the checkpoint that was selected.
    from endo_model.training.final_report import FinalReporter

    FinalReporter(
        reporter=reporter, loader=val_loader, device=device, out_dir=out_dir,
        # So the per-study table names studies rather than numbering them.
        study_names={s["id"]: s["name"] for s in study_configs},
    ).render(model, trainer.ckpt_manager)

    logger.info("Training complete. Outputs written to %s", out_dir)


def _tissue_level_names(root: str, data_cfg) -> dict[int, dict[int, str]]:
    """Level column index -> {code -> label}, for the tissue UMAP legends.

    Returns empty on any failure. These names are decoration on a diagnostic
    figure; a missing or malformed CSV must not stop a training run that is
    otherwise ready to go.
    """
    if not data_cfg.tissue_level_names_csv:
        return {}
    from endo_model.data.tissue_level_names import LEVEL_ORDER, TissueLevelNames

    try:
        names = TissueLevelNames(Path(root) / data_cfg.tissue_level_names_csv)
    except Exception as exc:  # noqa: BLE001 — legends are not load-bearing
        logger.warning(
            "Tissue level names unavailable (%s: %s); tissue legends will show "
            "integer codes.", type(exc).__name__, exc,
        )
        return {}
    return {i: names.for_index(i) for i in range(len(LEVEL_ORDER))}


def _save_run_config(out_dir: Path, exp_cfg, model_cfg, train_cfg, data_cfg) -> None:
    import dataclasses
    payload = {
        "experiment": dataclasses.asdict(exp_cfg),
        "model": dataclasses.asdict(model_cfg),
        "training": dataclasses.asdict(train_cfg),
        "data": dataclasses.asdict(data_cfg),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _write_metrics_csv(path: Path, history: dict) -> None:
    max_epochs = max(len(v) for v in history.values()) if history else 0
    if max_epochs == 0:
        return
    fieldnames = ["epoch"] + list(history.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(max_epochs):
            row: dict = {"epoch": epoch}
            for k, v in history.items():
                row[k] = v[epoch] if epoch < len(v) else ""
            writer.writerow(row)


if __name__ == "__main__":
    main()
