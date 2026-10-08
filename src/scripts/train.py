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
        donor_aliasing=data_cfg.splits.donor_aliasing,
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
    # The validation dataset also carries the cell-family source columns, so
    # the label probe can score the latent on family every epoch. They live in
    # `extra_obs`, outside the Batch; the model never sees them.
    from endo_model.evaluation.label_harmonisation import (
        LINEAGE_SOURCE_COLUMNS, PHASE_SOURCE_COLUMNS,
    )
    # Phase as well as family: phase is the v-series target and family
    # is kept as the control that says the latent is not simply broken.
    split_kwargs["extra_obs_columns"] = list(
        dict.fromkeys([*LINEAGE_SOURCE_COLUMNS, *PHASE_SOURCE_COLUMNS]))

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

    # ── Label probe: cell family on the validation latent, every epoch ──────
    # Labels are aligned to the validation sampler's order, which is fixed
    # (shuffle=False), so the latents collected in `_val_epoch` line up.
    from endo_model.training.label_probe import (
        BlockProbe, CompositeProbe, LabelProbe, family_labels_in_order,
        patients_in_order, phase_labels_in_order, studies_in_order,
    )
    val_order = [i for batch in val_sampler for i in batch]
    # Donor ids as well as labels, so every epoch reports both bounds on the
    # biology (with and without same-donor neighbours), the drop between
    # them, and donor identity itself. See training/label_probe.py.
    patients = patients_in_order(val_dataset, val_order)
    studies = studies_in_order(val_dataset, val_order)
    family_probe = LabelProbe(family_labels_in_order(val_dataset, val_order),
                              k=30, seed=exp_cfg.seed,
                              patients=patients, studies=studies)
    if family_probe.n_labelled == 0:
        logger.warning("No validation cell carries a family label; the probe "
                       "will report NaN and stop_metric falls back to nb")

    # Phase is scored on each latent block separately. Menstrual phase is
    # constant within a donor -- 269 patients, 269 samples -- so the
    # within-donor number is inflated by construction and only the
    # cross-donor one means anything. See label_harmonisation.py.
    phase_probe = BlockProbe(
        phase_labels_in_order(val_dataset, val_order),
        name="phase_knn", every=train_cfg.block_probe_every,
        k=30, seed=exp_cfg.seed, patients=patients, studies=studies)
    if phase_probe.n_labelled == 0:
        logger.warning("No validation cell carries a menstrual phase; the "
                       "phase head will train on nothing and its probe will "
                       "report NaN")
    label_probe = CompositeProbe([family_probe, phase_probe])
    # How far the encoder's output moves per epoch. Reported alongside the
    # losses, and the number that says when the encoder is settled enough
    # to freeze for a latent sweep. See training/drift.py.
    from endo_model.training.drift import EncoderDriftProbe

    drift_probe = EncoderDriftProbe.from_loader(val_loader)

    # Hides a share of each cell's measured genes from the encoder on
    # training steps. The per-study DNS pattern is a study fingerprint
    # the encoder reads for free; see training/gene_masking.py for what
    # this weakens and what it cannot remove.
    from endo_model.training.gene_masking import GeneMasker
    gene_masker = GeneMasker(target_measured=train_cfg.gene_mask_target,
                             seed=exp_cfg.seed)

    if train_cfg.stop_metric != "nb":
        logger.info("Checkpoints ranked and early stopping driven by "
                    "val_%s (negated), not val_nb", train_cfg.stop_metric)

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
        dab_dropout=model_cfg.dab.dropout,
        dab_n_layers=model_cfg.dab.n_layers,
        dab_normalise=model_cfg.dab.normalise,
        w_ecs=model_cfg.loss.w_ecs if model_cfg.ecs.enabled else 0.0,
        ecs_threshold=model_cfg.ecs.threshold,
        ecs_buffer_batches=(model_cfg.ecs.buffer_batches
                            if model_cfg.ecs.enabled else 0),
        w_nb=model_cfg.loss.w_nb,
        w_kl_gauss=model_cfg.loss.w_kl_gauss,
        w_kl_circ=model_cfg.loss.w_kl_circ,
        w_dab=model_cfg.loss.w_dab,
        w_cls=model_cfg.loss.w_cls if model_cfg.classifier.enabled else 0.0,
        cls_label_smoothing=model_cfg.classifier.label_smoothing,
        free_bits=model_cfg.loss.free_bits,
        free_bits_circ=model_cfg.loss.free_bits_circ,
        beta_max=train_cfg.kl_annealing.beta_max,
        warmup_fraction=train_cfg.kl_annealing.warmup_fraction,
        drain_targets=(tuple(model_cfg.drain.targets)
                       if model_cfg.drain.enabled and model_cfg.drain.d_drain
                       else ()),
        w_drain=model_cfg.loss.w_drain if model_cfg.drain.enabled else 0.0,
        w_drain_study=model_cfg.drain.w_study,
        w_drain_depth=model_cfg.drain.w_depth,
        w_drain_genes=model_cfg.drain.w_genes,
        drain_label_smoothing=model_cfg.drain.label_smoothing,
    ).to(device)
    if model_cfg.drain.enabled:
        logger.info(
            "Drain: d_drain=%d targets=%s w_drain=%.3g (study %.3g, depth "
            "%.3g, genes %.3g), blanking cov=%.3g drain=%.3g. The drain "
            "block is read ONLY "
            "by DrainHead; mu_gauss and mu_angle are the biology block and "
            "are what the adversary, the phase head and every exported "
            "embedding see.",
            model_cfg.drain.d_drain, ",".join(model_cfg.drain.targets),
            model_cfg.loss.w_drain, model_cfg.drain.w_study,
            model_cfg.drain.w_depth, model_cfg.drain.w_genes,
            model_cfg.drain.blank_covariates_p,
            model_cfg.drain.blank_drain_p,
        )
    logger.info(
        "Loss: w_nb=%.3g w_kl_gauss=%.3g w_kl_circ=%.3g w_dab=%.3g w_cce=%.3g "
        "free_bits=%.3g beta_max=%.3g warmup=%.3g lambda_dab=%.3g",
        model_cfg.loss.w_nb, model_cfg.loss.w_kl_gauss, model_cfg.loss.w_kl_circ,
        model_cfg.loss.w_dab,
        model_cfg.cce.lambda_cce if model_cfg.cce.enabled else 0.0,
        model_cfg.loss.free_bits, train_cfg.kl_annealing.beta_max,
        train_cfg.kl_annealing.warmup_fraction, model_cfg.dab.lambda_dab,
    )
    logger.info(
        "Adversary: %d Linear layers of %d units, LayerNorm=%s, dropout=%.3g",
        model_cfg.dab.n_layers, model_cfg.dab.hidden_dim,
        model_cfg.dab.normalise, model_cfg.dab.dropout,
    )
    if model_cfg.ecs.enabled:
        logger.info(
            "ECS on: w_ecs=%.4g threshold=%.3g buffer=%d batches. Pairs are "
            "%s. See objectives/elastic_similarity.py for the direction.",
            model_cfg.loss.w_ecs, model_cfg.ecs.threshold,
            model_cfg.ecs.buffer_batches,
            "within-study only" if model_cfg.ecs.buffer_batches == 0
            else "cross-study via the reference buffer",
        )
    if model_cfg.classifier.enabled:
        from endo_model.data.constants import phase_class_names
        logger.info(
            "Phase head: block=%s hidden=%s dropout=%.3g w_cls=%.3g "
            "smoothing=%.3g over %d classes %s",
            model_cfg.classifier.block, model_cfg.classifier.hidden,
            model_cfg.classifier.dropout, model_cfg.loss.w_cls,
            model_cfg.classifier.label_smoothing,
            len(phase_class_names()), list(phase_class_names()),
        )
    else:
        logger.info("Phase head: disabled; phase is reported, not optimised.")
    if model_cfg.cce.enabled:
        # Stated in the log because the embedding layout decides whether the
        # circular branch survives CCE, and the two arms of wave R6 differ in
        # nothing else visible at this point.
        logger.info(
            "CCE: temperature=%.3g block_normalise=%s circular_weight=%.3g",
            model_cfg.cce.temperature, model_cfg.cce.block_normalise,
            model_cfg.cce.circular_weight,
        )

    # ── Weights from a previous run, and the optional encoder freeze ─────────
    # Order matters and is the reason these sit above the optimiser. The
    # weights have to arrive before the freeze, because freezing a random
    # encoder would hold noise still; and the freeze has to happen before the
    # optimiser is built, because the optimiser is given only the parameters
    # that still require gradients.
    if train_cfg.init_from_checkpoint:
        from endo_model.training.checkpointing import CheckpointManager

        init_path = Path(root) / train_cfg.init_from_checkpoint
        logger.info("Initialising weights from %s (fresh run: no optimiser "
                    "state, no epoch, no history)", init_path)
        CheckpointManager.load_weights(str(init_path), model,
                                       map_location=str(device))
    elif train_cfg.freeze_encoder:
        raise ValueError(
            "training.freeze_encoder needs training.init_from_checkpoint. "
            "Freezing a randomly initialised encoder holds noise still: its "
            "cell summary scores 0.512 on cell family against a trained "
            "encoder's 0.819, so the latent would be trained on nothing."
        )

    if train_cfg.freeze_encoder:
        from endo_model.training.freezing import EncoderFreezer

        freeze_report = EncoderFreezer().apply(model)
        logger.info(
            "Only the latent and decoder train. Gradient checkpointing turns "
            "itself off in the group encoder because its input no longer "
            "requires grad. Every result from this run is conditional on the "
            "encoder it was frozen from."
        )
        _ = freeze_report

    # ── Optimizer and scheduler, by their config names ───────────────────────
    from endo_model.training.factories import build_optimizer, build_scheduler
    from endo_model.training.freezing import trainable_parameters

    all_params = trainable_parameters(model, composite_loss)
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
            str(resume_path), model, optimizer, map_location=str(device),
            # Without this the adversary comes back at its random
            # initialisation while the optimiser restores Adam's moments
            # for it. See CheckpointManager.save.
            loss=composite_loss,
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
        label_probe=label_probe,
        gene_masker=gene_masker,
        drift_probe=drift_probe,
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
