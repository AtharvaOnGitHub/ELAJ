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
import sys
from pathlib import Path

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
    return parser.parse_args()


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
    )
    train_dataset = EndometriosisDataset(split="train", **split_kwargs)
    val_dataset = EndometriosisDataset(split="val", **split_kwargs)
    logger.info(
        "Dataset: %d train cells, %d val cells",
        len(train_dataset), len(val_dataset),
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    from endo_model.data.collate import make_collate_fn
    from endo_model.data.samplers import PerStudyBatchSampler

    collate = make_collate_fn(vocab)
    train_sampler = PerStudyBatchSampler(
        train_dataset.study_ids, train_cfg.batch_size,
        shuffle=True, seed=exp_cfg.seed,
    )
    val_sampler = PerStudyBatchSampler(
        val_dataset.study_ids, train_cfg.batch_size,
        shuffle=False, seed=exp_cfg.seed,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate)

    # ── Model ─────────────────────────────────────────────────────────────────
    from endo_model.model.endo_model import EndoFoundationModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = EndoFoundationModel(vocab, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ── Composite loss ────────────────────────────────────────────────────────
    from endo_model.model.objectives.composite import CompositeLoss
    composite_loss = CompositeLoss(
        n_studies=model_cfg.n_studies,
        d_gauss=model_cfg.d_gauss,
        n_vMF=model_cfg.n_vMF,
        w_cce=model_cfg.cce.lambda_cce if model_cfg.cce.enabled else 0.0,
        lambda_dab_max=model_cfg.dab.lambda_dab,
        cce_temperature=model_cfg.cce.temperature,
        dab_hidden_dim=model_cfg.dab.hidden_dim,
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    all_params = list(model.parameters()) + list(composite_loss.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay,
        betas=train_cfg.optimizer.betas,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    total_steps = len(train_loader) * train_cfg.max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    if train_cfg.resume_from_checkpoint:
        from endo_model.training.checkpointing import CheckpointManager
        resume_path = Path(root) / train_cfg.resume_from_checkpoint
        logger.info("Resuming from %s", resume_path)
        payload = CheckpointManager.load(str(resume_path), model, optimizer, map_location=str(device))
        start_epoch = payload.get("epoch", 0) + 1

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
    )
    trainer.current_epoch = start_epoch

    # ── Save run config snapshot ──────────────────────────────────────────────
    _save_run_config(out_dir, exp_cfg, model_cfg, train_cfg, data_cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training — max_epochs=%d", train_cfg.max_epochs)
    history = trainer.fit()

    # ── Write metrics CSV ─────────────────────────────────────────────────────
    _write_metrics_csv(out_dir / "metrics.csv", history)
    logger.info("Training complete. Outputs written to %s", out_dir)


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
