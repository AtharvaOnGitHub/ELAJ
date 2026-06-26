"""evaluate.py — evaluate a trained ELAJ checkpoint on the test split.

Usage:
    python scripts/evaluate.py --checkpoint outputs/full_model/checkpoints/epoch_0099_step_0099999.pt \\
                               --config configs/experiment/full_model.yaml

Outputs (written alongside the checkpoint or to --out):
    eval_results.json    NB loss, Pearson r, median kappa, optional scib metrics
    embeddings.npz       mu_gauss embeddings + metadata for UMAP / scib (optional)
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ELAJ checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML used to train the checkpoint.",
    )
    parser.add_argument(
        "--root", default=None,
        help="Project root (default: current working directory).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: same directory as checkpoint).",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--save-embeddings", action="store_true",
        help="Also save embeddings.npz for downstream UMAP / scib analysis.",
    )
    parser.add_argument(
        "--scib", action="store_true",
        help="Compute scib batch-correction and bio-conservation metrics (requires scib-metrics).",
    )
    return parser.parse_args()


def main() -> None:
    import numpy as np
    import torch

    args = parse_args()
    root = args.root or str(Path.cwd())
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out) if args.out else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load configs ──────────────────────────────────────────────────────────
    from endo_model.configs import load_experiment
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(args.config, root=root)

    # ── Vocabulary ────────────────────────────────────────────────────────────
    from endo_model.data.vocabulary import GeneVocabulary
    vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))

    # ── Study configs ─────────────────────────────────────────────────────────
    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
        for s in data_cfg.resolve_studies(root)
    ]
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    from endo_model.data.collate import make_collate_fn
    from endo_model.data.dataset import EndometriosisDataset
    from endo_model.data.samplers import PerStudyBatchSampler

    dataset = EndometriosisDataset(
        study_configs=study_configs,
        vocab=vocab,
        split=args.split,
        val_fraction=data_cfg.splits.val_fraction,
        test_fraction=data_cfg.splits.test_fraction,
        seed=data_cfg.splits.seed,
    )
    sampler = PerStudyBatchSampler(
        dataset.study_ids, train_cfg.batch_size, shuffle=False, seed=0,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=make_collate_fn(vocab))
    logger.info("Evaluating on split='%s': %d cells", args.split, len(dataset))

    # ── Load model ────────────────────────────────────────────────────────────
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    CheckpointManager.load(str(ckpt_path), model, map_location=str(device))
    model.eval()
    logger.info("Loaded checkpoint: %s", ckpt_path)

    # ── Core evaluation metrics ───────────────────────────────────────────────
    from endo_model.evaluation.evaluator import Evaluator
    evaluator = Evaluator(model, loader, device=device)
    metrics = evaluator.evaluate()
    logger.info("Reconstruction metrics: %s", metrics)

    # ── Collect embeddings for scib / UMAP ───────────────────────────────────
    embeddings = study_ids = None
    if args.save_embeddings or args.scib:
        from endo_model.evaluation.latent_analysis import collect_embeddings
        embeddings, study_ids = collect_embeddings(model, loader, device=device, key="mu_gauss")
        logger.info("Collected embeddings: shape=%s", embeddings.shape)

    if args.save_embeddings and embeddings is not None:
        emb_path = out_dir / "embeddings.npz"
        np.savez_compressed(emb_path, embeddings=embeddings, study_ids=study_ids)
        logger.info("Saved embeddings to %s", emb_path)

    # ── Optional scib metrics ─────────────────────────────────────────────────
    if args.scib and embeddings is not None:
        from endo_model.evaluation.scib_metrics import compute_scib_metrics
        bio_labels = _collect_disease_labels(dataset)
        scib_results = compute_scib_metrics(
            embeddings=embeddings,
            batch_labels=study_ids,
            bio_labels=bio_labels,
        )
        metrics.update(scib_results)
        logger.info("scib metrics: %s", scib_results)

    # ── Write results ─────────────────────────────────────────────────────────
    result_path = out_dir / "eval_results.json"
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Results written to %s", result_path)

    # Pretty-print summary
    print("\n── Evaluation Results ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}" if isinstance(v, float) else f"  {k:<25} {v}")
    print("────────────────────────────────────────────────")


def _collect_disease_labels(dataset) -> "np.ndarray":
    """Extract integer disease_status labels from dataset records."""
    import numpy as np
    return np.array([r.disease_status for r in dataset._records], dtype=np.int64)


if __name__ == "__main__":
    main()
