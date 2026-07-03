"""infer.py — extract latent embeddings from a trained ELAJ checkpoint.

Runs inference on any dataset split and saves the resulting latent
representations for downstream use (UMAP visualisation, Leiden clustering,
transfer to another tool, or scib benchmarking).

Usage:
    python scripts/infer.py --checkpoint outputs/full_model/checkpoints/best.pt \\
                            --config configs/experiment/full_model.yaml \\
                            --split test --out outputs/embeddings/

Outputs:
    embeddings.npz    mu_gauss (N, d_gauss), z (N, d_z), mu_angle (N, n_vMF),
                      kappa (N, n_vMF), study_ids (N,), disease_status (N,)
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract latent embeddings from a trained ELAJ checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML used to train the checkpoint.",
    )
    parser.add_argument("--root", default=None, help="Project root (default: cwd).")
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Dataset split to run inference on (default: test).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for embeddings.npz (default: alongside checkpoint).",
    )
    parser.add_argument(
        "--keys", nargs="+",
        default=["mu_gauss", "z", "mu_angle", "kappa"],
        help="Which model output keys to save (default: mu_gauss z mu_angle kappa).",
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
        dataset.study_ids, train_cfg.batch_size, shuffle=False, seed=0, drop_last=False,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=make_collate_fn(vocab))
    logger.info("Running inference on split='%s': %d cells", args.split, len(dataset))

    # ── Load model ────────────────────────────────────────────────────────────
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    CheckpointManager.load(str(ckpt_path), model, map_location=str(device))
    model.eval()
    logger.info("Loaded checkpoint: %s", ckpt_path)

    # ── Collect all requested embedding keys ─────────────────────────────────
    collected: dict[str, list[np.ndarray]] = {k: [] for k in args.keys}
    study_ids_all: list[np.ndarray] = []
    disease_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = _to_device(batch, device)
            out = model(batch_dev)
            for key in args.keys:
                if key in out:
                    collected[key].append(out[key].cpu().numpy())
                else:
                    logger.warning("Key '%s' not in model output — skipping.", key)
            study_ids_all.append(batch["study_id"].numpy())
            disease_labels.append(batch["disease_status"].numpy())

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dict: dict[str, np.ndarray] = {}
    for key in args.keys:
        if collected[key]:
            save_dict[key] = np.concatenate(collected[key], axis=0)
    save_dict["study_ids"] = np.concatenate(study_ids_all, axis=0)
    save_dict["disease_status"] = np.concatenate(disease_labels, axis=0)

    out_path = out_dir / "embeddings.npz"
    np.savez_compressed(out_path, **save_dict)

    logger.info("Saved embeddings to %s", out_path)
    for key, arr in save_dict.items():
        logger.info("  %-20s shape=%s  dtype=%s", key, arr.shape, arr.dtype)


def _to_device(batch: dict, device: "torch.device") -> dict:
    import torch
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {
                kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
