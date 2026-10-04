"""What is the decoder getting from the study embedding, and what from the latent?

    python -m scripts.report_reliance --run m3_studydec --root . \
        --runs-dir outputs/m_series

THE QUESTION. From the m3 wave the decoder is told which study a cell came
from. That is the scVI pattern and it was meant to remove the encoder's
reason to store study identity. Both m3 arms still fragmented, and the one
measurement that would say whether the conditioning is doing anything at all
was NaN in every epoch of both runs: `DecoderReliance` was handed a
DataLoader's CPU batches while the model sat on the GPU, so every batch
raised a device mismatch, every one was caught, and the metric was silently
absent.

This runs the same probe over a saved checkpoint, so the question is
answered from the runs we already paid for rather than from a relaunch.

WHAT IT REPORTS. Two counterfactuals on the same cells, same weights, no
training: tell the decoder a study the cell is not from, and decode each cell
from its neighbour's latent. Each costs some reconstruction, and the cost is
how much the decoder depends on that input.

    study high, latent high   both carry real information
    study high, latent LOW    the decoder is coasting on the study embedding
    study LOW,  latent high   conditioning bought nothing; the latent is
                              still doing the whole job, which is the
                              hypothesis this wave was built to test
    both low                  reconstruction is coming from library size and
                              gene identity alone

COST. Minutes on one card. No training, no backward, one encoder pass per
batch over a handful of batches.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.reliance")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Experiment name.")
    p.add_argument("--root", default=".")
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--runs-dir", default=None,
                   help="Where the run directory lives.")
    p.add_argument("--checkpoint", default="auto",
                   help="Checkpoint path, or 'auto' for the run's best.")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--batches", type=int, default=16,
                   help="Validation batches to average over.")
    p.add_argument("--out", default=None, help="Write reliance.json here too.")
    return p.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    root = args.root or "."

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.purity_report import best_checkpoint
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.model.objectives.reconstruction import NegativeBinomialLoss
    from endo_model.training.checkpointing import CheckpointManager
    from endo_model.training.decoder_reliance import DecoderReliance
    from scripts.report_purity import build_loader, resolve_run_dir

    config = Path(root) / args.config_dir / f"{args.run}.yaml"
    if not config.exists():
        logger.error("no config at %s", config)
        sys.exit(1)
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(str(config), root=root)

    vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))
    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
        for s in data_cfg.resolve_studies(root)
    ]
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(exp_cfg.seed)
    model = EndoFoundationModel(vocab, model_cfg).to(device)

    if model.decoder_study is None:
        logger.error("%s was trained with d_study 0, so there is no study "
                     "embedding to substitute and nothing to measure.",
                     args.run)
        sys.exit(2)

    run_dir = resolve_run_dir(root, exp_cfg.output_dir, args.runs_dir)
    ckpt = (best_checkpoint(run_dir) if args.checkpoint == "auto"
            else Path(args.checkpoint))
    logger.info("Checkpoint: %s", ckpt)
    CheckpointManager.load_weights(str(ckpt), model, map_location=str(device))
    model.eval()

    loader = build_loader(root, exp_cfg, train_cfg, data_cfg, vocab,
                          study_configs, split=args.split)

    scores = DecoderReliance(max_batches=args.batches).measure(
        model, loader, NegativeBinomialLoss().to(device), device=device)
    if not scores:
        logger.error("no batch could be measured; nothing to report")
        sys.exit(3)

    logger.info("\n%s", describe(args.run, scores))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        logger.info("Written to %s", out)


def describe(run: str, scores: dict) -> str:
    """The table, with the reading spelled out rather than left to the reader."""
    study = scores.get("reliance_study", float("nan"))
    latent = scores.get("reliance_latent", float("nan"))
    ratio = scores.get("reliance_ratio", float("nan"))
    if latent > study:
        verdict = ("The latent matters more than the study embedding. "
                   "Conditioning has not hollowed it out.")
    elif study > 0 and latent < 0.25 * study:
        verdict = ("THE DECODER IS COASTING ON THE STUDY EMBEDDING. The "
                   "latent contributes little to reconstruction, so val_nb "
                   "is measuring the decoder rather than the model.")
    elif study < 0.05:
        verdict = ("The study embedding is barely used, so conditioning the "
                   "decoder on study bought nothing and the reason the "
                   "latent stores study identity lies elsewhere.")
    else:
        verdict = "Both inputs carry information, with the study ahead."
    return (f"Decoder reliance: {run}\n"
            f"  wrong study  costs {study:+.4f} nats/gene\n"
            f"  wrong latent costs {latent:+.4f} nats/gene\n"
            f"  ratio (latent / study) {ratio:.2f}\n"
            f"  {verdict}")


if __name__ == "__main__":
    main()
