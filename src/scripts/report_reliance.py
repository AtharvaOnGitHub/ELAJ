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

WHAT IT REPORTS. Three counterfactuals on the same cells, same weights, no
training: tell the decoder a study the cell is not from; swap every
provenance channel at once (study, platform, tissue, disease); and decode
each cell from its neighbour's latent. Each costs some reconstruction, and
the cost is how much the decoder depends on that input.

    technical high, latent high   both carry real information. Wanted.
    technical high, latent LOW    the decoder is coasting on the covariates
                                  and the latent is being starved
    technical LOW,  latent high   conditioning bought nothing; the latent is
                                  doing the whole job
    study low, technical high     the study embedding is redundant with
                                  platform and tissue, not unused

`reliance_technical` is the one to read. Substituting the study id alone
leaves platform and tissue still agreeing with the data, and every study in
this corpus has one or two platform combinations, so `d_platform` is a study
code in all but name. That is why the study-only readings of ~0.004 are an
artefact rather than a finding.

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
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds the latent derangement, so the reported "
                        "number is reproducible from the checkpoint.")
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

    scores = DecoderReliance(max_batches=args.batches,
                             seed=args.seed).measure(
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


#: Channel name -> whether it is nuisance or biology. The distinction is the
#: whole point of splitting the technical total: a decoder leaning on
#: `platform` is a batch-information problem, and a decoder leaning on
#: `tissue` is a biological variable handed over for free.
KIND = {"tissue": "BIOLOGY", "platform": "nuisance",
        "disease": "BIOLOGY", "study": "nuisance"}


def channels(scores: dict) -> str:
    """One line per channel, largest first, each as a share of the total."""
    total = scores.get("reliance_technical", float("nan"))
    rows = sorted(((name, scores.get(f"reliance_{name}", float("nan")))
                   for name in KIND),
                  key=lambda row: (row[1] != row[1], -row[1]))
    return "\n".join(
        f"    {name:<9} {cost:+.4f}  {100.0 * cost / total:5.1f}% of "
        f"technical   {KIND[name]}"
        for name, cost in rows)


def verdict(scores: dict) -> str:
    """The reading, chosen by the ratio rather than by absolute thresholds.

    What matters is not whether each penalty clears some bar, but which of
    the two the decoder leans on: the cell's own latent, or being told where
    the cell came from.
    """
    tech = scores.get("reliance_technical", float("nan"))
    latent = scores.get("reliance_latent", float("nan"))
    study = scores.get("reliance_study", float("nan"))
    tissue = scores.get("reliance_tissue", float("nan"))
    if tech != tech or latent != latent:
        return ("Not measured. No batch had a substitute from another study, "
                "so the technical counterfactual could not be formed. Raise "
                "--batches, or the loader is single-study.")
    ratio = latent / tech if tech else float("inf")
    if ratio >= 1.5:
        lines = ["The LATENT is what the decoder reconstructs from. "
                 "Conditioning has not hollowed it out."]
    elif ratio >= 0.9:
        lines = ["The latent and the provenance channels matter about "
                 "equally, which is more reliance on provenance than a "
                 "16-dimension latent against 48 covariate dimensions "
                 "should need."]
    else:
        lines = ["THE DECODER LEANS ON PROVENANCE MORE THAN ON THE CELL. "
                 "The latent is being out-competed as an information "
                 "source, not over-priced: anything predictable from "
                 "provenance is learned on channels that cost no KL and "
                 "draw no adversarial gradient."]
    if tech and study < 0.2 * tech:
        lines.append("The study embedding is REDUNDANT with the other "
                     "channels, not unused. Changing d_study will not move "
                     "this.")
    if tech and tissue > 0.4 * tech:
        lines.append("TISSUE is the channel the decoder reads, and tissue is "
                     "biology. This is not batch information leaking into "
                     "the decoder; it is a biological variable handed over "
                     "for free, so the latent has no reason to encode it.")
    return "\n  ".join(lines)


def describe(run: str, scores: dict) -> str:
    """The table, with the reading spelled out rather than left to the reader."""
    nan = float("nan")
    tech = scores.get("reliance_technical", nan)
    latent = scores.get("reliance_latent", nan)
    share = scores.get("reliance_study_share", nan)
    ratio = scores.get("reliance_ratio", nan)
    return (f"Decoder reliance: {run}\n"
            f"  wrong technical costs {tech:+.4f} nats/gene"
            f"   (study is {100.0 * share:.0f}% of it)\n"
            f"{channels(scores)}\n"
            f"  wrong latent    costs {latent:+.4f} nats/gene\n"
            f"  ratio (latent / technical) {ratio:.2f}\n"
            f"  {verdict(scores)}")


if __name__ == "__main__":
    main()
