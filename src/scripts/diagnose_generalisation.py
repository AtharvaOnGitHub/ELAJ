"""Why is val_nb worse than train_nb? Split the gap into its parts.

THE PROBLEM WITH THE NUMBER WE HAVE BEEN READING. `train_nb` is computed in
`model.train()` and `val_nb` in `model.eval()`, so the difference between them
is not the generalisation gap. It mixes three things:

    different cells            — generalisation, the thing we meant to measure
    dropout on vs off          — a different function, not a different fit
    DSBN batch vs running      — every batch holds one study, so in train mode
      statistics                 each study is normalised by its own statistics
                                 and its per-study scale and shift is learned
                                 against that; in eval mode every study is
                                 normalised by one corpus-wide running average
                                 and then corrected by a scale and shift
                                 calibrated for something else

No run in this project has ever scored training cells in eval mode, so nobody
can say which of the three the v-series gap is. That gap runs from 0.0264 to
0.0931 across five arms whose training fit spans 0.0057, so it is where every
architectural choice has actually been landing.

THE DESIGN. A 2x2 over {train cells, val cells} x {train mode, eval mode},
which decomposes exactly:

    val/eval - train/train  =  (val/eval - train/eval)   generalisation
                            +  (train/eval - train/train) the mode switch

and the other diagonal gives the same total by a different route, so a
disagreement between them is an interaction and is reported as one.

Then the mode axis is split again, because "eval mode" changes two things at
once. `eval_bn_train` is eval with only the BatchNorm layers put back into
training mode; `eval_dropout_on` is eval with only the dropout layers. Between
them they say which half of the mode switch costs what.

AND A THIRD ANSWER THIS CAN GIVE. Neither hypothesis need be right. The
per-study table is there for that: if the gap is concentrated in two or three
studies rather than spread across all eleven, the explanation is about the data
or the split and not about batch normalisation or the readout at all. Study
cell counts are printed beside it so a large gap on a thin study is visible as
such.

SAFETY. BatchNorm in training mode UPDATES its running statistics, which would
corrupt the checkpoint's buffers for every pass after the first. Every buffer
is saved before each pass and restored after. Dropout is stochastic, so the
seed is fixed before each pass and the same mask sequence is used throughout.

USAGE

    python -m scripts.diagnose_generalisation --run v1_bottleneck \\
        --out outputs/diagnostics/v1_bottleneck
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.diagnose_generalisation")

# The four ways the model can be run. "train" and "eval" are the two the
# metrics already use; the other two split the mode switch in half.
MODES = ("train", "eval", "eval_bn_train", "eval_dropout_on")
SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="Experiment name; its config gives corpus and model.")
    p.add_argument("--checkpoint", default="auto",
                   help="Checkpoint path, or 'auto' for the run's best.")
    p.add_argument("--runs-dir", default=None,
                   help="Parent of the run directory, if not the config's.")
    p.add_argument("--root", default=".")
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--batches-per-study", type=int, default=8,
                   help="Batches drawn from each study, per split. The whole "
                        "point is a like-for-like comparison, so the same "
                        "batches are used in every mode.")
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--out", required=True)
    return p.parse_args()


def set_mode(model, mode: str) -> None:
    """Put the model into one of the four regimes.

    Raises:
        ValueError: on an unknown mode, rather than silently leaving the model
            in whatever state the previous pass left it.
    """
    import torch.nn as nn

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "train":
        model.train()
        return
    model.eval()
    if mode == "eval_bn_train":
        wanted = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
    elif mode == "eval_dropout_on":
        wanted = (nn.Dropout, nn.Dropout1d, nn.Dropout2d)
    else:
        return
    for module in model.modules():
        if isinstance(module, wanted):
            module.train()


def batchnorm_buffers(model) -> dict:
    """A copy of every BatchNorm running statistic, to restore after a pass.

    BatchNorm in training mode updates these, so without this the first
    `eval_bn_train` pass would change what every later pass measures.
    """
    import torch.nn as nn

    saved = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.SyncBatchNorm)):
            saved[name] = (
                None if module.running_mean is None
                else module.running_mean.detach().clone(),
                None if module.running_var is None
                else module.running_var.detach().clone(),
                None if module.num_batches_tracked is None
                else module.num_batches_tracked.detach().clone(),
            )
    return saved


def restore_buffers(model, saved: dict) -> None:
    """Put the running statistics back exactly as they were."""
    for name, module in model.named_modules():
        if name not in saved:
            continue
        mean, var, count = saved[name]
        if mean is not None:
            module.running_mean.copy_(mean)
        if var is not None:
            module.running_var.copy_(var)
        if count is not None:
            module.num_batches_tracked.copy_(count)


def score_pass(model, batches, nb_loss, device, mode: str, seed: int):
    """Mean NB loss over `batches`, overall and per study.

    Returns `(overall, {study_id: (mean, n_cells)})`. Every batch holds one
    study, so the per-study means are exact rather than approximated.
    """
    import torch

    from endo_model.training.batch_device import to_device

    saved = batchnorm_buffers(model)
    set_mode(model, mode)
    torch.manual_seed(seed)          # dropout masks, identical across modes

    totals: dict[int, list[float]] = {}
    weighted, n_total = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            batch = to_device(batch, device)
            out = model(batch)
            x = batch["counts"][:, out["measured_global_idxs"]].float()
            value = float(nb_loss(x, out["mu_hat"], out["theta"]))
            n = int(x.shape[0])
            study = int(batch["study_id"][0].item())
            acc = totals.setdefault(study, [0.0, 0])
            acc[0] += value * n
            acc[1] += n
            weighted += value * n
            n_total += n

    restore_buffers(model, saved)
    model.eval()
    per_study = {s: (v / max(n, 1), n) for s, (v, n) in totals.items()}
    return (weighted / max(n_total, 1)), per_study


def collect_batches(loader, batches_per_study: int) -> list:
    """Up to `batches_per_study` batches from each study, held in memory.

    Held rather than re-iterated so that every mode sees exactly the same
    cells in the same order. That is the whole basis of the comparison.
    """
    seen: dict[int, int] = {}
    kept = []
    for batch in loader:
        study = int(batch["study_id"][0].item())
        if seen.get(study, 0) >= batches_per_study:
            continue
        seen[study] = seen.get(study, 0) + 1
        kept.append({k: (v.clone() if hasattr(v, "clone") else v)
                     for k, v in batch.items()})
    return kept


def decomposition(grid: dict[tuple[str, str], float]) -> list[str]:
    """The 2x2 read out as the two routes from train/train to val/eval."""
    a = grid[("train", "train")]
    b = grid[("val", "train")]
    c = grid[("train", "eval")]
    d = grid[("val", "eval")]
    lines = [
        f"  what metrics.csv reports as the gap   {d - a:+.4f}"
        f"   (val/eval {d:.4f} - train/train {a:.4f})",
        "",
        "  route 1: switch the cells first, then the mode",
        f"    generalisation, in train mode        {b - a:+.4f}",
        f"    the mode switch, on val cells        {d - b:+.4f}",
        "",
        "  route 2: switch the mode first, then the cells",
        f"    the mode switch, on train cells      {c - a:+.4f}",
        f"    generalisation, in eval mode         {d - c:+.4f}",
        "",
        f"  interaction (the two routes disagree) {(b - a) - (d - c):+.4f}",
    ]
    return lines


def main() -> None:
    import torch

    from endo_model.configs import load_experiment
    from endo_model.data.split_fingerprint import SplitGuard
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.purity_report import best_checkpoint
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.model.objectives.reconstruction import NegativeBinomialLoss
    from endo_model.training.checkpointing import CheckpointManager
    from scripts.report_purity import build_loader, resolve_run_dir

    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = Path(args.root) / args.config_dir / f"{args.run}.yaml"
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(str(config),
                                                              root=args.root)
    vocab = GeneVocabulary(str(Path(args.root) / data_cfg.vocabulary_path))
    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(args.root) / s.path)}
        for s in data_cfg.resolve_studies(args.root)
    ]
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)
    names = {s["id"]: s["name"] for s in study_configs}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = resolve_run_dir(args.root, exp_cfg.output_dir, args.runs_dir)
    ckpt = (best_checkpoint(run_dir) if args.checkpoint == "auto"
            else Path(args.checkpoint))
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    payload = CheckpointManager.load(str(ckpt), model, None, map_location=device)
    logger.info("Checkpoint %s", ckpt)

    nb_loss = NegativeBinomialLoss().to(device)

    # ── the same cells, every mode ──────────────────────────────────────────
    batches: dict[str, list] = {}
    for split in SPLITS:
        loader = build_loader(args.root, exp_cfg, train_cfg, data_cfg, vocab,
                              study_configs, split=split)
        if split == "val":
            SplitGuard().check(payload, loader.dataset)
        batches[split] = collect_batches(loader, args.batches_per_study)
        cells = sum(int(b["counts"].shape[0]) for b in batches[split])
        logger.info("%s split: %d batches, %d cells",
                    split, len(batches[split]), cells)

    grid: dict[tuple[str, str], float] = {}
    per_study: dict[tuple[str, str], dict] = {}
    for split in SPLITS:
        for mode in MODES:
            value, by_study = score_pass(model, batches[split], nb_loss,
                                         device, mode, args.seed)
            grid[(split, mode)] = value
            per_study[(split, mode)] = by_study
            logger.info("  %-5s cells, %-16s mode: nb %.4f", split, mode, value)

    # ── report ──────────────────────────────────────────────────────────────
    lines = [f"# Where the {args.run} gap comes from", "",
             f"Checkpoint `{ckpt.name}`. Same cells in every mode: "
             f"{args.batches_per_study} batches per study per split.", "",
             "## The four corners", "",
             "| cells | " + " | ".join(MODES) + " |",
             "|---|" + "---:|" * len(MODES)]
    for split in SPLITS:
        lines.append(f"| {split} | " + " | ".join(
            f"{grid[(split, m)]:.4f}" for m in MODES) + " |")
    lines += ["", "## Splitting the gap", "```"] + decomposition(grid) + ["```"]

    lines += ["", "## Which half of the mode switch", "```",
              f"  batch-norm alone, val cells   "
              f"{grid[('val','eval_bn_train')] - grid[('val','eval')]:+.4f}",
              f"  dropout alone, val cells      "
              f"{grid[('val','eval_dropout_on')] - grid[('val','eval')]:+.4f}",
              "```", "",
              "## Per study, val cells, eval mode against train mode", "",
              "| study | train cells | val cells | train/train | val/eval | gap |",
              "|---|---:|---:|---:|---:|---:|"]
    for study in sorted(per_study[("val", "eval")]):
        tr = per_study[("train", "train")].get(study, (float("nan"), 0))
        va = per_study[("val", "eval")].get(study, (float("nan"), 0))
        lines.append(f"| {names.get(study, study)} | {tr[1]} | {va[1]} | "
                     f"{tr[0]:.4f} | {va[0]:.4f} | {va[0] - tr[0]:+.4f} |")
    lines += ["", "A gap concentrated in a few studies is neither a "
              "batch-norm story nor a readout story, and is the reason this "
              "table is here.", ""]

    (out / "generalisation.md").write_text("\n".join(lines), encoding="utf-8")
    with open(out / "generalisation.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "mode", "study", "nb", "n_cells"])
        for (split, mode), by_study in per_study.items():
            w.writerow([split, mode, "ALL", round(grid[(split, mode)], 6), ""])
            for study, (value, n) in sorted(by_study.items()):
                w.writerow([split, mode, names.get(study, study),
                            round(value, 6), n])
    logger.info("\n%s", "\n".join(lines))
    logger.info("written to %s", out / "generalisation.md")


if __name__ == "__main__":
    main()
