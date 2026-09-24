#!/usr/bin/env python3
"""Locate the module that first produces a non-finite value in a forward pass.

A NaN reported by the latent sampler says nothing about where it came from:
DSBN normalises over the flattened (B x G) axis, so one bad value anywhere in a
batch makes every activation after it NaN.  By the time the error is raised the
evidence is gone.

This attaches a forward hook to every module, runs one batch, and reports the
first module whose output is non-finite *while all of its inputs were finite* —
that module is the origin, not a victim.

    python scripts/debug_nan.py --config configs/experiment/first_run.yaml \
        --root . --only-study shih_2022

Tests the first batch of every study it is given, so a fault confined to one
study is not missed.  Runs in train() mode by default — that is the
configuration that fails, and DSBN behaves differently under eval.

Reads only.  No optimiser is built and no weights are updated.  Note the
Dataset loads each h5ad in full during construction, so peak memory follows
--only-study rather than --max-cells-per-study.
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
logger = logging.getLogger("elaj.debug_nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Experiment YAML.")
    p.add_argument("--root", default=None, help="Project root.")
    p.add_argument(
        "--max-cells-per-study", type=int, default=2000,
        help="Subsample each study so setup is quick. 0 means no limit. "
             "Default: 2000.",
    )
    p.add_argument(
        "--match-training", action="store_true",
        help="Reproduce the batches training actually sees: shuffle=True with "
             "the experiment seed at epoch 0, scanning up to --batches of them. "
             "Without this the script tests the first 128 cells of each study in "
             "file order — a batch the training run never constructs — so a "
             "fault carried by particular cells is invisible.",
    )
    p.add_argument(
        "--batches", type=int, default=500, metavar="N",
        help="With --match-training, how many batches to scan. Default: 500.",
    )
    p.add_argument(
        "--only-study", nargs="+", metavar="NAME", default=None,
        help="Restrict to these studies. The Dataset loads every h5ad in full "
             "before subsampling, so this — not --max-cells-per-study — is what "
             "bounds peak memory. Use it to test one study at a time on a login "
             "node; omit it under sbatch.",
    )
    p.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Default: cuda when available. Use cpu to check whether the fault "
             "is device-specific — an out-of-range embedding index raises on "
             "CPU but reads unallocated memory on CUDA.",
    )
    p.add_argument(
        "--eval-mode", action="store_true",
        help="Run in eval() instead of train(). Train is the default because "
             "it is the configuration that fails: DSBN normalises with *batch* "
             "statistics in train mode and with running statistics (mean 0, "
             "var 1 at init) in eval, so a fault in the batch statistics does "
             "not reproduce under eval at all.",
    )
    return p.parse_args()


# The hook machinery and the finiteness helpers live in the package, not
# here, because the live training monitor uses the identical code.  Two
# copies would let this script and the training run disagree about what
# counts as a fault, which is the one thing a reproduction tool cannot do.
from endo_model.training.diagnostics import NaNGuard  # noqa: E402
from endo_model.utils.tensors import (  # noqa: E402
    describe,
    nonfinite_count,
    nonfinite_positions,
)


def report_batch(batch, label) -> bool:
    """Report non-finite values in each batch field. True if any are a fault.

    `age` is the one field where NaN is legal: it is the documented marker for
    "age unknown", and AgeEncoder substitutes a zero vector for it.  Counting
    that as a fault stops the run on healthy data — Inf in `age` is a genuine
    fault, because torch.isnan does not catch it and sin(inf) is NaN.
    """
    import torch

    faults = False
    for key, value in batch.items():
        n = nonfinite_count(value)
        if not n:
            continue

        if key == "age" and isinstance(value, torch.Tensor):
            n_inf = int(torch.isinf(value).sum().item())
            n_nan = n - n_inf
            note = f"{n_nan} NaN (expected: age unknown)"
            if n_inf:
                note += f", {n_inf} Inf  <-- FAULT, sin(inf) = NaN"
                faults = True
            print(f"    [{label}] age: {note}")
            continue

        faults = True
        print(f"    [{label}] '{key}': {n} non-finite — {describe(value)}  <-- FAULT")
        if isinstance(value, torch.Tensor):
            print(f"        first offending positions: {nonfinite_positions(value)}")

    if not faults:
        print(f"    [{label}] inputs OK")
    return faults


def main() -> None:
    args = parse_args()

    import torch

    root = args.root or str(Path.cwd())

    from endo_model.configs import load_experiment
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(args.config, root=root)

    from endo_model.data.vocabulary import GeneVocabulary
    vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))

    studies = data_cfg.resolve_studies(root)
    if args.only_study:
        wanted = set(args.only_study)
        unknown = wanted - {s.name for s in studies}
        if unknown:
            logger.error("--only-study names not in the corpus: %s", sorted(unknown))
            sys.exit(1)
        studies = [s for s in studies if s.name in wanted]

    # Renumber after filtering.  study.id indexes the DSBN and study-embedding
    # tables, whose width is n_studies below — keeping the original id of a
    # single filtered study would put it out of range.
    study_configs = [
        {"id": new_id, "name": s.name, "path": str(Path(root) / s.path)}
        for new_id, s in enumerate(studies)
    ]
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)
    logger.info(
        "Testing %d study/studies: %s",
        len(study_configs), ", ".join(s["name"] for s in study_configs),
    )

    from torch.utils.data import DataLoader
    from endo_model.data.collate import make_collate_fn
    from endo_model.data.dataset import EndometriosisDataset
    from endo_model.data.samplers import PerStudyBatchSampler

    if args.match_training and args.max_cells_per_study:
        logger.warning(
            "--match-training with --max-cells-per-study %d: subsampling changes "
            "which cells exist, so the batches will not be the ones training "
            "saw. Pass --max-cells-per-study 0 for an exact reproduction.",
            args.max_cells_per_study,
        )

    dataset = EndometriosisDataset(
        study_configs=study_configs, vocab=vocab, split="train",
        val_fraction=data_cfg.splits.val_fraction,
        test_fraction=data_cfg.splits.test_fraction,
        seed=data_cfg.splits.seed,
        max_cells_per_study=args.max_cells_per_study or None,
    )

    # Training shuffles with the experiment seed; survey mode does not shuffle
    # at all, which is why it only ever sees the head of each study.
    sampler = PerStudyBatchSampler(
        dataset.study_ids, train_cfg.batch_size,
        shuffle=args.match_training,
        seed=exp_cfg.seed if args.match_training else 0,
    )
    if args.match_training:
        sampler.set_epoch(0)  # Trainer calls this before epoch 0
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=make_collate_fn(vocab)
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    from endo_model.model.endo_model import EndoFoundationModel
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    model.eval() if args.eval_mode else model.train()

    print(f"\ndevice={device}  mode={'eval' if args.eval_mode else 'train'}  "
          f"batch_size={train_cfg.batch_size}")

    from endo_model.training.batch_device import to_device

    id_to_name = {s["id"]: s["name"] for s in study_configs}

    def test(batch, label: str, verbose: bool) -> bool:
        """Run one batch through the hooked model. True if a fault appeared."""
        batch_bad = report_batch(batch, label) if verbose else False
        moved = to_device(batch, device)

        with NaNGuard(model) as guard, torch.no_grad():
            try:
                model(moved)
            except Exception as exc:  # a raise is also a useful result
                print(f"\n── {label} " + "─" * 40)
                print(f"    forward raised {type(exc).__name__}: {str(exc)[:300]}")
                print("\n".join(guard.lines()))
                return True
            if guard.events:
                if not verbose:
                    print(f"\n── {label} " + "─" * 40)
                    report_batch(batch, label)
                print("\n".join(guard.lines()))
                return True
            if verbose:
                print("\n".join(guard.lines()))
        return batch_bad

    if args.match_training:
        print(f"\nScanning up to {args.batches} shuffled batches "
              f"(seed={exp_cfg.seed}, epoch 0) — the order training uses.\n")
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            study = int(batch["study_id"][0].item())
            label = f"batch {i} — study {study} ({id_to_name.get(study, '?')})"
            if test(batch, label, verbose=False):
                print(f"\nFault reproduced at {label} — stopping.")
                return
            if i % 25 == 0:
                print(f"    ... {i} batches clean")
        print(f"\nNo fault in {args.batches} shuffled batches. This model is "
              "freshly initialised and takes no optimiser step, so a fault that "
              "only appears once the weights have moved cannot show up here at "
              "all — run training itself, where TrainingMonitor reports the same "
              "trace at whichever step it happens.")
        return

    # Survey mode: one batch per study.  With shuffle=False the sampler is
    # study-major, so scanning batches in order would only exercise one study.
    seen: set[int] = set()
    for batch in loader:
        study = int(batch["study_id"][0].item())
        if study in seen:
            continue
        seen.add(study)

        print(f"\n── study {study} ({id_to_name.get(study, '?')}) " + "─" * 34)
        if test(batch, id_to_name.get(study, str(study)), verbose=True):
            print(f"\nFault found in study {study} "
                  f"({id_to_name.get(study, '?')}) — stopping.")
            return
        if len(seen) == len(study_configs):
            break

    print(f"\nNo fault in the first batch of any of {len(seen)} studies. "
          "Those are the first 128 cells of each study in file order, which "
          "training never sees — rerun with --match-training "
          "--max-cells-per-study 0.")


if __name__ == "__main__":
    main()
