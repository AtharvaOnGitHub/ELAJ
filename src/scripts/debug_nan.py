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
        --root . --max-cells-per-study 2000

Reads only.  No optimiser is built and no weights are updated.
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
        help="Subsample each study so setup takes seconds, not minutes. "
             "Default: 2000.",
    )
    p.add_argument(
        "--batches", type=int, default=5,
        help="How many batches to test before giving up. Default: 5.",
    )
    p.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Default: cuda when available. Use cpu to check whether the fault "
             "is device-specific — an out-of-range embedding index raises on "
             "CPU but reads unallocated memory on CUDA.",
    )
    p.add_argument(
        "--train-mode", action="store_true",
        help="Run the model in train() mode. Off by default so the stochastic "
             "latent samplers do not confound the trace; turn it on to test "
             "dropout- and sampling-dependent faults.",
    )
    return p.parse_args()


def nonfinite_count(obj) -> int:
    """Total non-finite elements in a tensor, or nested in a dict/tuple/list."""
    import torch

    if isinstance(obj, torch.Tensor):
        if not obj.is_floating_point():
            return 0
        return int((~torch.isfinite(obj)).sum().item())
    if isinstance(obj, dict):
        return sum(nonfinite_count(v) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(nonfinite_count(v) for v in obj)
    return 0


def describe(obj) -> str:
    """Shape and finite range summary, for the report."""
    import torch

    if isinstance(obj, torch.Tensor) and obj.is_floating_point():
        finite = obj[torch.isfinite(obj)]
        if finite.numel() == 0:
            return f"shape={tuple(obj.shape)} all non-finite"
        return (
            f"shape={tuple(obj.shape)} "
            f"range=[{finite.min().item():.4g}, {finite.max().item():.4g}]"
        )
    if isinstance(obj, torch.Tensor):
        return f"shape={tuple(obj.shape)} dtype={obj.dtype}"
    if isinstance(obj, dict):
        return f"dict[{len(obj)}]"
    return type(obj).__name__


class NaNLocator:
    """Records, in execution order, which modules emit non-finite output.

    A module whose inputs were all finite but whose output is not is where the
    fault begins.  Everything downstream of it is a consequence.
    """

    def __init__(self, model) -> None:
        self._model = model
        self._handles: list = []
        self.events: list[dict] = []

    def __enter__(self) -> "NaNLocator":
        for name, module in self._model.named_modules():
            if name:  # skip the root, whose output is the whole result dict
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            n_out = nonfinite_count(output)
            if n_out == 0:
                return
            self.events.append({
                "name": name,
                "type": type(module).__name__,
                "n_in": nonfinite_count(inputs),
                "n_out": n_out,
                "in_desc": ", ".join(describe(i) for i in inputs),
                "out_desc": describe(output),
            })
        return hook

    def report(self) -> bool:
        """Print the trace. Returns True if a non-finite value was seen."""
        if not self.events:
            print("    no non-finite values in any module output  OK")
            return False

        origins = [e for e in self.events if e["n_in"] == 0]
        print(f"    {len(self.events)} module(s) emitted non-finite output\n")

        if origins:
            first = origins[0]
            print(f"    ORIGIN: {first['name']}  ({first['type']})")
            print(f"      inputs  : all finite — {first['in_desc']}")
            print(f"      output  : {first['n_out']} non-finite — {first['out_desc']}")
            print("\n    Everything below is downstream of it:")
        else:
            print("    No module had finite inputs and non-finite output, so the "
                  "fault entered with the batch itself — see the batch report above.")

        for e in self.events[:15]:
            marker = "  <-- ORIGIN" if origins and e is origins[0] else ""
            print(f"      {e['name']:<52} {e['type']:<22} "
                  f"in={e['n_in']:>8} out={e['n_out']:>10}{marker}")
        if len(self.events) > 15:
            print(f"      ... and {len(self.events) - 15} more")
        return True


def report_batch(batch, index: int) -> bool:
    """Report non-finite values in each batch field. True if any were found."""
    import torch

    bad = False
    for key, value in batch.items():
        n = nonfinite_count(value)
        if n:
            bad = True
            print(f"    batch[{index}]['{key}']: {n} non-finite — {describe(value)}")
            if isinstance(value, torch.Tensor) and value.dim() == 1:
                rows = (~torch.isfinite(value)).nonzero().flatten()[:8].tolist()
                print(f"        first offending positions: {rows}")
    if not bad:
        print(f"    batch[{index}]: all fields finite")
    return bad


def main() -> None:
    import torch

    args = parse_args()
    root = args.root or str(Path.cwd())

    from endo_model.configs import load_experiment
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(args.config, root=root)

    from endo_model.data.vocabulary import GeneVocabulary
    vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))

    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
        for s in data_cfg.resolve_studies(root)
    ]
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)

    from torch.utils.data import DataLoader
    from endo_model.data.collate import make_collate_fn
    from endo_model.data.dataset import EndometriosisDataset
    from endo_model.data.samplers import PerStudyBatchSampler

    dataset = EndometriosisDataset(
        study_configs=study_configs, vocab=vocab, split="train",
        val_fraction=data_cfg.splits.val_fraction,
        test_fraction=data_cfg.splits.test_fraction,
        seed=data_cfg.splits.seed,
        max_cells_per_study=args.max_cells_per_study,
    )
    sampler = PerStudyBatchSampler(
        dataset.study_ids, train_cfg.batch_size, shuffle=False, seed=0
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=make_collate_fn(vocab)
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    from endo_model.model.endo_model import EndoFoundationModel
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    model.train() if args.train_mode else model.eval()

    print(f"\ndevice={device}  mode={'train' if args.train_mode else 'eval'}  "
          f"batch_size={train_cfg.batch_size}")

    from endo_model.training.trainer import _to_device

    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        study = int(batch["study_id"][0].item())
        print(f"\n── batch {i} (study_id={study}) "
              + "─" * 40)

        batch_bad = report_batch(batch, i)
        batch = _to_device(batch, device)

        with NaNLocator(model) as locator, torch.no_grad():
            try:
                model(batch)
            except Exception as exc:  # a raise is also a useful result
                print(f"    forward raised {type(exc).__name__}: "
                      f"{str(exc)[:200]}")
            found = locator.report()

        if found or batch_bad:
            print("\nStopping at the first batch that shows a fault.")
            return

    print("\nNo fault reproduced in the batches tested. Try --train-mode, "
          "more --batches, or a larger --max-cells-per-study.")


if __name__ == "__main__":
    main()
