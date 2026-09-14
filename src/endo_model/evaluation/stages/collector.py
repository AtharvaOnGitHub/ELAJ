"""StageCollector: run the encoder over a split and keep every stage.

The benchmark's `collect` keeps the latent; this keeps whatever the readers
ask for, in the loader's own order, so row i of every matrix is the i-th
cell the sampler yielded and the corpus built over `sampler_order` lines up
with all of them at once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from endo_model.evaluation.stages.readers import StageReader

logger = logging.getLogger(__name__)


@dataclass
class StageEmbeddings:
    """One (n_cells, width) float32 matrix per stage, all over the same rows."""

    matrices: dict[str, np.ndarray] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.matrices)

    @property
    def n_cells(self) -> int:
        if not self.matrices:
            return 0
        return int(next(iter(self.matrices.values())).shape[0])

    def width(self, name: str) -> int:
        return int(self.matrices[name].shape[1])

    def of_kind(self, kind: str) -> list[str]:
        return [n for n, k in self.kinds.items() if k == kind]

    def nbytes(self) -> int:
        return int(sum(m.nbytes for m in self.matrices.values()))

    @staticmethod
    def concatenate(parts: list["StageEmbeddings"]) -> "StageEmbeddings":
        """Rows of several splits, stacked in the order given. Every part
        must hold the same stages; a stage missing from one part would leave
        that stage's rows misaligned with the corpus."""
        parts = [p for p in parts if p.matrices]
        if not parts:
            return StageEmbeddings()
        names = parts[0].names
        for p in parts[1:]:
            if p.names != names:
                raise ValueError(f"stage sets differ across splits: {names} "
                                 f"vs {p.names}")
        out = StageEmbeddings(kinds=dict(parts[0].kinds))
        for name in names:
            out.matrices[name] = np.concatenate([p.matrices[name] for p in parts])
        return out


class StageCollector:
    """Applies a list of readers to every batch a loader yields."""

    def __init__(self, readers: list[StageReader]) -> None:
        names = [r.name for r in readers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate stage names: {duplicates}")
        self.readers = list(readers)

    def collect(self, model, loader, device, order_len: int) -> StageEmbeddings:
        """Embed the split at every stage, in the loader's order.

        `order_len` is the number of indices the sampler will yield; a
        mismatch means the metadata join downstream would pair each cell
        with another cell's labels, so it is an error, not a warning.
        """
        import torch

        from endo_model.training.trainer import _to_device

        parts: dict[str, list[np.ndarray]] = {r.name: [] for r in self.readers}
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, device)
                stages = model.encode_stages(batch)
                for reader in self.readers:
                    got = reader.read(stages, batch)
                    parts[reader.name].append(got.float().cpu().numpy())

        out = StageEmbeddings()
        for reader in self.readers:
            if not parts[reader.name]:
                continue
            matrix = np.concatenate(parts[reader.name])
            if len(matrix) != order_len:
                raise RuntimeError(
                    f"{reader.name}: collected {len(matrix)} rows but the "
                    f"sampler yielded {order_len} indices; the metadata join "
                    f"would be misaligned"
                )
            out.matrices[reader.name] = np.ascontiguousarray(matrix, dtype=np.float32)
            out.kinds[reader.name] = reader.kind
        logger.info("Collected %d stages over %d cells (%.2f GB)",
                    len(out.matrices), out.n_cells, out.nbytes() / 1e9)
        return out
