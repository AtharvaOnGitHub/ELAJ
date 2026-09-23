"""How far the encoder's output moved since the previous epoch.

WHY. A frozen-encoder latent sweep is only worth running once the encoder is
worth freezing, and "has the encoder converged" has no answer as usually
asked. With cosine annealing the weights always settle by the final epoch
because the step size goes to zero by construction, so a settled weight vector
says something about the schedule rather than about the model.

What does have an answer is whether the FUNCTION has stopped changing. That is
what this measures: the same cells, every epoch, through the encoder, and how
far their cell summaries moved.

THREE NUMBERS, AND THE FIRST IS THE ONE THAT MATTERS.

  drift_neighbour_overlap  of each cell's k nearest neighbours in the cohort,
                           the fraction that were also its neighbours last
                           epoch. 1.0 means the neighbourhood structure is
                           unchanged. This is the one to read, because every
                           metric the project reports (cross-donor kNN, the
                           scIB batch metrics, anything a user does with UMAP)
                           depends on neighbourhood structure and on nothing
                           else about the representation.
  drift_cosine             mean cosine similarity between a cell's summary now
                           and last epoch. 1.0 means unchanged.
  drift_relative           mean ||x_now - x_before|| / ||x_before||. 0 means
                           unchanged.

The last two are diagnostics rather than the answer, because a representation
can rotate or rescale wholesale without any neighbour changing place. When
neighbour overlap is high and cosine is low, the encoder is moving the cells
together and the geometry is stable; when both are low it is genuinely
rearranging them.

THE COHORT IS CACHED, NOT RESAMPLED. The batches are pulled from the
validation loader once, at construction, and held. Two consequences, both
wanted: the same cells in the same order every epoch regardless of what the
sampler does, and no dependence on the sampler being seeded. The cost is the
counts for those batches sitting in host memory, about 16 MB per batch of 256
at this vocabulary.

WHY A SEPARATE FORWARD PASS. The validation loop calls `forward`, which
returns the latent and not `cell_cls`. Reading the encoder's own output needs
`encode_stages`, so this runs its own pass over a small cohort rather than
widening what validation collects. Eight batches is a few seconds against a
250 second validation epoch.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from endo_model.training.batch_device import to_device

logger = logging.getLogger(__name__)

DEFAULT_BATCHES = 8
DEFAULT_K = 30


@dataclass(frozen=True)
class DriftScores:
    """One epoch's movement against the previous one. See the module docstring."""

    neighbour_overlap: float
    cosine: float
    relative: float

    def as_metrics(self, prefix: str = "drift") -> dict[str, float]:
        return {f"{prefix}_{name}": float(value)
                for name, value in asdict(self).items()}

    def describe(self) -> str:
        return (f"neighbours kept {self.neighbour_overlap:.3f}, "
                f"cosine {self.cosine:.4f}, "
                f"relative change {self.relative:.4f}")


class RepresentationDrift:
    """Compares two matrices of the same cells in the same order.

    Args:
        k: Neighbours per cell for the overlap measure. 30 matches the
           validation probe, so the two are talking about the same
           neighbourhood size.
    """

    def __init__(self, k: int = DEFAULT_K) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)

    def compare(self, previous, current) -> DriftScores:
        """Args: two (n_cells, width) float tensors. Returns: DriftScores."""
        import torch

        if previous.shape != current.shape:
            raise ValueError(
                f"drift compares the same cells: got {tuple(previous.shape)} "
                f"and {tuple(current.shape)}"
            )
        previous = previous.float()
        current = current.float()

        cosine = torch.nn.functional.cosine_similarity(
            previous, current, dim=-1
        ).mean()
        moved = (current - previous).norm(dim=-1)
        relative = (moved / previous.norm(dim=-1).clamp_min(1e-8)).mean()

        return DriftScores(
            neighbour_overlap=float(self._overlap(previous, current)),
            cosine=float(cosine),
            relative=float(relative),
        )

    def _overlap(self, previous, current) -> float:
        """Mean fraction of each cell's k nearest neighbours that are shared.

        A cell is never its own neighbour, so the diagonal is pushed to
        infinity before the search rather than taking k+1 and dropping the
        first, which would be wrong for any cell that has an exact duplicate.
        """
        n = previous.shape[0]
        k = min(self.k, n - 1)
        if k < 1:
            return float("nan")

        before = self._neighbour_mask(previous, k)
        after = self._neighbour_mask(current, k)
        return float((before & after).sum(dim=1).float().mean() / k)

    @staticmethod
    def _neighbour_mask(matrix, k: int):
        """(n, n) bool, True where column j is among row i's k nearest."""
        import torch

        distance = torch.cdist(matrix, matrix)
        distance.fill_diagonal_(float("inf"))
        nearest = distance.topk(k, dim=1, largest=False).indices
        mask = torch.zeros_like(distance, dtype=torch.bool)
        return mask.scatter_(1, nearest, True)


class EncoderDriftProbe:
    """A fixed cohort of cells, re-encoded each epoch to see what moved.

    Args:
        batches: Cached batch dicts, held on the host and moved to the device
                 per use. The same cells in the same order every epoch.
        k:       Neighbours per cell for the overlap measure.
    """

    def __init__(self, batches: list[dict], k: int = DEFAULT_K) -> None:
        if not batches:
            raise ValueError("the drift probe needs at least one batch")
        self.batches = batches
        self.drift = RepresentationDrift(k=k)
        self._previous = None
        self.n_cells = sum(int(b["counts"].shape[0]) for b in batches)

    @classmethod
    def from_loader(cls, loader, n_batches: int = DEFAULT_BATCHES,
                    k: int = DEFAULT_K) -> "EncoderDriftProbe | None":
        """Cache the first `n_batches` batches. None if the loader is empty.

        Returns None rather than raising so that a run with no validation
        cells loses the diagnostic and not the epoch, which is the same
        contract the label probe has.
        """
        batches = []
        for batch in loader:
            batches.append(batch)
            if len(batches) >= n_batches:
                break
        if not batches:
            logger.warning("Drift probe: the validation loader yielded no "
                           "batches; drift will not be reported")
            return None
        probe = cls(batches, k=k)
        logger.info("Drift probe: %d cells over %d cached batches, %d "
                    "neighbours each", probe.n_cells, len(batches), k)
        return probe

    def measure(self, model, device) -> dict[str, float]:
        """Encode the cohort and compare with last epoch.

        Returns an empty dict on the first call, because there is nothing to
        compare against yet, and on any failure: this is a diagnostic, and it
        must never cost an epoch.
        """
        try:
            current = self._encode(model, device)
        except Exception as exc:                            # noqa: BLE001
            logger.warning("Drift probe failed (%s: %s)",
                           type(exc).__name__, exc)
            return {}

        previous, self._previous = self._previous, current
        if previous is None:
            return {}

        scores = self.drift.compare(previous, current)
        logger.info("Encoder drift since last epoch: %s", scores.describe())
        return scores.as_metrics()

    def _encode(self, model, device):
        """(n_cells, d_agg) — the cohort's cell summaries, on the CPU.

        `eval` mode on purpose. Dropout would put noise into the comparison
        that has nothing to do with whether the weights moved, and two epochs
        would differ even if the encoder were frozen solid.
        """
        import torch

        was_training = model.training
        model.eval()
        parts = []
        try:
            with torch.no_grad():
                for batch in self.batches:
                    moved = to_device(batch, device)
                    stages = model.encode_stages(moved)
                    parts.append(stages.cell_cls.detach().float().cpu())
        finally:
            model.train(was_training)
        return torch.cat(parts, dim=0)
