"""Does the model beat the study-mean baseline? Asked once per epoch, out loud.

Runs kl_check and kl_check_no_dab both finished three epochs, wrote every
figure, and saved every checkpoint before anyone noticed that neither had ever
scored better than predicting each study's mean profile. Nothing in the loop
was wrong: the number was in metrics.csv the whole time, in a different column
from the loss everyone was watching, and the comparison had to be done by hand
afterwards.

So it is done here instead, at the end of every validation pass, as one line
that says better or worse in words.

The comparison is only meaningful because MeanProfileBaseline is *fitted* on
the training split and *scored* on validation — the same held-out cells the
model is scored on. A baseline scored on its own training data would be a
harder target than it deserves to be, and beating it would mean less.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineVerdict:
    """One epoch's comparison against the baseline.

    Args:
        epoch:    Epoch index.
        metric:   Which loss was compared, e.g. "nb" or "nb_hvg".
        model:    The model's validation value.
        baseline: The baseline's validation value on the same cells.
    """

    epoch: int
    metric: str
    model: float
    baseline: float

    @property
    def beats_baseline(self) -> bool:
        """Lower is better — these are negative log-likelihoods."""
        return self.model < self.baseline

    @property
    def relative_gain(self) -> float:
        """Fraction better than the baseline; negative means worse."""
        if self.baseline == 0:
            return 0.0
        return (self.baseline - self.model) / abs(self.baseline)

    def describe(self) -> str:
        word = "BETTER than" if self.beats_baseline else "WORSE than"
        return (
            f"epoch {self.epoch:3d}  val_{self.metric}={self.model:.4f}  "
            f"baseline={self.baseline:.4f}  "
            f"{abs(self.relative_gain):.1%} {word} baseline"
        )


class BaselineGate:
    """Compares each epoch's validation loss against the baseline and logs it.

    Args:
        baseline_all: Baseline NB loss over all measured genes, scored on the
                      validation split.
        baseline_hvg: The same over highly variable genes, or None when no HVG
                      set could be built.

    Not a callback that can stop the run. A model below the baseline early is
    normal and stopping there would be wrong; the value is in seeing it, every
    epoch, without having to open a CSV.
    """

    def __init__(
        self, baseline_all: float, baseline_hvg: float | None = None
    ) -> None:
        self.baseline_all = baseline_all
        self.baseline_hvg = baseline_hvg
        self.history: list[BaselineVerdict] = []

    def evaluate(self, epoch: int, val_metrics: dict) -> list[BaselineVerdict]:
        """Compare one epoch's validation metrics. Returns what it compared."""
        verdicts = []
        if "nb" in val_metrics:
            verdicts.append(
                BaselineVerdict(epoch, "nb", float(val_metrics["nb"]),
                                self.baseline_all)
            )
        # Only when an HVG set exists. A zero baseline means none was built —
        # comparing against it would report an impressive-looking loss every
        # epoch and mean nothing.
        if self.baseline_hvg and "nb_hvg" in val_metrics:
            verdicts.append(
                BaselineVerdict(epoch, "nb_hvg", float(val_metrics["nb_hvg"]),
                                self.baseline_hvg)
            )
        self.history.extend(verdicts)
        for verdict in verdicts:
            logger.info("BASELINE  %s", verdict.describe())
        if not verdicts:
            logger.warning(
                "BASELINE  epoch %d: nothing to compare — val metrics carried "
                "no 'nb'. The model is running unscored against the baseline.",
                epoch,
            )
        return verdicts

    @property
    def best_gain(self) -> float:
        """Best relative gain on the all-genes metric seen so far."""
        gains = [v.relative_gain for v in self.history if v.metric == "nb"]
        return max(gains) if gains else float("-inf")

    def summarise(self) -> str:
        """One line for the end of the run."""
        if not self.history:
            return "Baseline comparison: never ran."
        best = max(
            (v for v in self.history if v.metric == "nb"),
            key=lambda v: v.relative_gain,
        )
        word = "beat" if best.beats_baseline else "never beat"
        return (
            f"Baseline comparison: the model {word} the study-mean baseline. "
            f"Best epoch {best.epoch}, val_nb={best.model:.4f} vs "
            f"{best.baseline:.4f} ({best.relative_gain:+.1%})."
        )
