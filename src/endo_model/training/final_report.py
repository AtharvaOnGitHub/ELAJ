"""Redraw the latent figures from the best checkpoint once training ends.

The per-epoch figures are drawn on an interval — `latent_every`, usually 4 — so
the last one written is whatever epoch happened to land on that interval. It is
almost never the best epoch. In r3_nodab the best epoch was 16 and the figures
run to 16 by luck; in r3_capacity the best was 27 and the last figure is 28; in
r3_loose the best was 21 and the last is 24.

So the pictures a run is judged on have been of a model nobody chose. This
draws them once more from the checkpoint that early stopping and the
checkpoint manager actually selected, into `figures/best/`.

It also reports kNN study purity, which the per-epoch figures do not: the DAB
loss says whether a classifier can name the study, and purity says whether a
cell's neighbours come from its own study. Those are different questions and
this project has been reading the first as an answer to the second.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FinalReporter:
    """Draws the end-of-run latent figures from a chosen checkpoint.

    Args:
        reporter:   The EpochReporter whose collector and figure settings are
                    reused, so the final figures are directly comparable with
                    the per-epoch ones.
        loader:     Validation DataLoader.
        device:     Where the model lives.
        out_dir:    Run output directory; figures land in `out_dir/figures/best`.
    """

    def __init__(self, reporter, loader, device, out_dir: Path | str) -> None:
        self.reporter = reporter
        self.loader = loader
        self.device = device
        self.fig_dir = Path(out_dir) / "figures" / "best"

    def render(self, model, checkpoint_path: Path | str | None) -> None:
        """Load the checkpoint, collect val latents, write the figures.

        Never raises. This runs after `trainer.fit()` has returned and the
        metrics have been written, so a failure here must not turn a completed
        run into a failed one.
        """
        try:
            self._render(model, checkpoint_path)
        except Exception as exc:  # noqa: BLE001 — diagnostics are not load-bearing
            logger.warning(
                "Final figures skipped (%s: %s). Training itself is unaffected "
                "and the per-epoch figures are already written.",
                type(exc).__name__, exc,
            )

    def _render(self, model, checkpoint_path: Path | str | None) -> None:
        import torch

        from endo_model.evaluation.latent_collection import collect_latents
        from endo_model.evaluation.study_mixing import describe
        from endo_model.training.checkpointing import CheckpointManager

        if checkpoint_path is None:
            logger.info("No checkpoint retained; final figures skipped.")
            return

        payload = CheckpointManager.load(
            str(checkpoint_path), model, None, map_location=self.device
        )
        epoch = payload.get("epoch", "?")
        logger.info("Final figures: reloaded best checkpoint (epoch %s)", epoch)

        model.eval()
        with torch.no_grad():
            latents = collect_latents(
                model, self.loader, self.device, self.reporter.new_collector()
            )
        if latents is None:
            logger.info("Validation loader was empty; final figures skipped.")
            return

        logger.info("Best epoch, %s", describe(latents["mu_gauss"], latents["study_id"]))

        self.fig_dir.mkdir(parents=True, exist_ok=True)
        original, self.reporter.fig_dir = self.reporter.fig_dir, self.fig_dir
        try:
            # draw_latents, not on_epoch_end: the latter also redraws the loss
            # panels, and there is no history here to draw them from.
            self.reporter.draw_latents(
                epoch if isinstance(epoch, int) else 0, latents
            )
        finally:
            self.reporter.fig_dir = original
        logger.info("Final figures written to %s", self.fig_dir)
