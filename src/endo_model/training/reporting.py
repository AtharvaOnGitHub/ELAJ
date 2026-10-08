"""EpochReporter: writes the figures that make a run readable while it runs.

Called at the end of every epoch. Loss panels are cheap and redrawn each time;
the latent figures need a forward pass worth of embeddings and a UMAP fit, so
they run on an interval and on a fixed subsample.

The subsample is fixed deliberately. Choosing fresh cells each time would make
consecutive epochs incomparable — a change in the picture could be the model
moving or just different cells — and comparing epochs is the entire purpose.

Nothing here is allowed to end a run. A figure is a diagnostic; losing one to a
missing dependency or a bad axis limit must not cost the GPU hours already
spent, so every draw is guarded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from endo_model.data.constants import DISEASE_STATUS_TO_INT
from endo_model.viz import apply_theme, save_figure
from endo_model.viz.latent_panels import circular_latent, latent_umap, safe_umap
from endo_model.viz.training_panels import (
    adversarial_panel,
    baseline_comparison,
    loss_curves,
)

logger = logging.getLogger(__name__)

_DISEASE_NAMES = {v: k for k, v in DISEASE_STATUS_TO_INT.items()}


@dataclass
class LatentCollector:
    """Accumulates latent codes and their labels over a validation pass.

    Args:
        max_cells: Stop after this many. A UMAP with `random_state` set runs
                   single-threaded, so the full 244k-cell split would take
                   longer than the epoch that produced it.
    """

    max_cells: int = 20000
    n_studies: int = 1
    seed: int = 24
    mu_gauss: list = field(default_factory=list)
    mu_angle: list = field(default_factory=list)
    kappa: list = field(default_factory=list)
    study_id: list = field(default_factory=list)
    disease: list = field(default_factory=list)
    tissue: list = field(default_factory=list)
    # Only present when the model has a drain block. Collected so the
    # attribution figure can say what those dimensions hold, which no metric
    # reads: see evaluation/latent_attribution.py.
    mu_drain: list = field(default_factory=list)
    platform: list = field(default_factory=list)
    phase: list = field(default_factory=list)
    log_depth: list = field(default_factory=list)
    log_genes: list = field(default_factory=list)
    _n: int = 0
    _per_study: dict = field(default_factory=dict)

    def add(self, output: dict, batch: dict) -> None:
        """Record every cell. Subsampling happens in `stacked`.

        Deliberately unbounded. The previous version kept a per-study quota and
        filled it from whichever batches arrived first, and validation is not
        shuffled — so a study's sample was whatever sat at the top of its file.
        fonseca_2023 carries eight anatomical sites and contributed cells from
        one or two of them; bladder and rectum never appeared in any UMAP.

        Holding everything costs about 31 MB for a 163k-cell split (48 floats
        per cell), which is nothing next to a subsample that misrepresents the
        data it is drawn from.
        """
        def grab(t: Tensor) -> np.ndarray:
            return t.detach().float().cpu().numpy()

        self.mu_gauss.append(grab(output["mu_gauss"]))
        self.mu_angle.append(grab(output["mu_angle"]))
        self.kappa.append(grab(output["kappa"]))
        self.study_id.append(batch["study_id"].cpu().numpy())
        self.disease.append(batch["disease_status"].cpu().numpy())
        self.tissue.append(batch["tissue_levels"].cpu().numpy())
        # The drain's own targets and the biology labels it must NOT hold, so
        # the attribution figure can tell those two cases apart.
        self.platform.append(batch["platform_codes"].cpu().numpy())
        self.phase.append(batch["phase_label"].cpu().numpy())
        library = batch["library_size"].detach().float().cpu().numpy()
        self.log_depth.append(np.log1p(library))
        counts = batch["counts"]
        detected = ((counts > 0) & ~batch["is_dns"]).sum(dim=1)
        self.log_genes.append(np.log1p(grab(detected)))
        if "mu_drain" in output:
            self.mu_drain.append(grab(output["mu_drain"]))
        self._n += int(output["mu_gauss"].shape[0])

    def _subsample(self, study_id: np.ndarray) -> np.ndarray:
        """Row indices for an even, random sample across studies.

        Even because otherwise duempelmann_2026 and fonseca_2023 are 70% of the
        validation split between them and a UMAP would mostly describe those
        two. Random within each study because taking the first N of anything
        sorted by sample takes one sample.
        """
        from endo_model.evaluation.study_mixing import balanced_rows

        # One definition, shared with the benchmark's study-mixing battery,
        # so the purity reported here and the purity in the comparison table
        # are the same quantity against the same floor.
        return balanced_rows(study_id, self.max_cells, self.seed)

    def stacked(self) -> dict[str, np.ndarray] | None:
        """Everything collected, evenly subsampled across studies."""
        if not self.mu_gauss:
            return None
        full = {
            "mu_gauss": np.concatenate(self.mu_gauss),
            "mu_angle": np.concatenate(self.mu_angle),
            "kappa": np.concatenate(self.kappa),
            "study_id": np.concatenate(self.study_id),
            "disease": np.concatenate(self.disease),
            "tissue": np.concatenate(self.tissue),
            "platform": np.concatenate(self.platform),
            "phase": np.concatenate(self.phase),
            "log_depth": np.concatenate(self.log_depth),
            "log_genes": np.concatenate(self.log_genes),
        }
        # Absent on a model with no drain, and the key must then stay absent
        # rather than arrive empty: `draw_latents` keys the whole attribution
        # figure off its presence.
        if self.mu_drain:
            full["mu_drain"] = np.concatenate(self.mu_drain)
        rows = self._subsample(full["study_id"])
        return {name: values[rows] for name, values in full.items()}


class EpochReporter:
    """Draws every per-epoch figure into `out_dir/figures/`.

    Args:
        out_dir:        Run output directory.
        n_studies:      For the adversary's chance line, ln(n_studies).
        id_to_name:     study id -> study name, for legends.
        baseline_all:   Mean-profile baseline NB loss over all measured genes.
        baseline_hvg:   The same over highly variable genes.
        latent_every:   Epochs between latent figures. 0 disables them.
        max_latent_cells: Size of the fixed val subsample used for UMAP.
        seed:           UMAP seed, so consecutive epochs are comparable.
        tissue_names:   Level column index -> {code -> label}, for the tissue
                        legends. Absent, those panels legend integers.
    """

    def __init__(
        self,
        out_dir: Path | str,
        n_studies: int,
        id_to_name: dict[int, str] | None = None,
        baseline_all: float | None = None,
        baseline_hvg: float | None = None,
        latent_every: int = 10,
        max_latent_cells: int = 20000,
        seed: int = 24,
        tissue_names: dict[int, dict[int, str]] | None = None,
    ) -> None:
        self.fig_dir = Path(out_dir) / "figures"
        self.n_studies = n_studies
        self.id_to_name = id_to_name or {}
        self.tissue_names = tissue_names or {}
        self.baseline_all = baseline_all
        self.baseline_hvg = baseline_hvg
        self.latent_every = latent_every
        self.max_latent_cells = max_latent_cells
        self.seed = seed
        # Filled by draw_latents; FinalReporter writes the last set to CSV.
        self.last_mixing_rows: list[dict] = []
        # Compute nodes have no display. matplotlib usually detects that,
        # but a figure backend guessing wrong on a 24-hour job is not worth
        # the risk when one line removes the question.
        import matplotlib
        matplotlib.use("Agg", force=True)
        apply_theme()

    def wants_latents(self, epoch: int) -> bool:
        """Whether this epoch's validation pass should collect latent codes.

        Epoch 0 is always included: the first picture is the reference every
        later one is read against, and a model that is already collapsed at
        epoch 0 should not take ten epochs to reveal it.
        """
        if self.latent_every <= 0:
            return False
        return epoch == 0 or epoch % self.latent_every == 0

    def new_collector(self) -> LatentCollector:
        return LatentCollector(
            max_cells=self.max_latent_cells, n_studies=self.n_studies,
            seed=self.seed,
        )

    def on_epoch_end(
        self, epoch: int, history: dict, latents: dict[str, np.ndarray] | None
    ) -> None:
        """Redraw everything available for this epoch."""
        self._draw("losses", lambda: loss_curves(history))
        if self.baseline_all is not None:
            self._draw(
                "baseline",
                lambda: baseline_comparison(
                    history, self.baseline_all, self.baseline_hvg,
                    model_hvg=history.get("val_nb_hvg"),
                ),
            )
        self._draw("adversarial", lambda: adversarial_panel(history, self.n_studies))

        if latents is not None:
            self.draw_latents(epoch, latents)

    def draw_latents(self, epoch: int, latents: dict) -> None:
        """Every figure derived from latent codes, for one set of them.

        Public and separate from `on_epoch_end` because `FinalReporter` redraws
        exactly these from the best checkpoint once training ends, and must not
        also redraw the loss panels — it has no history to draw them from.
        """
        self.last_mixing_rows = self.report_mixing(latents)
        if latents["mu_angle"].shape[1] > 0:      # Gaussian-only models have none
            self._draw(
                f"latent_circular_epoch{epoch:03d}",
                lambda: circular_latent(latents["mu_angle"], latents["kappa"], epoch),
            )
        self._draw_umap(epoch, latents)
        self._draw_drain(epoch, latents)

    def _draw_drain(self, epoch: int, latents: dict) -> None:
        """What the drain block holds, against what the biology block holds.

        WHY IT IS WORTH A FIGURE. m6_drain's drain held 26.5 nats over eight
        dimensions. Its three targets cannot account for that: naming one of
        eleven studies is at most ln(11) = 2.40 nats, and depth and detected-
        gene count are one continuous quantity each. So it holds something
        else, and nothing in the project could say what.

        It is also the one block no metric reads. `DABClassifier`,
        `LabelClassifier`, `validate_latent` and the benchmark all take
        `mu_gauss` and `mu_angle`, which is the structural guarantee that the
        drain cannot inflate a reported number -- and equally the reason
        biology parked there would be invisible. This is where it shows.

        Drawn beside the biology block over the same variables, because the
        drain's numbers mean nothing on their own: a column high in both is
        shared structure, and high in the drain alone is the failure.
        """
        drain = latents.get("mu_drain")
        if drain is None or drain.shape[1] == 0:
            return
        from endo_model.evaluation.latent_attribution import attribute, describe
        from endo_model.viz.drain_panels import drain_attribution

        categorical = {
            "study": latents["study_id"],
            "platform": latents["platform"][:, 0],
            "tissue type": latents["tissue"][:, 2],
            "disease": latents["disease"],
            "phase": latents["phase"],
        }
        continuous = {
            "log depth": latents["log_depth"],
            "log genes": latents["log_genes"],
        }
        try:
            drain_table = attribute(drain, categorical, continuous)
            biology = np.concatenate(
                [latents["mu_gauss"],
                 np.cos(latents["mu_angle"]), np.sin(latents["mu_angle"])],
                axis=1) if latents["mu_angle"].shape[1] else latents["mu_gauss"]
            biology_table = attribute(biology, categorical, continuous)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Latent attribution skipped (%s: %s).",
                           type(exc).__name__, exc)
            return
        logger.info("%s", describe(drain_table, "DRAIN"))
        logger.info("%s", describe(biology_table, "BIOLOGY"))
        self._draw(
            f"latent_attribution_epoch{epoch:03d}",
            lambda: drain_attribution(
                drain_table, biology_table, drain.shape[1],
                biology.shape[1], epoch,
                note="variance explained is not information: a low value is "
                     "weak evidence of absence, a high one is strong evidence "
                     "of presence",
            ),
        )
        self._draw_drain_umap(epoch, latents)

    def _draw_drain_umap(self, epoch: int, latents: dict) -> None:
        """The drain block's own UMAP, coloured the way the Gaussian one is."""
        coords = safe_umap(latents["mu_drain"], seed=self.seed)
        if coords is None:
            return
        labels = {
            "by study": (latents["study_id"], self.id_to_name,
                         "separation here is the drain WORKING"),
            "by tissue type": (latents["tissue"][:, 2],
                               self.tissue_names.get(2), None),
            "by disease status": (latents["disease"], _DISEASE_NAMES,
                                  "separation here is biology in the wrong "
                                  "block"),
        }
        self._draw(
            f"latent_umap_drain_epoch{epoch:03d}",
            lambda: latent_umap(coords, labels, epoch, branch="Drain"),
        )

    def report_mixing(self, latents: dict) -> list[dict]:
        """Log study mixing for this set of latents, corpus and per study.

        Called from `draw_latents`, so it fires on exactly the epochs that get
        latent figures and once more at the best checkpoint. That pairing is
        deliberate: the UMAP is the picture and this is the number behind it,
        and reading the picture without the number is how three waves of this
        project mistook a rearranged blob for a mixed one.

        One neighbour search serves both readouts -- see study_mixing.summarise.
        Wrapped, because a diagnostic must not end a training run.
        """
        from endo_model.evaluation.study_mixing import format_per_study, summarise

        try:
            line, rows = summarise(
                latents["mu_gauss"], latents["study_id"], names=self.id_to_name
            )
            logger.info("%s", line)
            if rows:
                logger.info(
                    "kNN purity per study:\n%s",
                    format_per_study(rows, self.id_to_name),
                )
            return rows
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Study mixing skipped (%s: %s).", type(exc).__name__, exc
            )
            return []

    def _draw_umap(self, epoch: int, latents: dict[str, np.ndarray]) -> None:
        coords = safe_umap(latents["mu_gauss"], seed=self.seed)
        if coords is None:
            return
        labels = {
            "by study": (
                latents["study_id"], self.id_to_name,
                "separation here = batch effects survive",
            ),
            # No caveat any more: disease_status was removed from the model, so
            # separation here is a finding rather than a restatement of an
            # input. Leaving the old "CIRCULAR" warning up would have argued
            # against our own result.
            "by disease status": (latents["disease"], _DISEASE_NAMES, None),
            # Named, not numbered. These two panels used to legend themselves
            # "1 (17,303)" and "2 (2,009)", which says how many cells carry a
            # code and nothing about what the code is.
            "by tissue compartment": (
                latents["tissue"][:, 0], self.tissue_names.get(0), None,
            ),
            "by organ": (latents["tissue"][:, 1], self.tissue_names.get(1), None),
            # tissue_type is where endometrium and myometrium separate, and it
            # is the level with the most distinct values actually present in
            # this corpus. Omitting it left the two tissue panels dominated by
            # a single category — organ was 16,074 of 19,312 cells in code 1.
            "by tissue type": (
                latents["tissue"][:, 2], self.tissue_names.get(2), None,
            ),
        }
        self._draw(
            f"latent_umap_epoch{epoch:03d}",
            lambda: latent_umap(coords, labels, epoch),
        )
        self._draw_circular_umap(epoch, latents, labels)

    def _draw_circular_umap(self, epoch: int, latents, labels) -> None:
        """The same UMAP, over the circular branch instead of the Gaussian one.

        Run s1_circ_low's Gaussian UMAP showed studies thoroughly mixed, which
        reads as batch effects being gone — while val_dab sat at 0.405 against a
        chance of 2.079, meaning the study classifier could still name the study
        almost perfectly. Both cannot be true of the same latent.

        The resolution is that the existing panel plots `mu_gauss` only, and
        that run was the first with six live circular dimensions. The study
        information had somewhere else to go. This panel is where to look for it.

        Plotted from (cos θ, sin θ) pairs and not from the angles. The earlier
        version passed the angles straight in, on the reasoning that the pair
        doubles the width for no added information. That reasoning was wrong:
        UMAP builds its graph from Euclidean distances, and on a raw angle two
        cells at θ = −π and θ = +π sit 2π apart while occupying the same point
        on the circle. Every circular UMAP before this change carried a seam
        that belonged to the encoding rather than to the cells.
        """
        angles = latents.get("mu_angle")
        if angles is None or angles.shape[1] < 2:
            return
        circular = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
        coords = safe_umap(circular, seed=self.seed)
        if coords is None:
            return
        self._draw(
            f"latent_umap_circular_epoch{epoch:03d}",
            lambda: latent_umap(coords, labels, epoch, branch="Power Spherical"),
        )

    def _draw(self, name: str, build) -> None:
        """Build and save one figure, swallowing anything it raises.

        A broken figure must never end a run that has already spent hours on
        the GPU, so the exception is logged with its type and the run continues.
        """
        import matplotlib.pyplot as plt

        try:
            fig = build()
        except Exception as exc:  # noqa: BLE001 — diagnostics are not load-bearing
            logger.warning("figure %r failed (%s: %s)", name, type(exc).__name__, exc)
            return
        try:
            save_figure(fig, self.fig_dir / name)
        finally:
            plt.close(fig)
