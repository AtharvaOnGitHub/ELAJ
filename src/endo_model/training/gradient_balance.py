"""How hard each loss term actually pulls on the latent.

WHY THIS EXISTS. Every decision this project has made about `lambda_dab` was
made by reasoning about coefficients, and a coefficient is not a force. Two
terms with the same weight can exert wildly different pulls depending on their
gradients, and nothing here has ever measured one. Five waves walked
`lambda_dab` from 0.2 down to 0.0 on that basis.

It is worth measuring for a second reason. The adversarial gradient reaching
the latent is EXACTLY linear in `lambda_dab`, because the gradient reversal
multiplies the backward pass by it. So one reading of

    ||d(dab)/d(mu)||  against  ||d(nb)/d(mu)||

lets the next lambda be solved for rather than guessed: to move the ratio by a
factor, move lambda by the same factor. That is not true of the equilibrium the
two settle into, since the classifier trains on `w_dab` and not on lambda, so
this predicts the instantaneous pull and not the endpoint. It is still the
difference between choosing a number from a measurement and choosing it from a
hunch.

WHAT IT COSTS. One extra backward pass per term, on one batch per epoch. The
graph is already built, so this is a few seconds against a 4,900-second epoch.
It never runs on more than one batch and never touches the optimiser.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The order the report prints in: reconstruction first, because everything else
# is read as a fraction of it.
TERMS = ("nb", "kl_gauss", "kl_circ", "cce", "dab", "cls")


class GradientBalance:
    """Per-term gradient norms at `mu_gauss`, once per epoch.

    Args:
        composite: The CompositeLoss whose weights scale each term.
    """

    def __init__(self, composite) -> None:
        self.composite = composite
        self._epoch_done: int | None = None

    def due(self, epoch: int) -> bool:
        """One batch per epoch, and only the first."""
        return self._epoch_done != epoch

    def weights(self, beta_kl: float, n_measured: int) -> dict[str, float]:
        """What each term's raw value is multiplied by inside `total`.

        `kl_gauss`, `kl_circ` and `nb` arrive already per gene. `cce` and `dab`
        arrive per cell and are divided here, matching composite.py. `dab`'s
        gradient already carries `lambda_dab` through the reversal layer, so
        lambda does not appear again.
        """
        c = self.composite
        return {
            "nb": c.w_nb,
            "kl_gauss": beta_kl * c.w_kl_gauss,
            "kl_circ": beta_kl * c.w_kl_circ,
            "cce": c.w_cce / max(n_measured, 1),
            "dab": c.w_dab / max(n_measured, 1),
            # Per cell like cce and dab, so divided the same way. Its
            # absence from this report is why m1 ran two days with DAB
            # at 0.0000x and nobody could check the one number they
            # were told to check.
            "cls": c.w_cls / max(n_measured, 1),
        }

    def measure(self, losses: dict, mu_gauss, n_measured: int) -> dict[str, float]:
        """Weighted gradient norm of each term with respect to `mu_gauss`.

        Returns `{}` on any failure. This is a diagnostic and must never be
        able to stop a run.
        """
        import torch

        if not mu_gauss.requires_grad:
            return {}
        scale = self.weights(float(losses.get("beta_kl", 0.0)), n_measured)
        out: dict[str, float] = {}
        for name in TERMS:
            value = losses.get(name)
            if value is None or not getattr(value, "requires_grad", False):
                continue
            try:
                grad, = torch.autograd.grad(
                    value, mu_gauss, retain_graph=True, allow_unused=True,
                )
            except Exception as exc:                            # noqa: BLE001
                logger.warning("gradient balance: %s failed (%s: %s)",
                               name, type(exc).__name__, exc)
                continue
            if grad is None:
                continue
            out[name] = float(grad.norm()) * scale[name]
        return out

    def report(self, epoch: int, losses: dict, mu_gauss, n_measured: int) -> None:
        """Measure and log, once per epoch. Silent on failure."""
        try:
            norms = self.measure(losses, mu_gauss, n_measured)
        except Exception as exc:                                # noqa: BLE001
            logger.warning("gradient balance failed (%s: %s)",
                           type(exc).__name__, exc)
            return
        self._epoch_done = epoch
        if not norms:
            return
        base = norms.get("nb", 0.0)
        parts = []
        for name in TERMS:
            if name not in norms:
                continue
            share = norms[name] / base if base > 0 else float("nan")
            parts.append(f"{name} {norms[name]:.3e} ({share:.4f}x nb)")
        logger.info("Gradient pull on the latent, weighted: %s",
                    "; ".join(parts))
