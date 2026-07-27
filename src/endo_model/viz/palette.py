"""Project colour palette for all ELAJ visualisations.

A single source of truth for colour so every figure in the project reads as
one visual system.  Import these names rather than hard-coding hex strings.

Design goals
------------
- Simple: a small, deliberate set of colours, not a rainbow.
- Functional: the categorical palette is the Okabe–Ito colour-blind-safe set,
  distinguishable under all common forms of colour-vision deficiency.
- Good-looking: a restrained ink-on-paper base with two brand accents.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Palette:
    """Immutable colour palette for the project.

    Attributes are grouped into: neutrals (structure/text), brand accents
    (primary emphasis), and scales (categorical + sequential) for data marks.
    """

    # ── Neutrals — page structure and text ──────────────────────────────────
    ink: str = "#22223b"          # primary text, strong axis labels
    ink_soft: str = "#4a4a63"     # secondary text, tick labels
    grid: str = "#e7e7ee"         # gridlines
    spine: str = "#c9c9d6"        # axis spines
    paper: str = "#ffffff"        # figure/axes background
    paper_alt: str = "#f7f7fb"    # subtle panel fill

    # ── Brand accents — primary emphasis ────────────────────────────────────
    primary: str = "#2a9d8f"      # teal — main series, chosen operating point
    secondary: str = "#e76f51"    # coral — contrast series, highlights
    muted: str = "#8d99ae"        # de-emphasised / reference marks

    # ── Categorical scale — Okabe–Ito, colour-blind safe ────────────────────
    categorical: tuple[str, ...] = (
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # green
        "#CC79A7",  # reddish purple
        "#56B4E9",  # sky blue
        "#D55E00",  # vermillion
        "#F0E442",  # yellow
        "#999999",  # grey
    )

    # ── Sequential scale — perceptually ordered teal ramp ───────────────────
    # Light → dark; use for ordered/continuous encodings (e.g. a swept dial).
    sequential: tuple[str, ...] = (
        "#e8f6f4",
        "#b9e3dd",
        "#8ad0c6",
        "#5bbdb0",
        "#2a9d8f",
        "#1f7a6f",
        "#155a52",
    )

    # ── Semantic — status colours ───────────────────────────────────────────
    good: str = "#009E73"         # positive / selected
    warn: str = "#E69F00"         # caution
    bad: str = "#D55E00"          # negative / excluded

    def categorical_cycle(self, n: int) -> list[str]:
        """Return n categorical colours, cycling if n exceeds the base set."""
        base = self.categorical
        return [base[i % len(base)] for i in range(n)]

    def sequential_sampled(self, n: int) -> list[str]:
        """Return n colours sampled evenly across the sequential ramp."""
        ramp = self.sequential
        if n <= 1:
            return [ramp[-1]]
        step = (len(ramp) - 1) / (n - 1)
        return [ramp[round(i * step)] for i in range(n)]


# Module-level singleton — import this.
PALETTE = Palette()
