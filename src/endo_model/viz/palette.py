"""Project colour palette for all ELAJ visualisations.

A single source of truth for colour so every figure in the project reads as
one visual system.  Import these names rather than hard-coding hex strings.

Berry and cream, matching the thesis deck: berry #6D2E46 for emphasis, dusty
rose #A26769 for the second series, cream #ECE2D0 for fills, on white.

Design goals
------------
- Simple: a small, deliberate set of colours, not a rainbow.
- Coherent with the written thesis: a figure dropped onto a slide should not
  announce that it came from somewhere else.
- Muted: nothing saturated, so no single series shouts over the others.

⚠ The categorical scale is no longer Okabe–Ito. That set is colour-blind safe
and this one is only partly so — the twelve study colours sweep hue from berry
through olive to indigo, which survives the common deficiencies better than a
red/green pair would, but it has not been verified. Where colour is the *only*
identifier — `study_scale` — keep a key beside the figure. `study_key` in
`corpus_panels` draws one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Palette:
    """Immutable colour palette for the project.

    Attributes are grouped into: neutrals (structure/text), brand accents
    (primary emphasis), and scales (categorical + sequential) for data marks.
    """

    # ── Neutrals — page structure and text, warmed to sit with cream ────────
    ink: str = "#4A2B35"          # primary text: berry, desaturated
    ink_soft: str = "#8A7078"     # secondary text, tick labels
    grid: str = "#EDE4DC"         # gridlines: cream, lightened
    spine: str = "#D9C8C0"        # axis spines
    paper: str = "#FFFFFF"        # figure/axes background — the slides are white
    paper_alt: str = "#FBF6EF"    # subtle panel fill

    # ── Brand accents — the deck's own three ────────────────────────────────
    primary: str = "#6D2E46"      # berry — main series, emphasis
    secondary: str = "#A26769"    # dusty rose — contrast series
    cream: str = "#ECE2D0"        # card fills, area shading, "empty" bar segments
    muted: str = "#B3A29C"        # de-emphasised / reference marks

    # ── Categorical scale ───────────────────────────────────────────────────
    # Order is load-bearing: corpus_panels maps disease status onto the first
    # four (eutopic, ectopic, control, adenomyosis) and training_panels uses
    # 0 and 1 for train and val.
    categorical: tuple[str, ...] = (
        "#A26769",  # dusty rose
        "#6D2E46",  # berry
        "#A98C63",  # khaki
        "#4E7C82",  # teal slate
        "#8A9463",  # olive
        "#B87D6A",  # clay
        "#5C6B99",  # dusty indigo
        "#7A5C8A",  # muted purple
    )

    # ── Study scale — colour as the identifier for a study ──────────────────
    # Twelve, because the corpus has twelve studies and their names
    # ("garcia_alonso_2021_a") eat half of any axis they are printed on. Warm
    # to cool so that a study's colour is memorable across figures; assign with
    # `study_colours`, which fixes the mapping in one place.
    study_scale: tuple[str, ...] = (
        "#6D2E46",  # berry
        "#A24E5E",  # deep rose
        "#A26769",  # dusty rose
        "#B87D6A",  # clay
        "#C89A72",  # warm sand
        "#A98C63",  # khaki
        "#8A9463",  # olive
        "#5F8068",  # moss
        "#4E7C82",  # teal slate
        "#5C6B99",  # dusty indigo
        "#7A5C8A",  # muted purple
        "#8C6A79",  # taupe rose
    )

    # ── Sequential scale — cream to berry ───────────────────────────────────
    # Light → dark; use for ordered/continuous encodings (e.g. a swept dial).
    sequential: tuple[str, ...] = (
        "#FCF8F3",
        "#F0E4D6",
        "#DFC3B9",
        "#CB9B9C",
        "#B27585",
        "#8E4E67",
        "#6D2E46",
    )

    # ── Semantic — status colours ───────────────────────────────────────────
    good: str = "#5F8068"         # positive / selected
    warn: str = "#C89A72"         # caution
    bad: str = "#A24E5E"          # negative / excluded

    def categorical_cycle(self, n: int) -> list[str]:
        """Return n categorical colours, cycling if n exceeds the base set.

        ⚠ Cycling repeats a colour. Use `distinct` for anything where two
        categories sharing a colour would be read as one.
        """
        base = self.categorical
        return [base[i % len(base)] for i in range(n)]

    def distinct(self, n: int) -> list[str]:
        """n colours that do not repeat, for as long as the palette can manage.

        The categorical scale holds eight and the corpus has nine anatomical
        sites, so `categorical_cycle(9)` gave endometrium and rectum the same
        colour and a stacked bar showed one band where there were two. This
        falls through to the study scale for the overflow, and only repeats
        once both are exhausted — at which point the figure has too many
        categories to read by colour anyway.
        """
        pool: list[str] = []
        for colour in (*self.categorical, *self.study_scale):
            if colour not in pool:
                pool.append(colour)
        return [pool[i % len(pool)] for i in range(n)]

    def study_colours(self, studies) -> dict[str, str]:
        """study name -> colour, in the order given.

        The order is the caller's choice and it matters: pass studies largest
        first and the biggest contributors wear the deepest berry tones, which
        is what the eye reaches for. Whatever order is chosen, use the same one
        everywhere in a document or the same study changes colour between
        figures. Cycles if there are more studies than colours.
        """
        scale = self.study_scale
        return {name: scale[i % len(scale)] for i, name in enumerate(studies)}

    def sequential_sampled(self, n: int) -> list[str]:
        """Return n colours sampled evenly across the sequential ramp."""
        ramp = self.sequential
        if n <= 1:
            return [ramp[-1]]
        step = (len(ramp) - 1) / (n - 1)
        return [ramp[round(i * step)] for i in range(n)]


# Module-level singleton — import this.
PALETTE = Palette()
