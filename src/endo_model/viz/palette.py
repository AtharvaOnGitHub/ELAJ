"""Project colour palette for all ELAJ visualisations.

A single source of truth for colour so every figure in the project reads as
one visual system.  Import these names rather than hard-coding hex strings.

Two palettes, because the figures have two audiences.

`PALETTE` is the default and is what every diagnostic figure uses — loss
curves, UMAPs, the adversarial panel. Its categorical scale is Okabe–Ito, which
is colour-blind safe and, more to the point here, maximally separable: a UMAP
identifies twelve studies by colour alone and muted neighbours are unreadable.

`BERRY` matches the thesis deck — berry #6D2E46, dusty rose #A26769, cream
#ECE2D0 on white — and is used by the corpus figures, which are made to sit on
slides. `describe_corpus` selects it; nothing else does.

Design goals
------------
- Simple: a small, deliberate set of colours, not a rainbow.
- Functional: the default categorical palette is colour-blind safe and
  separable at a glance.
- Consistent: one palette per audience, chosen once at the top of a script.
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
    cream: str = "#eef0f4"        # "empty" bar segments, area shading
    muted: str = "#8d99ae"        # de-emphasised / reference marks
    # Same hue as `primary`, desaturated and lightened. For marks that belong
    # to the primary family but are superseded: the frontier draws the final
    # p1 wave in `primary` and the earlier m-series in this, so hue carries
    # "ours" and saturation carries "current". Kept clear of `muted`, which is
    # a grey-blue and names a different family entirely.
    primary_faded: str = "#86c9bf"

    # Ablations: the same model with one component removed. A hue of its own
    # because an ablation is neither "ours, current" nor "ours, superseded" --
    # it is a counterfactual, and plotting it in either ELAJ colour invites a
    # reader to take it for a candidate model. Reddish purple from the
    # Okabe-Ito set, so it stays colour-blind safe beside primary and
    # secondary.
    ablation: str = "#CC79A7"

    # ── Categorical scale — Okabe–Ito, colour-blind safe ────────────────────
    # Order is load-bearing: corpus_panels maps disease status onto the first
    # four and training_panels uses 0 and 1 for train and val.
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

    # ── Study scale — colour as the identifier for a study ──────────────────
    # A UMAP names twelve studies by colour and nothing else, so these are
    # chosen for separability rather than harmony: the Okabe–Ito eight, then
    # four more picked to stay distinct from all of them.
    study_scale: tuple[str, ...] = (
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#7B3294",  # purple
        "#666666",  # dark grey
        "#8C510A",  # brown
        "#01665E",  # dark teal
        "#F0E442",  # yellow
        "#C51B7D",  # magenta
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


# Module-level singleton — import this for anything diagnostic.
PALETTE = Palette()

# The thesis deck's palette. Used by the corpus figures, which are built to sit
# on slides; `describe_corpus` selects it and nothing else does. Kept as a
# separate instance rather than a global switch so a diagnostic figure drawn in
# the same process cannot silently pick it up.
BERRY = Palette(
    ink="#4A2B35", ink_soft="#8A7078", grid="#EDE4DC", spine="#D9C8C0",
    paper="#FFFFFF", paper_alt="#FBF6EF",
    primary="#6D2E46", secondary="#A26769", cream="#ECE2D0", muted="#B3A29C",
    categorical=(
        "#A26769",  # dusty rose
        "#6D2E46",  # berry
        "#A98C63",  # khaki
        "#4E7C82",  # teal slate
        "#8A9463",  # olive
        "#B87D6A",  # clay
        "#5C6B99",  # dusty indigo
        "#7A5C8A",  # muted purple
    ),
    study_scale=(
        "#6D2E46", "#A24E5E", "#A26769", "#B87D6A",
        "#C89A72", "#A98C63", "#8A9463", "#5F8068",
        "#4E7C82", "#5C6B99", "#7A5C8A", "#8C6A79",
    ),
    sequential=(
        "#FCF8F3", "#F0E4D6", "#DFC3B9", "#CB9B9C",
        "#B27585", "#8E4E67", "#6D2E46",
    ),
    good="#5F8068", warn="#C89A72", bad="#A24E5E",
)
