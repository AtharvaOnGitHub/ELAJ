# endo_model/viz — Project Visualisation Theme

One visual identity for every figure in the project.  Import the theme rather
than styling plots ad hoc, so figures from different scripts stay consistent.

## Files

| File | Responsibility |
|---|---|
| `palette.py` | `Palette` dataclass + the `PALETTE` singleton — the single source of colour |
| `theme.py` | `apply_theme()` (matplotlib rcParams), `save_figure()`, `sequential_line_colors()` |

## Usage

```python
from endo_model.viz import PALETTE, apply_theme, save_figure

apply_theme()                       # call once, before creating figures

fig, ax = plt.subplots()
ax.plot(x, y, color=PALETTE.primary)

save_figure(fig, out_dir / "my_plot", formats=("png",))
```

## Palette structure

| Group | Members | Use for |
|---|---|---|
| Neutrals | `ink`, `ink_soft`, `grid`, `spine`, `paper`, `paper_alt` | Text, axes, backgrounds |
| Brand accents | `primary` (teal), `secondary` (coral), `muted` | Main series, contrast series, de-emphasis |
| Categorical | `categorical` (8 colours) | Distinct, unordered series |
| Sequential | `sequential` (7-step teal ramp) | Ordered / continuous encodings |
| Semantic | `good`, `warn`, `bad` | Status |

The categorical scale is **Okabe–Ito**, which stays distinguishable under all
common forms of colour-vision deficiency.

### Choosing a scale

- **Distinct lines that must be told apart** → `PALETTE.categorical_cycle(n)`.
  Multi-hue separates adjacent series far better than lightness alone.
- **Ordered lines encoding a continuous variable** → `sequential_line_colors(n)`,
  which samples the darker part of the ramp so the light end stays visible on
  white.  (`PALETTE.sequential_sampled(n)` gives the full ramp, better suited to
  fills and heatmaps than to thin lines.)

## Conventions

- `save_figure()` writes PNG by default; pass `formats=("png", "pdf")` for
  vector output.  It creates parent directories and applies a consistent DPI.
- Figures render identically on a laptop and on an HPC compute node: the theme
  requests only **DejaVu Sans**, which ships with matplotlib, so no system fonts
  are required.
- Figure titles should describe *what is plotted*, not assert a conclusion —
  the reader draws the inference.

## Consumers

`scripts/plot_vocab_tradeoff.py` uses this theme for the parameter /
effective-gene-space surfaces.  Any new plotting script should do the same.
