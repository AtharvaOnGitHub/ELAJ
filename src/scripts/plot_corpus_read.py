"""Figures for the corpus-as-one-object read, drawn from the written tables.

    python -m scripts.plot_corpus_read --stats <validate out dir> --out <dir>

NO MATPLOTLIB AND NO CLUSTER. It reads level_statistics.csv and writes SVG by
hand, so it runs on a laptop with numpy alone. That is deliberate: two
twelve-hour GPU jobs were killed inside their figure stage with every
measurement already on disk (58239008, 58380379). Figures are the cheapest
thing to compute and the most expensive place to lose a GPU allocation, so the
cluster writes tables and this draws them afterwards.

THE QUESTION IT ANSWERS is not whether a label is recoverable. It is: the
corpus is 1,577,728 cells as sixteen numbers each, so group the cells by
whatever metadata exists and describe the arrangement -- where each group
sits, how far from the others, and in what layout.

EVERY DISTANCE IS DIVIDED BY THE GROUPS' OWN SCATTER. A centroid distance of
1.2 means nothing on its own; 1.2 between clouds that are each 4 wide means
the groups overlap almost completely. The ratio is the figure's unit.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

PHASE_ORDER = ("menstrual", "proliferative_early", "proliferative",
               "proliferative_late", "periovulatory", "secretory_early",
               "secretory_early_mid", "secretory_mid", "secretory_late")
LADDER = ("AnnotationRefined", "AnnotationMain", "cell family",
          "menstrual phase", "severity (study vocabularies)", "phase",
          "Endometriosis_stage", "disease_status", "EndometriosisGrade")
PRETTY = {"phase": "cell cycle (G1/S/G2M)",
          "severity (study vocabularies)": "severity, both vocabularies",
          "Endometriosis_stage": "rASRM stage (+control)",
          "EndometriosisGrade": "disease grade (CTL/mild/severe)",
          "AnnotationMain": "cell type, 5 classes",
          "AnnotationRefined": "cell type, 63 classes",
          "disease_status": "disease status"}


def groups_for(stats: Path, label: str) -> dict:
    """Centroid, n and scatter per level, merged from the per-donor rows."""
    dims = [f"dim{i}" for i in range(16)]
    acc: dict[str, dict] = {}
    with open(stats / "level_statistics.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["label"] != label:
                continue
            a = acc.setdefault(r["level"], {"n": 0, "s": None, "sq": 0.0})
            if a["s"] is None:
                a["s"] = np.zeros(len(dims))
            a["n"] += int(r["n"])
            a["sq"] += float(r["sumsq"])
            a["s"] += np.array([float(r[d]) for d in dims])
    out = {}
    for k, v in acc.items():
        c = v["s"] / v["n"]
        out[k] = {"c": c, "n": v["n"],
                  "scatter": float(np.sqrt(
                      max(v["sq"] - v["n"] * float(c @ c), 0.0) / v["n"]))}
    return out


def widest(g: dict) -> tuple:
    """The pair furthest apart RELATIVE TO THEIR OWN WIDTH, and that ratio."""
    best, ratio = None, -1.0
    names = sorted(g)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(np.linalg.norm(g[a]["c"] - g[b]["c"]))
            w = (g[a]["scatter"] + g[b]["scatter"]) / 2
            if w and d / w > ratio:
                ratio, best = d / w, (a, b, d)
    return best, ratio


def mds(dist: np.ndarray) -> np.ndarray:
    """Classical MDS to two dimensions, by hand. No sklearn needed."""
    n = len(dist)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (dist ** 2) @ j
    w, v = np.linalg.eigh(b)
    idx = np.argsort(w)[::-1][:2]
    return v[:, idx] * np.sqrt(np.maximum(w[idx], 0))


def esc(t: str) -> str:
    return (str(t).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


# ── panels ──────────────────────────────────────────────────────────────────

def ladder_panel(stats: Path) -> str:
    """One bar per metadata column: how separated its groups are, in widths."""
    rows = []
    for label in LADDER:
        g = groups_for(stats, label)
        if len(g) < 2:
            continue
        pair, ratio = widest(g)
        rows.append((PRETTY.get(label, label), ratio, len(g),
                     sum(x["n"] for x in g.values()), pair))
    rows.sort(key=lambda r: -r[1])

    w, left, top, bh, gap = 920, 300, 64, 30, 10
    h = top + len(rows) * (bh + gap) + 86
    xmax = max(r[1] for r in rows) * 1.18
    def sx(v):
        return left + v / xmax * (w - left - 150)

    p = [f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg"',
         ' aria-label="Separation of metadata groups in the latent, in units of',
         ' the groups own scatter">',
         f'<text x="24" y="28" class="t">What the space is organised by</text>',
         f'<text x="24" y="48" class="s">Widest pair of groups, divided by how '
         f'wide those groups themselves are. Above 1, the groups are further '
         f'apart than they are broad.</text>']
    # the 1.0 reference, which is where separation starts to mean anything
    p.append(f'<line x1="{sx(1):.1f}" y1="{top - 6}" x2="{sx(1):.1f}" '
             f'y2="{top + len(rows) * (bh + gap) - gap + 4}" class="ref"/>')
    p.append(f'<text x="{sx(1):.1f}" y="{top - 12}" class="ax" '
             f'text-anchor="middle">1.0 = as far apart as they are wide</text>')
    for i, (name, ratio, nlev, ncell, pair) in enumerate(rows):
        y = top + i * (bh + gap)
        cls = "bar hi" if ratio >= 1 else "bar lo"
        p.append(f'<rect x="{left}" y="{y}" width="{max(sx(ratio) - left, 1):.1f}" '
                 f'height="{bh}" rx="3" class="{cls}"/>')
        p.append(f'<text x="{left - 12}" y="{y + bh * 0.62}" class="lb" '
                 f'text-anchor="end">{esc(name)}</text>')
        p.append(f'<text x="{sx(ratio) + 8:.1f}" y="{y + bh * 0.62}" '
                 f'class="val">{ratio:.2f}</text>')
        p.append(f'<text x="{left + 8}" y="{y + bh * 0.62}" class="inb">'
                 f'{nlev} levels &#183; {ncell:,} cells</text>')
    p.append('</svg>')
    return "".join(p)


def cycle_panel(stats: Path) -> str:
    """The phase centroids in 2-D, with the cycle drawn through them in order."""
    g = groups_for(stats, "menstrual phase")
    names = [n for n in PHASE_ORDER if n in g]
    cent = np.stack([g[n]["c"] for n in names])
    dist = np.linalg.norm(cent[:, None] - cent[None], axis=-1)
    xy = mds(dist)

    w, h, pad = 620, 470, 92
    lo, hi = xy.min(0), xy.max(0)
    span = np.maximum(hi - lo, 1e-9)
    px = pad + (xy[:, 0] - lo[0]) / span[0] * (w - 2 * pad)
    py = h - pad - (xy[:, 1] - lo[1]) / span[1] * (h - 2 * pad - 30)
    rad = [6 + 20 * math.sqrt(g[n]["n"] / max(x["n"] for x in g.values()))
           for n in names]

    p = [f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg"',
         ' aria-label="The nine menstrual phase centroids projected to two',
         ' dimensions, joined in cycle order">',
         '<text x="22" y="26" class="t">The cycle, as the latent lays it out</text>',
         '<text x="22" y="46" class="s">Nine phase centroids, projected to 2-D. '
         'Line joins them in cycle order; circle area is cell count.</text>']
    for i in range(len(names) - 1):
        p.append(f'<line x1="{px[i]:.1f}" y1="{py[i]:.1f}" x2="{px[i+1]:.1f}" '
                 f'y2="{py[i+1]:.1f}" class="path"/>')
    # the wrap, dashed: it is the step the data does NOT support
    p.append(f'<line x1="{px[-1]:.1f}" y1="{py[-1]:.1f}" x2="{px[0]:.1f}" '
             f'y2="{py[0]:.1f}" class="wrap"/>')
    for i, n in enumerate(names):
        hue = "mid" if "secretory" in n else ("pro" if "prolifer" in n
                                             else ("peri" if n == "periovulatory"
                                                   else "men"))
        p.append(f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="{rad[i]:.1f}" '
                 f'class="pt {hue}"/>')
        dy = -rad[i] - 7 if py[i] > h / 2 else rad[i] + 15
        p.append(f'<text x="{px[i]:.1f}" y="{py[i] + dy:.1f}" class="pl" '
                 f'text-anchor="middle">{esc(n.replace("_", " "))}</text>')
    p.append(f'<text x="22" y="{h - 16}" class="s">Dashed: late secretory back '
             f'to menstrual. The arc does not close — reading the levels as '
             f'a cycle scores WORSE than as a line.</text>')
    p.append('</svg>')
    return "".join(p)


def heat_panel(stats: Path) -> str:
    """The 9x9 phase distance matrix, rows and columns in cycle order."""
    g = groups_for(stats, "menstrual phase")
    names = [n for n in PHASE_ORDER if n in g]
    cent = np.stack([g[n]["c"] for n in names])
    d = np.linalg.norm(cent[:, None] - cent[None], axis=-1)

    cell, left, top = 44, 176, 112
    w = left + len(names) * cell + 30
    h = top + len(names) * cell + 40
    vmax = d.max()
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg"',
         ' aria-label="Distance between every pair of menstrual phase centroids">',
         '<text x="22" y="26" class="t">Every pair, in cycle order</text>',
         '<text x="22" y="46" class="s">Darker is closer. A recovered cycle '
         'would darken along the diagonal and at the two far corners.</text>']
    for i, a in enumerate(names):
        p.append(f'<text x="{left - 10}" y="{top + i * cell + cell * 0.64}" '
                 f'class="hl" text-anchor="end">{esc(a.replace("_", " "))}</text>')
        p.append(f'<text x="{left + i * cell + cell / 2}" y="{top - 10}" '
                 f'class="hl" transform="rotate(-45 '
                 f'{left + i * cell + cell / 2} {top - 10})" '
                 f'text-anchor="start">{esc(a.replace("_", " ")[:14])}</text>')
        for j in range(len(names)):
            v = d[i, j] / vmax
            op = 0.07 + 0.88 * (1 - v)
            p.append(f'<rect x="{left + j * cell}" y="{top + i * cell}" '
                     f'width="{cell - 2}" height="{cell - 2}" rx="2" '
                     f'class="hc" opacity="{op:.3f}"/>')
            if i != j:
                p.append(f'<text x="{left + j * cell + cell / 2 - 1}" '
                         f'y="{top + i * cell + cell * 0.63}" class="hv" '
                         f'text-anchor="middle">{d[i, j]:.1f}</text>')
    p.append('</svg>')
    return "".join(p)


def severity_panel(stats: Path) -> str:
    """Severity, pooled and split by study, on one distance axis."""
    g = groups_for(stats, "severity (study vocabularies)")
    merged = {"control": ["stage_0", "grade_ctl"],
              "mild": ["stage_1", "stage_2", "grade_mild"],
              "severe": ["stage_3", "stage_4", "grade_severe"]}

    def combine(levels):
        levels = [l for l in levels if l in g]
        n = sum(g[l]["n"] for l in levels)
        c = sum(g[l]["c"] * g[l]["n"] for l in levels) / n
        return {"c": c, "n": n}

    pooled = {k: combine(v) for k, v in merged.items()}
    by_study = {}
    for study, pre in (("marecko", "stage_"), ("duempelmann", "grade_")):
        for lvl, mem in merged.items():
            sel = [m for m in mem if m.startswith(pre)]
            if any(s in g for s in sel):
                by_study[f"{study} {lvl}"] = combine(sel)

    def pairs(d):
        ks = sorted(d)
        return [(a, b, float(np.linalg.norm(d[a]["c"] - d[b]["c"])))
                for i, a in enumerate(ks) for b in ks[i + 1:]]

    pp, ps = pairs(pooled), pairs(by_study)
    allv = [v for *_, v in pp + ps]
    w, left, top, bh, gap = 920, 300, 86, 24, 8
    rowsn = len(pp) + len(ps) + 2
    h = top + rowsn * (bh + gap) + 40
    xmax = max(allv) * 1.15
    def sx(v): return left + v / xmax * (w - left - 120)

    p = [f'<svg viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg"',
         ' aria-label="Distances between severity groups, pooled and split by study">',
         '<text x="24" y="26" class="t">Disease severity is a step, not a gradient'
         ' — and study is bigger than both</text>',
         '<text x="24" y="46" class="s">Distance between group centroids. '
         'Having endometriosis moves a donor; how much of it they have barely '
         'does. Two studies’ controls are further apart than any disease '
         'contrast within either.</text>']
    y = top
    for title, data in (("pooled across both studies", pp),
                        ("kept apart by study", ps)):
        p.append(f'<text x="24" y="{y + 14}" class="sec">{esc(title)}</text>')
        y += 26
        for a, b, v in sorted(data, key=lambda r: r[2]):
            cross = a.split()[0] != b.split()[0] and " " in a
            cls = "bar cross" if cross else "bar hi"
            p.append(f'<rect x="{left}" y="{y}" '
                     f'width="{max(sx(v) - left, 1):.1f}" height="{bh}" rx="3" '
                     f'class="{cls}"/>')
            p.append(f'<text x="{left - 12}" y="{y + bh * 0.72}" class="lb" '
                     f'text-anchor="end">{esc(a)} &#8596; {esc(b)}</text>')
            p.append(f'<text x="{sx(v) + 8:.1f}" y="{y + bh * 0.72}" '
                     f'class="val">{v:.2f}</text>')
            y += bh + gap
        y += 10
    p.append('</svg>')
    return "".join(p)


CSS = """
:root{--ink:#17201f;--mut:#5d6b69;--ground:#fbfaf7;--rule:#d8d4ca;
 --hi:#2f7d72;--lo:#b9bdb4;--cross:#b4603e;--pro:#3f7fa6;--peri:#c08a2e;
 --mid:#2f7d72;--men:#8a5a86;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
 --ink:#eceae4;--mut:#9aa6a3;--ground:#14191a;--rule:#39403f;
 --hi:#59b3a4;--lo:#4a524f;--cross:#d98a63;--pro:#6aa9cd;--peri:#dbab54;
 --mid:#59b3a4;--men:#b185ad;}}
:root[data-theme=dark]{--ink:#eceae4;--mut:#9aa6a3;--ground:#14191a;
 --rule:#39403f;--hi:#59b3a4;--lo:#4a524f;--cross:#d98a63;--pro:#6aa9cd;
 --peri:#dbab54;--mid:#59b3a4;--men:#b185ad;}
body{background:var(--ground);color:var(--ink);margin:0;
 font:14px/1.55 "Iowan Old Style",Palatino,Georgia,serif;}
.wrap{max-width:1000px;margin:0 auto;padding:40px 24px 72px;}
h1{font-size:27px;line-height:1.2;margin:0 0 6px;letter-spacing:-.01em;}
h2{font-size:15px;font-weight:600;margin:46px 0 4px;letter-spacing:.06em;
 text-transform:uppercase;color:var(--mut);}
p{margin:10px 0;max-width:66ch;}
p.lede{font-size:16px;color:var(--mut);margin-bottom:28px;max-width:70ch;}
figure{margin:18px 0 6px;overflow-x:auto;}
svg{max-width:100%;height:auto;display:block;}
figcaption{color:var(--mut);font-size:13px;margin-top:6px;max-width:70ch;}
.t{font:600 16px/1.2 ui-sans-serif,system-ui,sans-serif;fill:var(--ink);}
.s{font:12.5px/1.35 ui-sans-serif,system-ui,sans-serif;fill:var(--mut);}
.sec{font:600 12px ui-sans-serif,system-ui,sans-serif;fill:var(--mut);
 letter-spacing:.06em;text-transform:uppercase;}
.lb{font:13px ui-sans-serif,system-ui,sans-serif;fill:var(--ink);}
.ax,.pl{font:11.5px ui-sans-serif,system-ui,sans-serif;fill:var(--mut);}
.val{font:600 13px ui-sans-serif,system-ui,sans-serif;fill:var(--ink);
 font-variant-numeric:tabular-nums;}
.inb{font:11.5px ui-sans-serif,system-ui,sans-serif;fill:var(--ground);
 opacity:.92;}
.hl{font:11px ui-sans-serif,system-ui,sans-serif;fill:var(--mut);}
.hv{font:11px ui-sans-serif,system-ui,sans-serif;fill:var(--ink);
 font-variant-numeric:tabular-nums;opacity:.85;}
.bar.hi{fill:var(--hi);} .bar.lo{fill:var(--lo);}
.bar.cross{fill:var(--cross);}
.ref{stroke:var(--ink);stroke-width:1.5;stroke-dasharray:4 3;opacity:.45;}
.path{stroke:var(--mut);stroke-width:2;opacity:.6;}
.wrap-line,.wrap{stroke:var(--cross);stroke-width:2;stroke-dasharray:6 4;
 opacity:.75;}
.pt{stroke:var(--ground);stroke-width:1.5;}
.pt.pro{fill:var(--pro);} .pt.peri{fill:var(--peri);}
.pt.mid{fill:var(--mid);} .pt.men{fill:var(--men);}
.hc{fill:var(--hi);}
table{border-collapse:collapse;margin:14px 0;font-size:13.5px;
 font-variant-numeric:tabular-nums;}
th,td{padding:5px 14px 5px 0;text-align:left;border-bottom:1px solid var(--rule);}
th{font:600 11.5px ui-sans-serif,system-ui,sans-serif;color:var(--mut);
 letter-spacing:.05em;text-transform:uppercase;}
td.n{text-align:right;}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="p1_dab168_kl1")
    a = ap.parse_args()
    stats, out = Path(a.stats), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    panels = {"separation_ladder": ladder_panel(stats),
              "phase_cycle": cycle_panel(stats),
              "phase_matrix": heat_panel(stats),
              "severity_step": severity_panel(stats)}
    for name, svg in panels.items():
        (out / f"{name}.svg").write_text(svg, encoding="utf-8")
        print(f"wrote {out / name}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
