"""X12 step B -- which genes does ELAJ move along its disease direction?

Step A (`disease_direction_by_family.py`) found that the control-versus-case
direction in the latent does not generalise globally, but inside the IMMUNE
family it does: held-out donor AUC 0.708 against a shuffled-label null of
0.481, p = 0.015 on 94 donors.

This script takes that direction and asks the decoder what it means in genes.
For immune cells it computes the predicted expression at the cell's own
latent position, then again at that position nudged along the disease
direction, and reports the per-gene difference. Genes that move most are the
genes the model's own representation uses to tell the two apart.

**Nothing here is a differential-expression test.** It never looks at
measured counts. It asks the trained decoder a counterfactual question, which
is why the answer is a statement about ELAJ rather than about the data.

Why it runs on a laptop: the encoder is not needed. The latent is already
dumped, so only the latent-to-decoder path is exercised.

Three decisions worth knowing about:

* **Covariates are blanked.** Tissue, platform and study embeddings are set
  to zero rather than to some arbitrary level. The model was trained with
  `blank_covariates_p = 0.25`, so this is a state it has seen for a quarter of
  training, not an extrapolation. It also makes the answer a statement about
  the latent alone.
* **Gene queries are built per study**, because each study has its own
  measured-gene layout (DNS genes differ), so the loop is over studies and
  the per-gene results are averaged weighted by the cells behind them.
* **The step size is checked.** The decoder is bilinear in the latent, not
  linear, so the directional derivative is only meaningful if the answer is
  stable across epsilon. The script reports the correlation between the gene
  rankings at two step sizes and refuses to write if they disagree.

Usage:
    python -m scripts.decode_disease_direction \\
        --checkpoint /path/to/epoch_XXXX.pt --config configs/experiment/p1.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path("/Users/atharvawaghmare/Endo research/ELAJ")
ARM = "p1_dab168_kl1"
FAMILY = "immune"
EPS = (0.5, 1.0)
MIN_CELLS_PER_DONOR = 50
HECA_GENES = {"GREB1": "up in case", "DKK1": "down in case",
              "IGF1": "up in case", "IGF2": "down in case"}


def disease_direction(Z: np.ndarray, m: pd.DataFrame) -> np.ndarray:
    """Control minus case, on per-donor means, centred within study."""
    codes, uniq = pd.factorize(m["patient_id"])
    M = np.zeros((len(uniq), Z.shape[1]))
    np.add.at(M, codes, Z)
    n = np.bincount(codes, minlength=len(uniq))
    M /= n[:, None]
    info = m.groupby("patient_id").first().reindex(uniq)
    ok = n >= MIN_CELLS_PER_DONOR
    M, info = M[ok], info[ok]
    study = info["study"].to_numpy()
    for s in np.unique(study):
        k = study == s
        M[k] -= M[k].mean(0)
    case = (info["disease_status"] == "eutopic").to_numpy()
    w = M[~case].mean(0) - M[case].mean(0)
    print(f"direction from {len(M)} donors "
          f"({(~case).sum()} control, {case.sum()} case)")
    return w / np.linalg.norm(w)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=str(ROOT / "axw1008/src"))
    ap.add_argument("--cells-per-study", type=int, default=4000)
    ap.add_argument("--transferred", action="store_true",
                    help="use transferred cell families, so all studies take part")
    args = ap.parse_args()

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager

    exp, mcfg, tcfg, dcfg = load_experiment(args.config, root=args.root)
    vocab = GeneVocabulary(str(Path(args.root) / dcfg.vocabulary_path))
    mcfg.n_groups = len(vocab.group_names)
    mcfg.n_studies = len(dcfg.resolve_studies(args.root))

    model = EndoFoundationModel(vocab, mcfg)
    CheckpointManager.load(args.checkpoint, model, map_location="cpu")
    model.eval()
    print(f"loaded {args.checkpoint}")

    Z = np.load(ROOT / f"Outputs/cells/{ARM}/latent.npy")
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False)
    if args.transferred:
        t = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/cell_family_transferred.csv.gz")
        m["family"] = t["cell_family_transferred"].to_numpy()
    else:
        m["family"] = m["cell family"]

    keep = ((m["family"] == FAMILY)
            & m["disease_status"].isin(["control", "eutopic"])).to_numpy()
    m, Z = m[keep].reset_index(drop=True), Z[keep]
    print(f"{FAMILY} cells: {len(m):,} from {m['patient_id'].nunique()} donors")

    w = disease_direction(Z, m)
    assert Z.shape[1] == model.latent_space.z_proj.in_features, (
        f"latent is {Z.shape[1]} wide, z_proj expects "
        f"{model.latent_space.z_proj.in_features}")

    study_to_id = {s.name: s.id for s in dcfg.resolve_studies(args.root)}
    rng = np.random.default_rng(24)
    per_eps: dict[float, pd.Series] = {}

    for eps in EPS:
        acc: dict[str, float] = {}
        wt: dict[str, float] = {}
        for study, sub in m.groupby("study"):
            sid = study_to_id.get(study)
            if sid is None:
                print(f"  skipping {study}: not in the study config")
                continue
            idx = sub.index.to_numpy()
            if len(idx) > args.cells_per_study:
                idx = rng.choice(idx, args.cells_per_study, replace=False)
            z = torch.from_numpy(Z[idx]).float()

            batch = {"study_id": torch.tensor([sid]),
                     "is_dns": torch.zeros(1, len(vocab), dtype=torch.bool)}
            with torch.no_grad():
                gq, gidx = model._build_gene_queries(batch)
                zp = model.latent_space.z_proj(z)
                zp_pert = model.latent_space.z_proj(
                    z + eps * torch.from_numpy(w).float())
                B = z.shape[0]
                zeros = lambda d: torch.zeros(B, d) if d else None
                base = model.decoder(gq, zp, zeros(mcfg.d_tissue),
                                     zeros(mcfg.d_platform), None,
                                     zeros(mcfg.d_study))
                pert = model.decoder(gq, zp_pert, zeros(mcfg.d_tissue),
                                     zeros(mcfg.d_platform), None,
                                     zeros(mcfg.d_study))
                delta = (pert - base).mean(0).numpy()

            names = [vocab.index_to_gene(int(i)) for i in gidx.tolist()]
            for g, d in zip(names, delta):
                acc[g] = acc.get(g, 0.0) + d * len(idx)
                wt[g] = wt.get(g, 0.0) + len(idx)
            print(f"  {study:22s} {len(idx):6d} cells, {len(names):6d} genes")

        per_eps[eps] = pd.Series({g: acc[g] / wt[g] for g in acc}).sort_values()

    a, b = per_eps[EPS[0]], per_eps[EPS[1]]
    shared = a.index.intersection(b.index)
    rho = float(np.corrcoef(a[shared].rank(), b[shared].rank())[0, 1])
    print(f"\nrank correlation between eps={EPS[0]} and eps={EPS[1]}: {rho:.4f}")
    assert rho > 0.95, ("the two step sizes disagree, so the direction is not "
                        "locally linear and this attribution is meaningless")

    s = b.rename("shift_towards_control")
    print(f"\ntop 15 genes UP in control relative to case:\n"
          f"{s.tail(15)[::-1].round(4).to_string()}")
    print(f"\ntop 15 genes UP in case relative to control:\n"
          f"{s.head(15).round(4).to_string()}")

    pct = s.rank(pct=True)
    print("\nHECA's four named genes (their direction, then ours):")
    for g, claim in HECA_GENES.items():
        if g in s.index:
            ours = "up in case" if s[g] < 0 else "up in control"
            print(f"  {g:7s} HECA says {claim:16s} | ours {ours:14s} "
                  f"shift {s[g]:+.4f}, percentile {pct[g]:.3f}")
        else:
            print(f"  {g:7s} not in the measured gene set")

    out = Path(__file__).resolve().parents[2] / "docs" / "paper"
    s.to_csv(out / "x12_gene_shifts.csv")
    print(f"\nwrote {out / 'x12_gene_shifts.csv'}")


if __name__ == "__main__":
    main()
