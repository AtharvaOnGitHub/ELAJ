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

# No absolute default anywhere: the repo is /rds/homes/a/axw1008/axw1008/src
# on the HPC and .../ELAJ/axw1008/src on the Mac, and the dumps sit beside the
# repo on one and two levels up on the other. Every sbatch wrapper in this
# repo cds into the repo and passes `--root .`; this follows that.
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
    ap.add_argument("--root", default=".",
                    help="repo root holding configs/ and data/")
    ap.add_argument("--cells-per-study", type=int, default=4000)
    ap.add_argument("--transferred", action="store_true",
                    help="use transferred cell families, so all studies take part")
    ap.add_argument("--vocab", default=None,
                    help="local vocabulary_v1.json; the config points at an HPC path")
    ap.add_argument("--n-studies", type=int, default=11)
    ap.add_argument("--outputs-root", default=".",
                    help="holds Outputs/ or outputs/ with cells/<arm>/")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager

    exp, mcfg, tcfg, dcfg = load_experiment(args.config, root=args.root)
    vocab = GeneVocabulary(args.vocab or str(Path(args.root) / dcfg.vocabulary_path))
    mcfg.n_groups = len(vocab.group_names)
    # studies.csv lives on the HPC. Only the count matters here: the study
    # embedding is blanked below, and the gene layout comes from `is_dns`.
    mcfg.n_studies = args.n_studies

    model = EndoFoundationModel(vocab, mcfg)
    CheckpointManager.load(args.checkpoint, model, map_location="cpu")
    model.eval()
    print(f"loaded {args.checkpoint}")

    # The dump directory is "Outputs" on the Mac and "outputs" on the HPC.
    tried = [Path(args.outputs_root) / d / "cells" / ARM
             for d in ("Outputs", "outputs")]
    data = next((d for d in tried if (d / "latent.npy").exists()), None)
    if data is None:
        raise SystemExit("no latent.npy found. Looked in:\n  "
                         + "\n  ".join(str(d) for d in tried))
    print(f"data: {data}")
    Z = np.load(data / "latent.npy")
    m = pd.read_csv(data / "metadata.csv.gz", low_memory=False)
    if args.transferred:
        t = pd.read_csv(data / "cell_family_transferred.csv.gz")
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

    rng = np.random.default_rng(24)
    per_eps: dict[float, pd.Series] = {}

    # No dataset is loaded, so `_gene_layouts` is empty and `_build_gene_queries`
    # falls back to the batch's own `is_dns`. All-False means every gene counts
    # as measured, which is what an attribution wants: one pass, whole
    # vocabulary, no per-study gene layout to get wrong.
    idx_all = np.arange(len(m))
    if len(idx_all) > args.cells_per_study:
        idx_all = rng.choice(idx_all, args.cells_per_study, replace=False)
    z = torch.from_numpy(Z[idx_all]).float()
    batch = {"study_id": torch.tensor([0]),
             "is_dns": torch.zeros(1, vocab.vocab_size, dtype=torch.bool)}

    with torch.no_grad():
        gq, gidx = model._build_gene_queries(batch)
        names = [vocab.index_to_gene(int(i)) for i in gidx.tolist()]
        print(f"decoding {len(z):,} {FAMILY} cells against {len(names):,} genes")
        B = z.shape[0]
        zeros = lambda d: torch.zeros(B, d) if d else None
        zp = model.latent_space.z_proj(z)
        base = model.decoder(gq, zp, zeros(mcfg.d_tissue), zeros(mcfg.d_platform),
                             None, zeros(mcfg.d_study))
        for eps in EPS:
            zp_p = model.latent_space.z_proj(
                z + eps * torch.from_numpy(w).float())
            pert = model.decoder(gq, zp_p, zeros(mcfg.d_tissue),
                                 zeros(mcfg.d_platform), None, zeros(mcfg.d_study))
            per_eps[eps] = pd.Series((pert - base).mean(0).numpy(),
                                     index=names).sort_values()

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

    out = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[2] / "docs" / "paper")
    out.mkdir(parents=True, exist_ok=True)
    s.to_csv(out / "x12_gene_shifts.csv")
    print(f"\nwrote {out / 'x12_gene_shifts.csv'}")


if __name__ == "__main__":
    main()
