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


CENTRING = ("study", "none", "controls")


def disease_direction(Z: np.ndarray, m: pd.DataFrame, centring: str = "study") -> np.ndarray:
    """Control minus case, on per-donor means.

    centring (AW, 17 Nov, asked to see all three side by side):
      study     subtract each study's mean donor. The original. Each study's
                own control-minus-case difference survives exactly; only the
                gap BETWEEN studies is removed.
      none      no centring. The between-study gap stays in, and with
                duempelmann 59% cases against marecko 28% it reads as disease.
      controls  subtract each study's mean CONTROL donor, i.e. stratified by
                disease status: studies aligned on controls only, so no case
                signal enters the subtraction.
    """
    assert centring in CENTRING, centring
    codes, uniq = pd.factorize(m["patient_id"])
    M = np.zeros((len(uniq), Z.shape[1]))
    np.add.at(M, codes, Z)
    n = np.bincount(codes, minlength=len(uniq))
    M /= n[:, None]
    info = m.groupby("patient_id").first().reindex(uniq)
    ok = n >= MIN_CELLS_PER_DONOR
    M, info = M[ok], info[ok]
    study = info["study"].to_numpy()
    case = (info["disease_status"] == "eutopic").to_numpy()
    for s in np.unique(study):
        k = study == s
        if centring == "study":
            M[k] -= M[k].mean(0)
        elif centring == "controls":
            M[k] -= M[k & ~case].mean(0)
    w = M[~case].mean(0) - M[case].mean(0)
    print(f"direction from {len(M)} donors "
          f"({(~case).sum()} control, {case.sum()} case), centring={centring}")
    return w / np.linalg.norm(w)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="auto",
                    help="'auto' = the run's selected best (purity_report."
                         "resolve_checkpoint), the rule the latent dump used. The "
                         "wrapper once passed the NEWEST file (ls -t), which is "
                         "the last epoch, not the one the dumped latent came from.")
    ap.add_argument("--run-dir", default="outputs/p_series/p1_dab168_kl1",
                    help="for --checkpoint auto")
    ap.add_argument("--centring", nargs="+", default=["study"], choices=CENTRING)
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
    ap.add_argument("--orthogonalise", action="store_true",
                    help="remove the cell-identity component from the direction")
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

    from endo_model.evaluation.purity_report import resolve_checkpoint
    ckpt = resolve_checkpoint(Path(args.root) / args.run_dir, args.checkpoint)
    model = EndoFoundationModel(vocab, mcfg)
    CheckpointManager.load(str(ckpt), model, map_location="cpu")
    model.eval()
    ck_tag = Path(ckpt).stem.split("_step")[0]          # e.g. epoch_0004
    print(f"loaded {ckpt}")

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

    out_files = []
    for centring in args.centring:
        print(f"\n══ centring = {centring} ══")
        w = disease_direction(Z, m, centring)

        if args.orthogonalise:
            # The raw direction has a component along the cell-identity axes, and
            # identity genes respond far more steeply to those axes than anything
            # responds to disease, so they dominate the ranking without tracking
            # disease at all. Project them out: Gram-Schmidt against each
            # (other family - FAMILY) axis, using the FULL metadata, not the
            # disease-restricted subset.
            full = pd.read_csv(data / "metadata.csv.gz", low_memory=False)
            full["family"] = (pd.read_csv(data / "cell_family_transferred.csv.gz")
                              ["cell_family_transferred"].to_numpy()
                              if args.transferred else full["cell family"])
            Zfull = np.load(data / "latent.npy")
            here = Zfull[(full["family"] == FAMILY).to_numpy()].mean(0)
            for other in ("epithelial", "stromal", "endothelial"):
                k = (full["family"] == other).to_numpy()
                if not k.any():
                    continue
                a = Zfull[k].mean(0) - here
                a /= np.linalg.norm(a)
                before = float(w @ a)
                w = w - before * a
                w /= np.linalg.norm(w)
                print(f"  removed {FAMILY}->{other:12s} component, cosine was {before:+.4f}")
            print(f"  residual cosines: " + ", ".join(
                f"{o} {float(w @ ((Zfull[(full['family'] == o).to_numpy()].mean(0) - here) / np.linalg.norm(Zfull[(full['family'] == o).to_numpy()].mean(0) - here))):+.4f}"
                for o in ("epithelial", "stromal", "endothelial")))
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
        tag = "_orth" if args.orthogonalise else ""
        # A NEW NAME per centring and checkpoint, so the ranking the paper
        # uses (x12_gene_shifts_orth.csv) is never overwritten by a variant.
        name = f"x12_gene_shifts{tag}_centre-{centring}_{ck_tag}.csv"
        s.to_csv(out / name)
        out_files.append(out / name)
        print(f"\nwrote {out / name}")




if __name__ == "__main__":
    main()
