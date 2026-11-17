"""X12 step B, done the way the model decodes: real cells, real covariates.

    python -m scripts.decode_real --run p1_dab168_kl1 --root . --outputs-root .

WHY THIS REPLACES decode_disease_direction.py (17 Nov). That script decoded
dumped latents with ALL-ZERO covariates. decode_sanity.py (job 62036623)
showed the result inverts cell identity: with the selected epoch-4 decoder,
labelled lymphoid cells decode LOWER on CD3D/CD3E/NKG7 and HIGHER on
LYZ/CD68/C1QA than myeloid cells. The decoder is

    log_mu(gene) = gain(cell) * (k(cell) . q(gene)) + b,   gain = W_out . W_v(context)

and `context` is the latent AND the covariates, so `gain` is one signed
number per cell that scales the whole profile. Changing the covariates
changes gain; if it changes sign, every gene's relative level flips.

THIS SCRIPT encodes real held-out cells through the trained model with their
OWN tissue, platform, study and library size, and uses the model's own probe
`decode_once(..., z_proj=...)`, which substitutes the latent and leaves
everything else as the model saw it. Steps, each printed:

  1  checkpoint = the run's selected best (resolve_checkpoint), epoch 4.
  2  every immune cell (control/eutopic, labelled lymphoid or myeloid), all
     splits: cell family and disease are not model inputs, so no split is
     privileged (29,355 cells, duempelmann + marecko).
  3  each cell's encoded mu_gauss is compared with its row of the dumped
     latent: confirms the epoch and the row mapping. Stops if they disagree.
  4  TEST A with real covariates: lymphoid markers higher in lymphoid cells,
     myeloid markers higher in myeloid cells. Stops if it fails.
  5  for six directions (centring study/none/controls x orthogonalised or
     not, the same definitions as decode_disease_direction.py) the per-gene
     mean of log(mu(z + eps*w)) - log(mu(z)), eps 1 and 0.5, over cells where
     the gene is measured.
  6  how often zeroing the covariates flips the sign of gain.

Writes x12_gene_shifts{_orth}_realcov_centre-<c>_epoch_0004.csv into
docs/paper (relative to --root), column shift_towards_control, as before.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.decode_disease_direction import disease_direction  # noqa: E402

LYM = ["CD3D", "CD3E", "NKG7", "GNLY", "CD2", "KLRD1"]
# Not TYROBP: NK cells express it strongly, and NK cells are lymphoid here.
MYE = ["LYZ", "CD68", "CD14", "C1QA", "AIF1", "CSF1R"]
CENTRINGS = ("study", "none", "controls")


def orthogonalised(w, Z, fam):
    here = Z[fam == "immune"].mean(0)
    for o in ("epithelial", "stromal", "endothelial"):
        a = Z[fam == o].mean(0) - here
        a /= np.linalg.norm(a)
        w = w - (w @ a) * a
        w /= np.linalg.norm(w)
    return w


def main() -> None:
    import torch
    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager
    from endo_model.evaluation.purity_report import resolve_checkpoint
    from scripts.report_purity import build_loader
    from scripts.validate_latent import sampler_order

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p1_dab168_kl1")
    ap.add_argument("--root", default=".")
    ap.add_argument("--runs-dir", default="outputs/p_series")
    ap.add_argument("--outputs-root", default=".")
    ap.add_argument("--n-cells", type=int, default=0,
                    help="0 = every labelled immune cell, all splits (the default). "
                         "Cell family and disease status are never model inputs, so "
                         "held-out cells buy nothing here (AW, standing rule).")
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    exp, mcfg, tcfg, dcfg = load_experiment(str(Path(a.root) / "configs/experiment" / f"{a.run}.yaml"),
                                            root=a.root)
    vocab = GeneVocabulary(str(Path(a.root) / dcfg.vocabulary_path))
    study_configs = [{"id": s.id, "name": s.name, "path": str(Path(a.root) / s.path)}
                     for s in dcfg.resolve_studies(a.root)]
    mcfg.n_groups = len(vocab.group_names); mcfg.n_studies = len(study_configs)
    ckpt = resolve_checkpoint(Path(a.root) / a.runs_dir / a.run, "auto")
    model = EndoFoundationModel(vocab, mcfg)
    CheckpointManager.load(str(ckpt), model, map_location="cpu"); model.eval()
    ck_tag = Path(ckpt).stem.split("_step")[0]
    print(f"1  checkpoint {ckpt}")

    # ── the dump: labels, directions ─────────────────────────────────────
    data = Path(a.outputs_root) / "outputs" / "cells" / a.run
    Z = np.load(data / "latent.npy")
    m = pd.read_csv(data / "metadata.csv.gz", low_memory=False)
    fam = m["cell family"].astype(str).to_numpy()
    lab = np.where(m.study == "marecko_2024", m["celltype"], m["AnnotationMain"]).astype(str)
    kind = np.where(np.isin(lab, ["Immune_Lymphoid", "lymphocyte"]), "LYM",
                    np.where(np.isin(lab, ["Immune_Myeloid", "myeloid"]), "MYE", ""))
    imm = (fam == "immune") & m.disease_status.isin(["control", "eutopic"]).to_numpy()
    mi = m[imm].reset_index(drop=True)
    W = {}
    for c in CENTRINGS:
        w = disease_direction(Z[imm], mi, c)
        W[(c, False)] = w
        W[(c, True)] = orthogonalised(w.copy(), Z, fam)
    wl = W[("study", True)]
    print(f"   position along w (study, orth; towards CONTROL): LYM {(Z[imm & (kind == 'LYM')] @ wl).mean():+.3f}, "
          f"MYE {(Z[imm & (kind == 'MYE')] @ wl).mean():+.3f}")

    # ── 2: which dump rows, and their dataset indices ────────────────────
    split = m.split.astype(str).to_numpy()
    # The dump's blocks run val, train, test (checked 17 Nov). Nothing here
    # assumes an order: each split's dump rows are matched to that split's
    # own sampler order, and step 3 checks every cell's latent.
    rng = np.random.default_rng(24)
    cand = np.flatnonzero(imm & (kind != ""))
    rows = cand if a.n_cells <= 0 else np.sort(rng.choice(cand, min(a.n_cells, len(cand)), replace=False))
    print(f"2  {len(rows)} immune cells, all splits ({(kind[rows] == 'LYM').sum()} lymphoid, "
          f"{(kind[rows] == 'MYE').sum()} myeloid)")

    acc = {key: [np.zeros(vocab.vocab_size), np.zeros(vocab.vocab_size)] for key in W}
    acc_half = [np.zeros(vocab.vocab_size), np.zeros(vocab.vocab_size)]
    marker = {g: {"LYM": [], "MYE": []} for g in LYM + MYE}
    gidx_of = {g: vocab.gene_to_index(g) for g in LYM + MYE if g in vocab}
    max_diff, flips, negs, n_seen = 0.0, 0, 0, 0

    for sp in ("train", "val", "test"):
        loader = build_loader(a.root, exp, tcfg, dcfg, vocab, study_configs, split=sp)
        order = sampler_order(loader)
        dump_pos = np.flatnonzero(split == sp)
        assert len(order) == len(dump_pos), (sp, len(order), len(dump_pos))
        want = rows[np.isin(rows, dump_pos)]
        local = np.searchsorted(dump_pos, want)
        ds = loader.dataset; collate = loader.collate_fn
        # `study_ids` (plural) is a property that rebuilds a list of every
        # cell on each access, so it is read once per split.
        sid_of = np.asarray(ds.study_ids, dtype=np.int64)
        by_study = {}
        for r, p in zip(want, local):
            by_study.setdefault(int(sid_of[order[p]]), []).append((r, order[p]))
        for sid, items in by_study.items():
            for i in range(0, len(items), a.batch):
                chunk = items[i:i + a.batch]
                batch = collate([ds[j] for _, j in chunk])
                rr = np.array([r for r, _ in chunk])
                lat = model.encode(batch)
                mu = lat["mu_gauss"]
                # 3: the encoded latent must be the dumped latent
                d = float((mu - torch.from_numpy(Z[rr]).float()).abs().max())
                max_diff = max(max_diff, d)
                # GPU dump vs CPU recompute differ by rounding (TF32 on the A100),
                # ~1e-4; a wrong epoch or a shifted row differs by whole units.
                if d > 0.05:
                    raise SystemExit(f"STOP: encoded latent differs from the dump by {d:.4g} "
                                     f"(study {sid}); wrong checkpoint or row mapping")
                gq, gidx = model._build_gene_queries(batch)
                gidx = gidx.numpy()
                zp = lambda z: model.latent_space.z_proj(z)
                base = model.decode_once(batch, lat, gq, z_proj=zp(mu))
                lb = torch.log(base.clamp_min(1e-12))
                # 4: markers on the real decode
                col = {g: np.flatnonzero(gidx == j) for g, j in gidx_of.items()}
                for g, cpos in col.items():
                    if len(cpos):
                        for k_ in ("LYM", "MYE"):
                            sel = torch.from_numpy(kind[rr] == k_)
                            if bool(sel.any()):
                                marker[g][k_].extend(lb[sel, int(cpos[0])].tolist())
                # 5: the six directions
                for key, w in W.items():
                    wt = torch.from_numpy(w).float()
                    lp = torch.log(model.decode_once(batch, lat, gq, z_proj=zp(mu + wt)).clamp_min(1e-12))
                    acc[key][0][gidx] += (lp - lb).sum(0).numpy(); acc[key][1][gidx] += len(rr)
                    if key == ("study", True):
                        lh = torch.log(model.decode_once(batch, lat, gq, z_proj=zp(mu + .5 * wt)).clamp_min(1e-12))
                        acc_half[0][gidx] += (lh - lb).sum(0).numpy(); acc_half[1][gidx] += len(rr)
                # 6: gain with real and with zeroed covariates
                dec = model.decoder
                tis = model.decoder_tissue(batch["tissue_levels"])
                pla = model.decoder_platform(batch["platform_codes"]) if model.decoder_platform is not None else None
                stu = model.decoder_study(batch["study_id"]) if model.decoder_study is not None else None
                ctx_r = dec._context(zp(mu), tis, pla, None, stu)
                ctx_b = dec._context(zp(mu), torch.zeros_like(tis), None if pla is None else torch.zeros_like(pla),
                                     None, None if stu is None else torch.zeros_like(stu))
                g_r = dec.W_v(ctx_r) @ dec.W_out.weight.squeeze(0)
                g_b = dec.W_v(ctx_b) @ dec.W_out.weight.squeeze(0)
                flips += int((torch.sign(g_r) != torch.sign(g_b)).sum()); negs += int((g_r < 0).sum())
                n_seen += len(rr)
        print(f"   {sp}: {len(want)} cells done", flush=True)
        del loader, ds, order, sid_of              # one split's dataset in memory at a time

    print(f"3  encoded vs dumped latent, max abs difference {max_diff:.2e} (rounding only; stops above 0.05)")
    print("4  TEST A, real covariates: mean decoded log-expression, lymphoid minus myeloid cells")
    right = {"LYM": 0, "MYE": 0}; tested = {"LYM": 0, "MYE": 0}
    for g in LYM + MYE:
        k_ = "LYM" if g in LYM else "MYE"
        if marker[g]["LYM"] and marker[g]["MYE"]:
            dlt = np.mean(marker[g]["LYM"]) - np.mean(marker[g]["MYE"])
            good = dlt > 0 if k_ == "LYM" else dlt < 0
            right[k_] += good; tested[k_] += 1
            print(f"     {k_} {g:7s} {dlt:+.3f}  {'ok' if good else 'WRONG'}")
    # at least 5 of 6 per side: one marker can disagree for a biological
    # reason; an inverted decode gets them all wrong, as decode_sanity did
    ok = all(tested[k] >= 5 and right[k] >= tested[k] - 1 for k in right)
    print(f"   lymphoid markers right {right['LYM']}/{tested['LYM']}, myeloid {right['MYE']}/{tested['MYE']}")
    print(f"6  gain: negative for {negs}/{n_seen} cells with real covariates; "
          f"sign flips when covariates are zeroed for {flips}/{n_seen}")
    if not ok:
        raise SystemExit("STOP: TEST A fails with real covariates too (needs >= 5/6 per side); no ranking written")

    out = Path(a.root) / "docs" / "paper"
    names = np.array([vocab.index_to_gene(i) for i in range(vocab.vocab_size)])
    for (c, orth), (s, n) in acc.items():
        keep = n > 0
        ser = pd.Series(s[keep] / n[keep], index=names[keep], name="shift_towards_control").sort_values()
        f = out / f"x12_gene_shifts{'_orth' if orth else ''}_realcov_centre-{c}_{ck_tag}.csv"
        ser.to_csv(f); print(f"   wrote {f.name}  (top towards disease: {' '.join(ser.index[:8])})")
    s1 = acc[("study", True)]; keep = (s1[1] > 0) & (acc_half[1] > 0)
    rho = pd.Series(s1[0][keep] / s1[1][keep]).rank().corr(pd.Series(acc_half[0][keep] / acc_half[1][keep]).rank())
    print(f"5  step-size check, study/orth, eps 1 vs 0.5: rank correlation {rho:.4f}")


if __name__ == "__main__":
    main()
