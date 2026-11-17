"""Does the decoder path the gene ranking relies on behave correctly?

    python -m scripts.decode_sanity --config configs/experiment/p1_dab168_kl1.yaml \
        --root . --run-dir outputs/p_series/p1_dab168_kl1 --outputs-root .

WHY (17 Nov). The ranking decoded with the matched epoch-4 decoder puts T/NK
genes at the disease end, while the labelled cells say the direction points
from lymphoid (control side) to myeloid (disease side): donor position vs
lymphoid share, Spearman -0.93. A decoder that reconstructs T cells from
their latents must predict T genes there. So test the decoder on cells whose
identity is KNOWN before trusting any counterfactual decoded from it.

Three tests, at the selected checkpoint (epoch 4) AND epoch 9 for contrast:
  A  decode the REAL latents of labelled lymphoid and myeloid immune cells;
     lymphoid markers must be higher for lymphoid cells, myeloid markers for
     myeloid cells. If this fails, the decode path itself is broken.
  B  where lymphoid and myeloid cells sit along the disease direction w
     (control minus case, study-centred, family-orthogonalised: identical to
     decode_disease_direction.py), recomputed here, not trusted from a laptop.
  C  the decoded change in those markers for z + w (towards control).
     If A passes and B says lymphoid is on the control side, C must show
     lymphoid markers UP. If it does not, the counterfactual is wrong.

The decode call is copied from decode_disease_direction.py exactly (blanked
covariates, generic gene queries) AND ALSO run through the model's own
decode_once softmax, so a difference between raw decoder scores and what the
model calls predicted expression would show up.
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
MYE = ["LYZ", "CD68", "CD14", "C1QA", "AIF1", "TYROBP"]


def main() -> None:
    import torch
    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager
    from endo_model.evaluation.purity_report import resolve_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--outputs-root", default=".")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-studies", type=int, default=11)
    a = ap.parse_args()

    exp, mcfg, tcfg, dcfg = load_experiment(a.config, root=a.root)
    vocab = GeneVocabulary(str(Path(a.root) / dcfg.vocabulary_path))
    mcfg.n_groups = len(vocab.group_names); mcfg.n_studies = a.n_studies
    data = next(d for d in (Path(a.outputs_root) / "outputs" / "cells" / "p1_dab168_kl1",
                            Path(a.outputs_root) / "Outputs" / "cells" / "p1_dab168_kl1")
                if (d / "latent.npy").exists())
    Z = np.load(data / "latent.npy")
    m = pd.read_csv(data / "metadata.csv.gz", low_memory=False)
    print(f"latent {Z.shape} from {data}")

    lab = np.where(m.study == "marecko_2024", m["celltype"], m["AnnotationMain"]).astype(str)
    lym = np.isin(lab, ["Immune_Lymphoid", "lymphocyte"])
    mye = np.isin(lab, ["Immune_Myeloid", "myeloid"])
    imm = ((m["cell family"] == "immune") & m.disease_status.isin(["control", "eutopic"])).to_numpy()

    # the direction, exactly as decode_disease_direction.py --orthogonalise
    mi = m[imm].reset_index(drop=True)
    w = disease_direction(Z[imm], mi, "study")
    here = Z[(m["cell family"] == "immune").to_numpy()].mean(0)
    for o in ("epithelial", "stromal", "endothelial"):
        ax_ = Z[(m["cell family"] == o).to_numpy()].mean(0) - here
        ax_ /= np.linalg.norm(ax_); w = w - (w @ ax_) * ax_; w /= np.linalg.norm(w)
    print("\nB  position along w (towards CONTROL): lymphoid "
          f"{(Z[imm & lym] @ w).mean():+.3f}, myeloid {(Z[imm & mye] @ w).mean():+.3f}")

    rng = np.random.default_rng(24)
    il = rng.choice(np.flatnonzero(imm & lym), a.n, replace=False)
    iy = rng.choice(np.flatnonzero(imm & mye), a.n, replace=False)

    for ep in ("auto", "checkpoints/epoch_0009_step_0045370.pt"):
        ckpt = resolve_checkpoint(Path(a.root) / a.run_dir, ep)
        model = EndoFoundationModel(vocab, mcfg)
        CheckpointManager.load(str(ckpt), model, map_location="cpu"); model.eval()
        print(f"\n════ {Path(ckpt).name} ════")
        batch = {"study_id": torch.tensor([0]),
                 "is_dns": torch.zeros(1, vocab.vocab_size, dtype=torch.bool)}
        with torch.no_grad():
            gq, gidx = model._build_gene_queries(batch)
            names = [vocab.index_to_gene(int(i)) for i in gidx.tolist()]
            col = {g: names.index(g) for g in LYM + MYE if g in names}

            def dec(z):
                z = torch.from_numpy(np.asarray(z, np.float32))
                B = z.shape[0]; zeros = lambda d: torch.zeros(B, d) if d else None
                raw = model.decoder(gq, model.latent_space.z_proj(z), zeros(mcfg.d_tissue),
                                    zeros(mcfg.d_platform), None, zeros(mcfg.d_study))
                return raw, torch.log_softmax(raw, dim=-1)

            rl, sl = dec(Z[il]); ry, sy = dec(Z[iy])
            print("A  decoded log-expression, lymphoid cells minus myeloid cells "
                  "(should be + for lymphoid markers, - for myeloid):")
            for g, j in col.items():
                kind = "LYM" if g in LYM else "MYE"
                print(f"     {kind} {g:7s} raw {(rl[:, j] - ry[:, j]).mean():+.3f}   "
                      f"log-softmax {(sl[:, j] - sy[:, j]).mean():+.3f}")
            both = np.concatenate([il, iy])
            r0, s0 = dec(Z[both]); r1, s1 = dec(Z[both] + w)
            print("C  decoded change for z + w (towards CONTROL):")
            for g, j in col.items():
                kind = "LYM" if g in LYM else "MYE"
                print(f"     {kind} {g:7s} raw {(r1[:, j] - r0[:, j]).mean():+.4f}   "
                      f"log-softmax {(s1[:, j] - s0[:, j]).mean():+.4f}")


if __name__ == "__main__":
    main()
