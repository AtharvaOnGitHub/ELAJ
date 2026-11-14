"""Parameter count as a function of D, G and d, measured not derived.

    python -m scripts.count_params_sweep --run p1_dab168_kl1 --root . \
        --out outputs/analysis/param_sweep.csv

WHY THIS EXISTS. Supplementary Note 1 states

    P(D, G, d) = D*d  +  (D/G)*E(d)  +  C(d)

and the figures that go with it plot both P and the effective gene-space
dimension d*D/G against G. The shape of that equation is readable off the
architecture: the gene embedding is one vector per gene, the groups are K
INDEPENDENT transformers rather than one shared encoder applied K times, and
everything downstream reads a fixed-width aggregation. E(d) and C(d) are not
readable off it, because the transformer layers' feed-forward width is not in
the run config.

So they are measured. This builds the real model at each configuration and
counts its parameters. Nothing is fitted and nothing is assumed, and the
shipped configuration must come back as 1,883,986 or the sweep is wrong and
the figure must not be drawn from it.

NO DATA AND NO CHECKPOINT. The model is constructed from the config and a
synthetic vocabulary of the requested shape, then counted and discarded. It
takes seconds and needs no GPU.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("elaj.paramsweep")

SHIPPED = (15496, 44, 32, 1_883_986)   # D, K, d, parameters. The anchor.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--genes", nargs="+", type=int, default=[12000, 16000, 20000],
                   help="D, genes modelled. The shipped 15,496 is added.")
    p.add_argument("--group-sizes", nargs="+", type=int,
                   default=[250, 350, 500, 700, 900, 1100, 1500, 2000, 2500],
                   help="G, median genes per group. The shipped 352 is added.")
    p.add_argument("--token-dims", nargs="+", type=int, default=[16, 32, 48, 64],
                   help="d, token dimension.")
    p.add_argument("--out", required=True)
    return p.parse_args()


def count(root, config, D: int, G: int, d: int) -> int:
    """Build the model at this configuration and count its parameters."""
    from endo_model.configs import load_experiment
    from endo_model.model.endo_model import EndoFoundationModel

    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(str(config), root=root)
    K = max(1, round(D / G))
    base, extra = divmod(D, K)
    sizes = [base + (1 if i < extra else 0) for i in range(K)]
    names = [f"group_{i:03d}" for i in range(K)]

    model_cfg.d_token = d
    model_cfg.n_groups = K
    # n_studies IS NOT IN THE YAML. benchmark_baselines.py sets it from the
    # resolved study list, and the constructor reads it for the DSBN
    # embedding tables; left None it reaches torch.zeros(None, d) and the
    # error names zeros() rather than the field. Same for n_groups above.
    model_cfg.n_studies = len(data_cfg.resolve_studies(root))

    # FAIL ON THE FIELD, NOT INSIDE TORCH. Every attribute the constructor
    # reads, checked before it is handed over, so a missing one is reported
    # by name instead of surfacing as a tensor-shape error ten frames down.
    required = ("d_token", "d_agg", "d_gauss", "d_dec", "d_platform",
                "d_study", "d_tissue", "d_disease", "n_circular", "n_groups",
                "n_studies", "kappa_floor", "dsbn_learn_affine",
                "circular_parameterisation", "tokens", "group_encoder",
                "aggregation_encoder", "classifier", "drain")
    missing = [f for f in required
               if getattr(model_cfg, f, None) is None]
    if missing:
        raise ValueError("model config fields are unset: "
                         + ", ".join(missing)
                         + ". benchmark_baselines.py sets n_studies and "
                           "n_groups by hand; anything else here is a config "
                           "the sweep must be given rather than invent.")

    # THE FOUR ATTRIBUTES THE MODEL READS, checked against
    # endo_model.py rather than assumed: group_names, group_sizes,
    # vocab_size and global_indices_for_group. The signature is
    # (vocabulary, config) in that order.
    class _Vocab:
        def __init__(self, names, sizes):
            self.group_names = list(names)
            # group_sizes is a dict[str, int] keyed by group name, not a
            # list. ModuleGroupEmbedding calls .items() on it.
            self.group_sizes = dict(zip(names, sizes))
            self.vocab_size = int(sum(sizes))
            start, self._idx = 0, {}
            for n, k in zip(names, sizes):
                self._idx[n] = list(range(start, start + k))
                start += k

        def global_indices_for_group(self, group):
            return self._idx[group]

    model = EndoFoundationModel(_Vocab(names, sizes), model_cfg)
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    args = parse_args()
    config = Path(args.root) / args.config_dir / f"{args.run}.yaml"
    if not config.exists():
        logger.error("no config at %s", config)
        return 1

    D0, K0, d0, P0 = SHIPPED
    try:
        got = count(args.root, config, D0, round(D0 / K0), d0)
    except (TypeError, ValueError, AttributeError) as e:
        logger.error("the model's constructor does not take the arguments this "
                     "script assumes (%s). Read EndoFoundationModel.__init__ "
                     "and fix the call rather than the expectation.", e)
        return 2
    logger.info("anchor: D=%d K=%d d=%d -> %s parameters (shipped %s)",
                D0, K0, d0, f"{got:,}", f"{P0:,}")
    if got != P0:
        logger.error("ANCHOR MISMATCH. The sweep does not reproduce the "
                     "shipped model, so no figure may be drawn from it. "
                     "Difference %s.", f"{got - P0:+,}")
        return 3

    rows = []
    genes = sorted(set(args.genes + [D0]))
    gs = sorted(set(args.group_sizes + [round(D0 / K0)]))
    for D in genes:
        for G in gs:
            for d in args.token_dims:
                K = max(1, round(D / G))
                rows.append({"D": D, "G": G, "d": d, "K": K,
                             "d_effective": d * K,
                             "parameters": count(args.root, config, D, G, d),
                             "is_shipped": int(D == D0 and K == K0 and d == d0)})
        logger.info("  D=%d done", D)

    out = Path(args.root) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", out, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
