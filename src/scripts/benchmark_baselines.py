"""Score scVI, scANVI, Harmony, Scanorama and PCA on the cells ELAJ is scored on.

    python -m scripts.benchmark_baselines --root . --run r8_direction \
        --runs-dir outputs/r8 --splits val test \
        --methods pca harmony scanorama scvi scanvi \
        --gene-policies all --out outputs/benchmarks/r8_all

Built like validate_latent.py: the split is materialised ONCE, every method
embeds the same cells, and one scorer applies the same battery -- kNN
agreement, the cross-donor test with its permutation null, and scIB -- to
every embedding, including ELAJ's own. When two rows of the final table
differ, the difference is the method.

WHICH CELLS. The per-study sampler drops each study's incomplete tail, so the
cells ELAJ embeds are a subset of the split. Every baseline is given exactly
that subset, read out of the sampler, rather than the whole split -- otherwise
the rows would not correspond and no comparison would be fair.

WHICH GENES. `--gene-policies all` hands every method the 15,495-gene
vocabulary ELAJ reads, with unmeasured genes as zero. That is the comparison
on ELAJ's terms. `intersection` removes the genes any study did not measure,
which removes the DNS sentinel's advantage. `hvg2000` is each method's own
best case. Run `all` first; see evaluation/baselines/gene_policy.py.

THE SUPERVISED ROW. scANVI is trained on `--scanvi-label`. Its score on THAT
label is not comparable to anyone else's and the table says so; its scores on
other labels are the fair comparison.

TRANSDUCTIVE OR INDUCTIVE. By default every baseline is fitted on the cells
it is scored on, which is how integration benchmarks are run and is NOT what
ELAJ is asked to do: ELAJ is trained on the training donors and scored on
donors it never saw. `--fit-split train` asks the same of the baselines that
can do it (PCA, scVI, scANVI): fit on the training split, embed the scored
split. Harmony and Scanorama have no out-of-sample map and are reported as
skipped in that mode. The two modes produce separate rows (`scvi/all` and
`scvi/all/train`) and the table keeps them apart.

Nothing here retrains ELAJ. It reads the checkpoint on disk, if `--include-elaj`
is set, and embeds the same cells so the model sits in the same table.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="Experiment name whose data config defines the corpus "
                        "and splits, e.g. r8_direction.")
    p.add_argument("--root", default=None)
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--runs-dir", default=None,
                   help="Where the run directory lives, for --include-elaj.")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--methods", nargs="+",
                   default=["pca", "harmony", "scanorama", "scvi", "scanvi"],
                   help="Any of: pca harmony scanorama scvi scanvi. A method "
                        "whose package is missing is skipped and reported.")
    p.add_argument("--gene-policies", nargs="+", default=["all"],
                   help="all | intersection | hvg<N>. Each is a separate pass.")
    p.add_argument("--scanvi-label", default="cell family",
                   help="Column scANVI is supervised on. Its score on this "
                        "label is flagged in the table.")
    p.add_argument("--fit-split", default=None, choices=["train"],
                   help="Fit the baselines on this split and apply them to "
                        "--splits, as ELAJ is. Methods without an "
                        "out-of-sample map are skipped and reported.")
    p.add_argument("--pca-components", type=int, default=None)
    p.add_argument("--scvi-latent", type=int, default=None)
    p.add_argument("--harmony-pcs", type=int, default=None,
                   help="Default 50. Pass 16 for a width-matched row.")
    p.add_argument("--scanorama-dimred", type=int, default=None,
                   help="Default 100. Pass 16 for a width-matched row.")
    p.add_argument("--include-elaj", action="store_true",
                   help="Embed the checkpoint on the same cells and score it "
                        "alongside. Needs --runs-dir to find it.")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--k-cross-donor", type=int, default=100)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--scvi-epochs", type=int, default=None,
                   help="Override scvi-tools' own epoch heuristic.")
    p.add_argument("--out", required=True,
                   help="Directory for benchmark.csv, benchmark.md, settings.json.")
    return p.parse_args()


def _method_kwargs(name: str, args) -> dict:
    """Per-method construction arguments taken from the command line.

    Only arguments the user set are passed, so a method's own default stays
    its default and is what the settings file records.
    """
    given = {
        "pca": {"n_components": args.pca_components},
        "harmony": {"n_pcs": args.harmony_pcs},
        "scanorama": {"dimred": args.scanorama_dimred},
        "scvi": {"n_latent": args.scvi_latent, "max_epochs": args.scvi_epochs},
        "scanvi": {"n_latent": args.scvi_latent, "max_epochs": args.scvi_epochs,
                   "label_column": args.scanvi_label},
    }.get(name, {})
    return {k: v for k, v in given.items() if v is not None}


def _build_corpus(root, exp_cfg, train_cfg, data_cfg, vocab, study_configs,
                  splits, orders, obs_columns, gene_names, names):
    """One CorpusMatrix over the sampler order of each split, concatenated."""
    from endo_model.evaluation.baselines import CorpusMatrix, extract_corpus_matrix
    from endo_model.evaluation.baselines.scorer import add_harmonised_labels
    from scripts.report_purity import build_loader
    from scripts.validate_latent import sampler_order

    parts = []
    for split in splits:
        loader = build_loader(root, exp_cfg, train_cfg, data_cfg, vocab,
                              study_configs, extra_obs_columns=obs_columns,
                              split=split)
        # Whether or not ELAJ was embedded, use the sampler's order: it is the
        # set of cells the model CAN embed, so the table is comparable to any
        # validate_latent run over the same splits.
        order = orders.get(split) or sampler_order(loader)
        parts.append(extract_corpus_matrix(loader.dataset, gene_names,
                                           names, indices=order))
    corpus = CorpusMatrix.concatenate(parts)
    add_harmonised_labels(corpus)
    return corpus


def _embed_elaj(root, exp_cfg, model_cfg, train_cfg, data_cfg, vocab,
                study_configs, run_dir, splits, device, obs_columns):
    """ELAJ's combined latent over the same sampler order the corpus uses.

    Returns (latent, orders) where `orders[split]` is the dataset index list
    the loader yielded, so the caller can build the corpus over exactly these
    cells. Reuses validate_latent's collect and sampler_order rather than
    re-implementing the embedding pass.
    """
    import numpy as np

    from endo_model.evaluation.latent_validation import circular_coordinates
    from endo_model.evaluation.purity_report import best_checkpoint
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager
    from scripts.report_purity import build_loader
    from scripts.validate_latent import collect, sampler_order

    ckpt = best_checkpoint(run_dir)
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    CheckpointManager.load(str(ckpt), model, None, map_location=device)
    logger.info("ELAJ checkpoint: %s", ckpt)

    parts, orders = [], {}
    for split in splits:
        loader = build_loader(root, exp_cfg, train_cfg, data_cfg, vocab,
                              study_configs, extra_obs_columns=obs_columns,
                              split=split)
        order = sampler_order(loader)
        got = collect(model, loader, device, len(order))
        if got is None:
            continue
        mu_gauss, mu_angle = got[0], got[1]
        parts.append(np.concatenate([mu_gauss, circular_coordinates(mu_angle)],
                                    axis=-1))
        orders[split] = order
    return (np.concatenate(parts) if parts else None), orders


def main() -> None:
    args = parse_args()
    root = args.root or str(Path.cwd())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.baselines import (
        BenchmarkReport, Embedding, EmbeddingScorer, MissingDependency,
        ScorerSettings, TransductiveOnly, build_method, policy_from_name,
    )
    from endo_model.evaluation.baselines.scorer import required_obs_columns
    from scripts.report_purity import resolve_run_dir

    config = Path(root) / args.config_dir / f"{args.run}.yaml"
    if not config.exists():
        logger.error("no config at %s", config)
        sys.exit(1)
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(str(config), root=root)
    vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))
    gene_names = [vocab.index_to_gene(i) for i in range(vocab.vocab_size)]
    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
        for s in data_cfg.resolve_studies(root)
    ]
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)
    names = {s["id"]: s["name"] for s in study_configs}
    obs_columns = required_obs_columns()
    logger.info("Loading %d obs columns: %s", len(obs_columns), ", ".join(obs_columns))

    # ── ELAJ first, if asked, because it fixes which cells everyone gets ────
    elaj_latent, orders = None, {}
    if args.include_elaj:
        import torch

        run_dir = resolve_run_dir(root, exp_cfg.output_dir, args.runs_dir)
        if not run_dir.is_dir():
            logger.error("no run directory at %s", run_dir)
            sys.exit(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elaj_latent, orders = _embed_elaj(
            root, exp_cfg, model_cfg, train_cfg, data_cfg, vocab,
            study_configs, run_dir, args.splits, device, obs_columns,
        )

    # ── the corpus: one CorpusMatrix over the same cells, in the same order ──
    corpus = _build_corpus(root, exp_cfg, train_cfg, data_cfg, vocab,
                           study_configs, args.splits, orders, obs_columns,
                           gene_names, names)
    logger.info("Corpus: %d cells x %d genes, %d studies, splits %s",
                corpus.n_cells, corpus.n_genes, corpus.n_studies, corpus.split)

    # ── the fit corpus, if the baselines are to be held out like ELAJ ──────
    fit_corpus = None
    if args.fit_split:
        fit_corpus = _build_corpus(root, exp_cfg, train_cfg, data_cfg, vocab,
                                   study_configs, [args.fit_split], {},
                                   obs_columns, gene_names, names)
        logger.info("Fit corpus: %d cells x %d genes, %d studies, split %s",
                    fit_corpus.n_cells, fit_corpus.n_genes,
                    fit_corpus.n_studies, fit_corpus.split)

    scorer = EmbeddingScorer(ScorerSettings(
        k=args.k, k_cross_donor=args.k_cross_donor,
        permutations=args.permutations, seed=args.seed,
    ))
    report = BenchmarkReport()

    if elaj_latent is not None:
        if elaj_latent.shape[0] != corpus.n_cells:
            logger.error("ELAJ embedded %d cells but the corpus holds %d",
                         elaj_latent.shape[0], corpus.n_cells)
            sys.exit(1)
        logger.info("── elaj (checkpoint) ──")
        # ELAJ is always fitted on the training donors; say so in its key so
        # it lines up with the baselines' inductive rows.
        emb = Embedding(method="elaj", latent=elaj_latent, fit_seconds=0.0,
                        settings={"checkpoint": str(run_dir)}, fit_on="train")
        report.add(scorer.score(emb, corpus, gene_policy="native"))
        report.record_settings("elaj", "native", emb.settings, fit_on="train")

    # ── every policy x every method ─────────────────────────────────────────
    fit_tag = args.fit_split or ""
    for policy_name in args.gene_policies:
        policy = policy_from_name(policy_name)
        try:
            # The gene set is chosen on the fit corpus when there is one, so
            # an HVG list is selected without looking at the scored cells,
            # then applied to both so their gene axes match.
            chosen = fit_corpus if fit_corpus is not None else corpus
            mask = policy.select(chosen)
            restricted = corpus.restrict_genes(mask)
            restricted_fit = (fit_corpus.restrict_genes(mask)
                              if fit_corpus is not None else None)
        except ImportError as exc:
            report.record_skip(f"*/{policy.name}", str(exc))
            continue
        logger.info("Gene policy %-12s keeps %6d of %d genes",
                    policy.name, restricted.n_genes, corpus.n_genes)
        for method_name in args.methods:
            method = build_method(method_name, seed=args.seed,
                                  **_method_kwargs(method_name, args))
            key = "/".join(p for p in (method.name, policy.name, fit_tag) if p)
            if not method.available():
                report.record_skip(key, f"needs {', '.join(method.missing())}")
                continue
            try:
                emb = method.embed(restricted, fit_corpus=restricted_fit)
            except (MissingDependency, TransductiveOnly) as exc:
                report.record_skip(key, str(exc))
                continue
            except Exception as exc:                      # noqa: BLE001
                # One method failing must not cost the others their results.
                logger.exception("%s failed on %s", method.name, policy.name)
                report.record_skip(key, f"{type(exc).__name__}: {exc}")
                continue
            supervised = (args.scanvi_label if method_name == "scanvi" else "")
            report.add(scorer.score(emb, restricted, gene_policy=policy.name,
                                    supervised_on=supervised))
            report.record_settings(method.name, policy.name, emb.settings,
                                   fit_on=emb.fit_on)
            # Write after every method so a killed job still leaves a table.
            report.write_csv(out / "benchmark.csv")
            report.write_markdown(out / "benchmark.md")
            report.write_settings(out / "settings.json")

    report.write_csv(out / "benchmark.csv")
    report.write_markdown(out / "benchmark.md")
    report.write_settings(out / "settings.json")
    logger.info("\n%s", report.comparison_table())
    logger.info("Written to %s", out)


if __name__ == "__main__":
    main()
