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
    """Per-method construction arguments taken from the command line."""
    if name == "scvi":
        return {"max_epochs": args.scvi_epochs}
    if name == "scanvi":
        return {"label_column": args.scanvi_label, "max_epochs": args.scvi_epochs}
    return {}


def _embed_elaj(root, exp_cfg, model_cfg, train_cfg, data_cfg, vocab,
                study_configs, run_dir, splits, device, obs_columns):
    """ELAJ's combined latent over the same sampler order the corpus uses.

    Returns (latent, orders) where `orders[split]` is the dataset index list
    the loader yielded, so the caller can build the corpus over exactly these
    cells. Reuses validate_latent's collect and sampler_order rather than
    re-implementing the embedding pass.
    """
    import numpy as np
    import torch

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
    import numpy as np

    args = parse_args()
    root = args.root or str(Path.cwd())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.baselines import (
        BenchmarkReport, CorpusMatrix, Embedding, EmbeddingScorer,
        MissingDependency, ScorerSettings, build_method, extract_corpus_matrix,
        policy_from_name,
    )
    from endo_model.evaluation.baselines.scorer import (
        add_harmonised_labels, required_obs_columns,
    )
    from scripts.report_purity import build_loader, resolve_run_dir
    from scripts.validate_latent import sampler_order

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
    parts = []
    for split in args.splits:
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
    logger.info("Corpus: %d cells x %d genes, %d studies, splits %s",
                corpus.n_cells, corpus.n_genes, corpus.n_studies, corpus.split)

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
        emb = Embedding(method="elaj", latent=elaj_latent, fit_seconds=0.0,
                        settings={"checkpoint": str(run_dir)})
        report.add(scorer.score(emb, corpus, gene_policy="native"))
        report.record_settings("elaj", "native", emb.settings)

    # ── every policy x every method ─────────────────────────────────────────
    for policy_name in args.gene_policies:
        policy = policy_from_name(policy_name)
        try:
            restricted = policy.apply(corpus)
        except ImportError as exc:
            report.record_skip(f"*/{policy.name}", str(exc))
            continue
        for method_name in args.methods:
            method = build_method(method_name, seed=args.seed,
                                  **_method_kwargs(method_name, args))
            if not method.available():
                report.record_skip(f"{method.name}/{policy.name}",
                                   f"needs {', '.join(method.missing())}")
                continue
            try:
                emb = method.embed(restricted)
            except MissingDependency as exc:
                report.record_skip(f"{method.name}/{policy.name}", str(exc))
                continue
            except Exception as exc:                      # noqa: BLE001
                # One method failing must not cost the others their results.
                logger.exception("%s failed on %s", method.name, policy.name)
                report.record_skip(f"{method.name}/{policy.name}",
                                   f"{type(exc).__name__}: {exc}")
                continue
            supervised = (args.scanvi_label if method_name == "scanvi" else "")
            report.add(scorer.score(emb, restricted, gene_policy=policy.name,
                                    supervised_on=supervised))
            report.record_settings(method.name, policy.name, emb.settings)
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
