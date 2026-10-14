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

AND THAT VOCABULARY IS NOT THE UNION. `build_vocabulary.py --drop-unassigned`
keeps 15,495 of the 30,962 measured genes, discarding every gene that matched
no curated pathway, because ELAJ's group encoder needs each gene to sit in a
coherent group. Giving scVI that set gives it a curation step it would not
have in the field. `--baseline-vocabulary <wide.json>` puts the baselines on
their own axis over every measured gene while ELAJ keeps its checkpoint's own,
and the gene count goes into the row key so the two tables cannot be mixed by
`--reuse-baselines`. See evaluation/baselines/gene_axis.py.

THE SUPERVISED ROW. scANVI is trained on `--scanvi-label`. Its score on THAT
label is not comparable to anyone else's and the table says so; its scores on
other labels are the fair comparison.

REUSING BASELINE ROWS. `--reuse-baselines <benchmark.csv>` copies an earlier
run's baseline rows into the report and runs only the methods it lacks, so
scoring a new ELAJ arm against fixed baselines costs the ELAJ embedding and
its scoring, not another scVI training. The earlier run must be on the same
corpus (same experiment data config, splits, seed); nothing here checks
that, so the arm's job passes the file its campaign owns.

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
                   help="Any of: pca harmony scanorama scvi scanvi, or the "
                        "single word 'none' to fit nothing. Use 'none' for a "
                        "pass that only scores --external-latent rows beside "
                        "a --reuse-baselines file: naming methods that are all "
                        "being reused achieves nothing and risks a tag "
                        "mismatch making them pending, which is how a "
                        "1.16M-cell fit corpus got built for nobody. A method "
                        "whose package is missing is skipped and reported.")
    p.add_argument("--gene-policies", nargs="+", default=["all"],
                   help="all | intersection | hvg<N>. Each is a separate pass.")
    p.add_argument("--external-latent", nargs="+", default=[],
                   help="Directories written by scripts/embed_pretrained.py, "
                        "each holding latent.npy and manifest.json. For models "
                        "that cannot run in this interpreter: Geneformer, "
                        "scGPT and scBERT each need their own venv. The row "
                        "name and the gene coverage come from the manifest, "
                        "and the rows are joined by corpus row position under "
                        "a fingerprint of the corpus, rather than by assuming "
                        "a shared order.")
    p.add_argument("--baseline-vocabulary", default=None,
                   help="A vocabulary.json built WITHOUT --drop-unassigned, "
                        "giving the baselines every measured gene instead of "
                        "the pathway-annotated half ELAJ's group encoder "
                        "requires. ELAJ still reads its checkpoint's own "
                        "vocabulary. The gene count goes into the policy tag, "
                        "so a wide row cannot collide with a narrow one.")
    p.add_argument("--scanvi-label", default="cell family",
                   help="Column scANVI is supervised on. Its score on this "
                        "label is flagged in the table.")
    p.add_argument("--reuse-baselines", default=None,
                   help="A benchmark.csv from an earlier run on the same corpus. "
                        "Its baseline rows are copied in and those methods are "
                        "not run again; only methods absent from it run. The "
                        "elaj row is never reused.")
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
    p.add_argument("--checkpoint", default="auto",
                   help="Checkpoint to score, or 'auto' for the run's best by "
                        "val_nb. An explicit choice is RECORDED in the output "
                        "settings, because an epoch picked after reading its "
                        "label metrics cannot be described as selected by "
                        "reconstruction loss alone.")
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


NO_METHODS = "none"
"""`--methods none`: fit nothing, score only what is reused or handed in.

The combination this exists for is `--reuse-baselines` plus
`--external-latent`, where every fitted row already exists and the only work is
scoring a pretrained model's latent. Before it, that pass had to name the
methods it was NOT going to run, so that `pending_keys` could tick them off,
and a key that did not match the reuse file turned them all pending again.
"""


def requested_methods(args) -> list[str]:
    """`args.methods`, with the `none` sentinel resolved to an empty list."""
    if [m.strip().lower() for m in args.methods] == [NO_METHODS]:
        return []
    return list(args.methods)


def pending_keys(args, wide_axis, already) -> list[str]:
    """Row keys this run would compute, after `--reuse-baselines` is applied.

    WHY THIS IS WORKED OUT BEFORE THE FIT CORPUS IS BUILT. `--fit-split train`
    builds a 1,161,472-cell CorpusMatrix, and on 12 October 2026 a job did that
    alongside `--reuse-baselines` supplying every requested row, reached
    111.9 GB of a 128G allocation, and was OOM-killed eleven minutes in over a
    corpus no method was going to read.

    The policy tag is computable from the vocabulary alone, so this needs
    neither corpus.
    """
    fit_tag = args.fit_split or ""
    keys = []
    for policy_name in args.gene_policies:
        tag = (wide_axis.tag(policy_name) if wide_axis else policy_name)
        for method_name in requested_methods(args):
            key = "/".join(p for p in (method_name, tag, fit_tag) if p)
            if key not in already:
                keys.append(key)
    return keys


def build_corpus(root, exp_cfg, train_cfg, data_cfg, vocab, study_configs,
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
                study_configs, run_dir, splits, device, obs_columns,
                checkpoint="auto"):
    """ELAJ's combined latent over the same sampler order the corpus uses.

    Returns (latent, orders, ckpt) where `orders[split]` is the dataset index
    list the loader yielded, so the caller can build the corpus over exactly
    these cells, and `ckpt` is the checkpoint actually scored so it can be
    recorded in the output rather than living only in the log. Reuses
    validate_latent's collect and sampler_order rather than re-implementing
    the embedding pass.
    """
    import numpy as np

    from endo_model.evaluation.latent_validation import circular_coordinates
    from endo_model.evaluation.purity_report import resolve_checkpoint
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager
    from scripts.report_purity import build_loader
    from scripts.validate_latent import collect, sampler_order

    ckpt = resolve_checkpoint(run_dir, checkpoint)
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
    return (np.concatenate(parts) if parts else None), orders, ckpt


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
    elaj_latent, orders, elaj_ckpt = None, {}, None
    if args.include_elaj:
        import torch

        run_dir = resolve_run_dir(root, exp_cfg.output_dir, args.runs_dir)
        if not run_dir.is_dir():
            logger.error("no run directory at %s", run_dir)
            sys.exit(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elaj_latent, orders, elaj_ckpt = _embed_elaj(
            root, exp_cfg, model_cfg, train_cfg, data_cfg, vocab,
            study_configs, run_dir, args.splits, device, obs_columns,
            checkpoint=args.checkpoint,
        )

    # ── the corpus: one CorpusMatrix over the same cells, in the same order ──
    corpus = build_corpus(root, exp_cfg, train_cfg, data_cfg, vocab,
                           study_configs, args.splits, orders, obs_columns,
                           gene_names, names)
    logger.info("Corpus: %d cells x %d genes, %d studies, splits %s",
                corpus.n_cells, corpus.n_genes, corpus.n_studies, corpus.split)

    # ── the baselines' own gene axis, if they are not to read ELAJ's ───────
    # ELAJ keeps the vocabulary its checkpoint was trained against, because an
    # embedding table has no row for a gene that was not in it. The baselines
    # have no such constraint and are given every measured gene.
    wide_axis = None
    baseline_corpus = corpus
    if args.baseline_vocabulary:
        from endo_model.evaluation.baselines.gene_axis import (
            WideGeneAxis, require_same_cells,
        )

        wide_axis = WideGeneAxis(str(Path(root) / args.baseline_vocabulary)
                                 if not Path(args.baseline_vocabulary).is_absolute()
                                 else args.baseline_vocabulary)
        baseline_corpus = build_corpus(
            root, exp_cfg, train_cfg, data_cfg, wide_axis.vocabulary,
            study_configs, args.splits, orders, obs_columns,
            wide_axis.gene_names, names,
        )
        require_same_cells(corpus, baseline_corpus)
        logger.info("Baseline corpus: %d cells x %d genes (ELAJ reads %d)",
                    baseline_corpus.n_cells, baseline_corpus.n_genes,
                    corpus.n_genes)

    scorer = EmbeddingScorer(ScorerSettings(
        k=args.k, k_cross_donor=args.k_cross_donor,
        permutations=args.permutations, seed=args.seed,
    ))
    report = (BenchmarkReport.read_csv(args.reuse_baselines, exclude_methods=("elaj",))
              if args.reuse_baselines else BenchmarkReport())
    already = set(report.methods())

    # ── the fit corpus, if the baselines are to be held out like ELAJ ──────
    # Built on the BASELINES' axis: it is what they fit on, and a fit corpus on
    # a different gene axis from the scored corpus is not a comparison at all.
    #
    # AND ONLY IF SOMETHING WILL READ IT. It is 1,161,472 cells, which is the
    # single most expensive object this script builds; see pending_keys.
    pending = pending_keys(args, wide_axis, already)
    fit_corpus = None
    if args.fit_split and pending:
        fit_vocab = wide_axis.vocabulary if wide_axis else vocab
        fit_gene_names = wide_axis.gene_names if wide_axis else gene_names
        fit_corpus = build_corpus(root, exp_cfg, train_cfg, data_cfg, fit_vocab,
                                   study_configs, [args.fit_split], {},
                                   obs_columns, fit_gene_names, names)
        logger.info("Fit corpus: %d cells x %d genes, %d studies, split %s",
                    fit_corpus.n_cells, fit_corpus.n_genes,
                    fit_corpus.n_studies, fit_corpus.split)
    elif args.fit_split:
        logger.info(
            "--fit-split %s given but every requested row is already in %s, "
            "so the fit corpus is not built.", args.fit_split,
            args.reuse_baselines,
        )

    if elaj_latent is not None:
        if elaj_latent.shape[0] != corpus.n_cells:
            logger.error("ELAJ embedded %d cells but the corpus holds %d",
                         elaj_latent.shape[0], corpus.n_cells)
            sys.exit(1)
        logger.info("── elaj (checkpoint) ──")
        # ELAJ is always fitted on the training donors; say so in its key so
        # it lines up with the baselines' inductive rows.
        # The CHECKPOINT, not the run directory. Recording the directory
        # made two scoring passes of the same arm indistinguishable in the
        # output, which was survivable while the epoch was always the ranked
        # best and is not now that --checkpoint exists.
        settings = {"checkpoint": str(elaj_ckpt),
                    "selection": ("best val_nb" if args.checkpoint in
                                  (None, "", "auto") else "EXPLICIT")}
        emb = Embedding(method="elaj", latent=elaj_latent, fit_seconds=0.0,
                        settings=settings, fit_on="train")
        report.add(scorer.score(emb, corpus, gene_policy="native"))
        report.record_settings("elaj", "native", emb.settings, fit_on="train")

    # ── latents computed elsewhere, in their own environments ──────────────
    # Scored against `corpus` rather than `baseline_corpus`: the labels are
    # identical and a pretrained model reads its OWN gene vocabulary, so
    # neither gene axis here describes what it saw. Its coverage of our genes
    # is in the manifest instead.
    for directory in args.external_latent:
        from endo_model.evaluation.baselines.external_latent import ExternalLatent

        try:
            external = ExternalLatent.read(Path(directory))
            emb, scored = external.embedding_for(corpus)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("external latent %s: %s", directory, exc)
            report.record_skip(f"{Path(directory).name}/native", str(exc))
            continue
        logger.info("── %s (external, %d dims, fitted on %s) ──",
                    emb.method, emb.width, emb.fit_on or "the scored cells")
        report.add(scorer.score(emb, scored, gene_policy="native"))
        report.record_settings(emb.method, "native", emb.settings,
                               fit_on=emb.fit_on)
        report.write_csv(out / "benchmark.csv")
        report.write_markdown(out / "benchmark.md")
        report.write_settings(out / "settings.json")

    # ── every policy x every method ─────────────────────────────────────────
    # Skipped entirely when nothing is pending, so an HVG policy does not run
    # scanpy's gene selection for a set of methods that are all being reused.
    fit_tag = args.fit_split or ""
    if not pending:
        logger.info("No method left to run; every requested row was reused.")
    for policy_name in (args.gene_policies if pending else []):
        policy = policy_from_name(policy_name)
        try:
            # The gene set is chosen on the fit corpus when there is one, so
            # an HVG list is selected without looking at the scored cells,
            # then applied to both so their gene axes match.
            chosen = fit_corpus if fit_corpus is not None else baseline_corpus
            mask = policy.select(chosen)
            restricted = baseline_corpus.restrict_genes(mask)
            restricted_fit = (fit_corpus.restrict_genes(mask)
                              if fit_corpus is not None else None)
        except ImportError as exc:
            report.record_skip(f"*/{policy.name}", str(exc))
            continue
        # The tag carries the gene count when the baselines are on their own
        # axis, so `scvi/all/train` and `scvi/all30962/train` are two rows.
        policy_tag = wide_axis.tag(policy.name) if wide_axis else policy.name
        logger.info("Gene policy %-12s keeps %6d of %d genes",
                    policy_tag, restricted.n_genes, baseline_corpus.n_genes)
        for method_name in requested_methods(args):
            method = build_method(method_name, seed=args.seed,
                                  **_method_kwargs(method_name, args))
            key = "/".join(p for p in (method.name, policy_tag, fit_tag) if p)
            if key in already:
                logger.info("── %s reused from %s", key, args.reuse_baselines)
                continue
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
            if wide_axis:
                emb.settings["gene_vocabulary"] = wide_axis.path
                emb.settings["n_genes"] = restricted.n_genes
            report.add(scorer.score(emb, restricted, gene_policy=policy_tag,
                                    supervised_on=supervised))
            report.record_settings(method.name, policy_tag, emb.settings,
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
