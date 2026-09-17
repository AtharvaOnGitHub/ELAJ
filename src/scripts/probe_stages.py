"""Score every stage of ELAJ's encoder on the cells the benchmark scored.

    python -m scripts.probe_stages --root . --run r8_direction \
        --runs-dir outputs/r8 --splits val test --out outputs/probes/r8_stages

THE QUESTION. The benchmark put the latent below 16-component PCA on cell
family. The latent is the last of four per-cell tensors the encoder makes;
this scores all four, in order, under two references for what the encoder
was given (per-group PCA of the counts, raw as ELAJ reads them and
log-normalised as everyone else does), so the row where the family signal
falls is the stage that lost it. It also scores every group on its own,
three ways (raw PCA, log PCA, encoder output), so the table says which
pathway groups carry the signal, whether the transformer over a group added
anything a linear reading could not see, and what the raw-count
tokenisation costs before any attention runs.

SAME CELLS, SAME SCORER. The corpus is built over the sampler order exactly
as benchmark_baselines.py builds it, and the scorer is the benchmark's, so
every row here can be read next to that table. The `latent` row IS the
benchmark's elaj/native row and must reproduce it; if it does not, nothing
else on the page can be trusted.

Group stages get a shorter battery (family and phase agreement) than the
pipeline stages, because 135 of them at the full battery would run for
days. `--no-groups` skips them. scIB is off unless `--scib` is passed: its
silhouette materialises the cell-by-cell distance matrix and ran a
1,440-wide stage out of 120 GB.
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
logger = logging.getLogger("elaj.probe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="Experiment name, e.g. r8_direction.")
    p.add_argument("--root", default=None)
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--runs-dir", default=None,
                   help="Where the run directory lives.")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--no-groups", action="store_true",
                   help="Score the five pipeline stages only.")
    p.add_argument("--scib", action="store_true",
                   help="Run the scIB battery on the pipeline stages too. "
                        "Needs memory for a full distance matrix at the "
                        "widest stage.")
    p.add_argument("--no-scib", action="store_true",
                   help="Accepted for compatibility; scIB is already off.")
    p.add_argument("--reference-components", type=int, default=32,
                   help="Width of the per-group PCA references; 32 matches "
                        "d_token so the searches are in spaces of equal size.")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--k-cross-donor", type=int, default=100)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--out", required=True,
                   help="Directory for stages.csv, stages.md, settings.json.")
    return p.parse_args()


def build_readers(group_names: list[str], groups: bool):
    from endo_model.evaluation.stages import group_readers, pipeline_readers

    readers = pipeline_readers(group_names)
    if groups:
        readers += group_readers(group_names)
    return readers


def collect_all(model, root, exp_cfg, train_cfg, data_cfg, vocab, study_configs,
                splits, device, obs_columns, readers):
    """StageEmbeddings over every split, plus the sampler order per split."""
    from endo_model.evaluation.stages import StageCollector, StageEmbeddings
    from scripts.report_purity import build_loader
    from scripts.validate_latent import sampler_order

    collector = StageCollector(readers)
    parts, orders = [], {}
    for split in splits:
        loader = build_loader(root, exp_cfg, train_cfg, data_cfg, vocab,
                              study_configs, extra_obs_columns=obs_columns,
                              split=split)
        order = sampler_order(loader)
        if not order:
            continue
        parts.append(collector.collect(model, loader, device, len(order)))
        orders[split] = order
    return StageEmbeddings.concatenate(parts), orders


def build_corpus(root, exp_cfg, train_cfg, data_cfg, vocab, study_configs,
                 orders, obs_columns, gene_names, names):
    from endo_model.evaluation.baselines import CorpusMatrix, extract_corpus_matrix
    from endo_model.evaluation.baselines.scorer import add_harmonised_labels
    from scripts.report_purity import build_loader

    parts = []
    for split, order in orders.items():
        loader = build_loader(root, exp_cfg, train_cfg, data_cfg, vocab,
                              study_configs, extra_obs_columns=obs_columns,
                              split=split)
        parts.append(extract_corpus_matrix(loader.dataset, gene_names, names,
                                           indices=order))
    corpus = CorpusMatrix.concatenate(parts)
    add_harmonised_labels(corpus)
    return corpus


def main() -> None:
    import torch

    args = parse_args()
    root = args.root or str(Path.cwd())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.baselines import BenchmarkReport, ScorerSettings
    from endo_model.evaluation.baselines.scorer import required_obs_columns
    from endo_model.evaluation.purity_report import best_checkpoint
    from endo_model.evaluation.stages import (
        ProbeSettings, StageEmbeddings, StageProbe, StageTables,
    )
    from endo_model.evaluation.stages.references import (
        LOG, RAW, GroupPCAReference, group_masks_from_vocabulary,
    )
    from endo_model.model.endo_model import EndoFoundationModel
    from endo_model.training.checkpointing import CheckpointManager
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

    run_dir = resolve_run_dir(root, exp_cfg.output_dir, args.runs_dir)
    if not run_dir.is_dir():
        logger.error("no run directory at %s", run_dir)
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = best_checkpoint(run_dir)
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    CheckpointManager.load(str(ckpt), model, None, map_location=device)
    logger.info("Checkpoint: %s", ckpt)

    readers = build_readers(list(vocab.group_names), groups=not args.no_groups)
    logger.info("%d stages to collect", len(readers))
    embeddings, orders = collect_all(
        model, root, exp_cfg, train_cfg, data_cfg, vocab, study_configs,
        args.splits, device, obs_columns, readers,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    corpus = build_corpus(root, exp_cfg, train_cfg, data_cfg, vocab,
                          study_configs, orders, obs_columns, gene_names, names)
    logger.info("Corpus: %d cells, %d studies, splits %s",
                corpus.n_cells, corpus.n_studies, corpus.split)

    # The input references, from the counts, ahead of the collected stages:
    # raw first, then log-normalised, then what the model computed. Per-group
    # references only if the group stages were asked for.
    masks = group_masks_from_vocabulary(vocab, corpus.n_genes)
    staged = StageEmbeddings()
    for transform in (RAW, LOG):
        reference = GroupPCAReference(transform, args.reference_components,
                                      seed=args.seed)
        staged.merge(reference.build(corpus, masks, per_group=not args.no_groups))
    staged.merge(embeddings)
    embeddings = staged

    base = ScorerSettings(k=args.k, k_cross_donor=args.k_cross_donor,
                          permutations=args.permutations, seed=args.seed)
    probe = StageProbe(ProbeSettings.default(base, scib=args.scib))
    widths = {n: embeddings.width(n) for n in embeddings.names}
    tables_of = lambda r: StageTables(r, embeddings.kinds, widths)  # noqa: E731

    def write(report: BenchmarkReport) -> None:
        report.write_csv(out / "stages.csv")
        report.write_settings(out / "settings.json")
        tables_of(report).write_markdown(out / "stages.md",
                                         title=f"Stage probe: {args.run}")

    report = probe.score(embeddings, corpus, after_each=write)
    write(report)
    tables = tables_of(report)
    logger.info("\n%s\n\n%s\n%s", tables.pipeline_table(), tables.summary(),
                tables.group_table())
    logger.info("Written to %s", out)


if __name__ == "__main__":
    main()
