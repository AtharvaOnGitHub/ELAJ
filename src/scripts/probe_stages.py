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

COST, MEASURED. The patient-identity search is brute force, so a stage costs
the square of the cell count. On r8's 238k cells a 1,408-wide stage took
over an hour; on the v-series 382,464 it took four hours twenty-six minutes
(job 55708680, 1 Oct 2026). Three wide stages at that rate do not fit a
twelve-hour wall, and the two that are references come first, so a job that
runs out of time returns the references and nothing about the model.
`--no-references` drops those two.
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
    p.add_argument("--only-groups", action="store_true",
                   help="Score ONLY the 44 per-group stages, skipping every "
                        "pipeline stage. The question 'which pathway groups "
                        "carry which signal' needs the per-group stages and "
                        "not the 12,672-wide concat of them, and that one "
                        "stage consumed a whole sixteen-hour wall on 416,256 "
                        "cells. Mutually exclusive with --no-groups.")
    p.add_argument("--no-groups", action="store_true",
                   help="Score the five pipeline stages only.")
    p.add_argument("--scib", action="store_true",
                   help="Run the scIB battery on the pipeline stages too. "
                        "Needs memory for a full distance matrix at the "
                        "widest stage.")
    p.add_argument("--no-scib", action="store_true",
                   help="Accepted for compatibility; scIB is already off.")
    p.add_argument("--untrained", action="store_true",
                   help="Probe the model at random initialisation instead of "
                        "the checkpoint. Same architecture, no training: says "
                        "whether the readout or the training put the group "
                        "summaries where they are.")
    p.add_argument("--no-linear", action="store_true",
                   help="Skip the linear probe on the pipeline stages.")
    p.add_argument("--no-references", action="store_true",
                   help="Score the model's own stages only, without the two "
                        "per-group PCA references. They are 1,408 wide, which "
                        "is two thirds of the job's whole cost, and they say "
                        "what a linear reading of the counts scores, which the "
                        "baseline table already says. Drop them when the "
                        "question is where along the encoder the signal goes.")
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


def build_readers(group_names: list[str], groups: bool, only_groups: bool = False):
    """The stages to score.

    ONLY_GROUPS EXISTS BECAUSE OF ONE STAGE. `groups/concat` is the 44 group
    readouts side by side, and with the cls+pool8 readout that is 44 x 288 =
    12,672 wide. The cross-donor search is brute force, so its cost is the
    cell count squared times the width, and on 416,256 cells that single
    stage took the whole sixteen-hour wall on job 61726249 without ever
    reaching the per-group stages the question actually needs. The 44
    individual stages are 288 wide each and get the shorter two-label
    battery, so together they cost a fraction of the one concat stage.
    """
    from endo_model.evaluation.stages import group_readers, pipeline_readers

    if only_groups:
        return group_readers(group_names)
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


def build_references(corpus, vocab, components: int, seed: int, per_group: bool):
    """The two input references: per-group PCA of the counts, raw and log.

    Raw is what ELAJ reads, log-normalised is what every other method reads.
    Together they say what a linear reading of the same genes scores, so a
    model stage below them lost something the counts still held.

    They cost what the widest model stage costs, twice, because they are the
    same width. `--no-references` is the caller's way of saying it already
    knows these numbers from the baseline table.
    """
    from endo_model.evaluation.stages import StageEmbeddings
    from endo_model.evaluation.stages.references import (
        LOG, RAW, GroupPCAReference, group_masks_from_vocabulary,
    )

    masks = group_masks_from_vocabulary(vocab, corpus.n_genes)
    staged = StageEmbeddings()
    for transform in (RAW, LOG):
        reference = GroupPCAReference(transform, components, seed=seed)
        staged.merge(reference.build(corpus, masks, per_group=per_group))
    return staged


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(exp_cfg.seed)
    model = EndoFoundationModel(vocab, model_cfg).to(device)
    if args.untrained:
        logger.info("Untrained model: random initialisation at seed %d, no "
                    "checkpoint loaded", exp_cfg.seed)
    else:
        run_dir = resolve_run_dir(root, exp_cfg.output_dir, args.runs_dir)
        if not run_dir.is_dir():
            logger.error("no run directory at %s", run_dir)
            sys.exit(1)
        ckpt = best_checkpoint(run_dir)
        CheckpointManager.load(str(ckpt), model, None, map_location=device)
        logger.info("Checkpoint: %s", ckpt)

    if args.only_groups and args.no_groups:
        logger.error("--only-groups and --no-groups cancel out; pick one")
        return 2
    readers = build_readers(list(vocab.group_names), groups=not args.no_groups,
                            only_groups=args.only_groups)
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
    #
    # Scoring runs in this order, so the references are also the two stages
    # that stand between the job starting and the first row about the model.
    # On 382,464 cells one of them takes four and a half hours.
    staged = StageEmbeddings()
    if args.no_references:
        logger.info("References skipped; the model's own stages only.")
    else:
        staged.merge(build_references(corpus, vocab, args.reference_components,
                                      args.seed, per_group=not args.no_groups))
    staged.merge(embeddings)
    embeddings = staged

    base = ScorerSettings(k=args.k, k_cross_donor=args.k_cross_donor,
                          permutations=args.permutations, seed=args.seed)
    probe = StageProbe(ProbeSettings.default(base, scib=args.scib,
                                             linear=not args.no_linear))
    widths = {n: embeddings.width(n) for n in embeddings.names}
    tables_of = lambda r: StageTables(r, embeddings.kinds, widths)  # noqa: E731

    def write(report: BenchmarkReport) -> None:
        report.write_csv(out / "stages.csv")
        report.write_settings(out / "settings.json")
        tables_of(report).write_markdown(
            out / "stages.md",
            title=f"Stage probe: {args.run}" + (" (untrained)" if args.untrained else ""))

    report = probe.score(embeddings, corpus, after_each=write)
    write(report)
    tables = tables_of(report)
    logger.info("\n%s\n\n%s\n%s", tables.pipeline_table(), tables.summary(),
                tables.group_table())
    logger.info("Written to %s", out)


if __name__ == "__main__":
    main()
