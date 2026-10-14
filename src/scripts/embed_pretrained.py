"""Embed our cells with a pretrained foundation model, in that model's own venv.

    python -m scripts.embed_pretrained --run m6_drain --model geneformer \
        --repo /rds/projects/.../models/Geneformer \
        --vocabulary /rds/projects/.../data/processed/vocabulary_wide.json \
        --work /tmp/geneformer_work \
        --out outputs/benchmarks/latents/geneformer

WHY THIS IS A SEPARATE SCRIPT. Geneformer, scGPT and scBERT cannot run inside
benchmark_baselines.py. Each needs its own virtual environment, and the reason
they are kept apart is that their dependency resolution must not move
scvi-tools' pins underneath a live campaign. So this runs in the model's
environment, writes an ExternalLatent, and the scorer reads it:

    sbatch ... benchmark_baselines.sh m6_drain --no-elaj \
        --external-latent outputs/benchmarks/latents/geneformer ...

WHICH CELLS. `build_corpus`, imported from benchmark_baselines rather than
reimplemented, so there is one definition of which cells the table is over.
The sampler order fixes it, with or without ELAJ embedded.

WHICH GENES. `--vocabulary` defaults to the data config's, which is the
15,496-gene pathway-annotated half of 30,962 measured genes, filtered to suit
ELAJ's group encoder. PASS THE WIDE ONE. Handing a competitor our curation is
the mistake we found in the scVI comparison on 12 October 2026, and a
pretrained model reads its own vocabulary anyway, so the only question is how
much of its vocabulary we offer. Coverage is recorded in the manifest either
way.

NOTHING IS FINE-TUNED. Zero-shot, so `fit_on` is `pretrained`: these models
never saw our training split either, and filing them with the inductive
baselines would say something false about what they were fitted on.
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
logger = logging.getLogger("elaj.embed_pretrained")

MODELS = ("geneformer", "scgpt", "scimilarity", "scfoundation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="Experiment name whose data config defines the corpus "
                        "and splits. Only the DATA half is read; no checkpoint "
                        "is loaded and no model of ours is built.")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--repo", required=True,
                   help="The staged model directory, e.g. .../models/Geneformer")
    p.add_argument("--root", default=None)
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--vocabulary", default=None,
                   help="A vocabulary.json to use as the gene axis instead of "
                        "the data config's. Pass the WIDE one: see above.")
    p.add_argument("--work", default=None,
                   help="Scratch for the h5ad, the tokenised dataset and the "
                        "embedding csv. Defaults to <out>/work. Kept, because "
                        "when a run goes wrong those three files are the "
                        "evidence.")
    p.add_argument("--emb-layer", type=int, default=-1,
                   help="-1 is the second-to-last layer, Geneformer's own "
                        "convention. 0 is the last.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--max-cells", type=int, default=None,
                   help="Smoke test: embed only this many cells, balanced "
                        "across studies. The full path runs in minutes "
                        "instead of hours, on the same code.")
    p.add_argument("--out", required=True,
                   help="Directory for latent.npy and manifest.json.")
    return p.parse_args()


def build_embedder(args, work: Path):
    """One model name -> one embedder. The registry, such as it is."""
    if args.model == "geneformer":
        from endo_model.evaluation.baselines.pretrained import GeneformerEmbedder

        return GeneformerEmbedder(
            repo=args.repo, work=work, emb_layer=args.emb_layer,
            batch_size=args.batch_size, nproc=args.nproc,
        )
    if args.model == "scgpt":
        from endo_model.evaluation.baselines.pretrained import ScGPTEmbedder

        return ScGPTEmbedder(
            repo=args.repo, work=work, batch_size=args.batch_size,
        )
    if args.model == "scimilarity":
        from endo_model.evaluation.baselines.pretrained import (
            ScimilarityEmbedder,
        )

        # buffer_size, not the forward batch: SCimilarity embeds in buffers of
        # 10,000 by default and --batch-size here is in units of cells either
        # way, so the wrapper's 64 would be needlessly slow.
        return ScimilarityEmbedder(repo=args.repo, work=work)
    if args.model == "scfoundation":
        from endo_model.evaluation.baselines.pretrained import (
            ScFoundationEmbedder,
        )

        return ScFoundationEmbedder(repo=args.repo, work=work)
    raise ValueError(f"unknown model {args.model!r}; known: {list(MODELS)}")


def main() -> None:
    args = parse_args()
    root = args.root or str(Path.cwd())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else out / "work"

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.baselines.external_latent import (
        ExternalLatent, corpus_fingerprint,
    )
    from endo_model.evaluation.baselines.scorer import required_obs_columns
    from scripts.benchmark_baselines import build_corpus

    config = Path(root) / args.config_dir / f"{args.run}.yaml"
    if not config.exists():
        logger.error("no config at %s", config)
        sys.exit(1)
    exp_cfg, _model_cfg, train_cfg, data_cfg = load_experiment(str(config),
                                                               root=root)

    vocabulary_path = (Path(args.vocabulary) if args.vocabulary
                       else Path(root) / data_cfg.vocabulary_path)
    if not args.vocabulary:
        logger.warning(
            "No --vocabulary given, so using the data config's %d-gene axis. "
            "That is the pathway-annotated half of the measured transcriptome, "
            "filtered for ELAJ's group encoder, and handing it to a pretrained "
            "model gives it our curation. Pass the wide vocabulary.",
            GeneVocabulary(str(vocabulary_path)).vocab_size,
        )
    vocab = GeneVocabulary(str(vocabulary_path))
    gene_names = [vocab.index_to_gene(i) for i in range(vocab.vocab_size)]
    logger.info("Gene axis: %d genes from %s", vocab.vocab_size, vocabulary_path)

    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
        for s in data_cfg.resolve_studies(root)
    ]
    names = {s["id"]: s["name"] for s in study_configs}

    # No `orders`: nothing is embedded by us here, so build_corpus takes the
    # sampler's order itself. That is the same order the benchmark uses, which
    # is what makes these rows joinable to that table.
    corpus = build_corpus(root, exp_cfg, train_cfg, data_cfg, vocab,
                          study_configs, args.splits, {},
                          required_obs_columns(), gene_names, names)
    logger.info("Corpus: %d cells x %d genes, %d studies, splits %s",
                corpus.n_cells, corpus.n_genes, corpus.n_studies, corpus.split)

    if args.max_cells and args.max_cells < corpus.n_cells:
        from endo_model.evaluation.study_mixing import balanced_rows

        # Balanced across studies, not the first N: tokenisation is per study
        # and a study with unusual gene names is exactly what a smoke test
        # should hit.
        corpus = corpus.subset_cells(
            balanced_rows(corpus.study_id, args.max_cells, 24))
        logger.warning("SMOKE TEST: %d cells across %d studies. This is NOT a "
                       "reportable embedding.", corpus.n_cells, corpus.n_studies)

    embedder = build_embedder(args, work)
    latent, row_index, settings = embedder.embed(corpus)
    settings["gene_vocabulary"] = str(vocabulary_path)
    settings["splits"] = "+".join(args.splits)

    external = ExternalLatent(
        method=embedder.name, latent=latent, row_index=row_index,
        fingerprint=corpus_fingerprint(corpus), settings=settings)
    external.write(out)
    logger.info("Score it with: --external-latent %s", out)


if __name__ == "__main__":
    main()
