"""End to end: real Dataset -> CorpusMatrix -> a method -> scorer -> report.

The unit tests prove each piece against a hand-built CorpusMatrix. This one
proves the pieces fit: that a matrix pulled from the actual dataset class,
over the actual sampler order with its dropped tails, carries labels that
land on the right rows all the way to a written table. It is the run the
HPC job performs, on twenty cells instead of two hundred thousand.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
anndata = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from endo_model.data.collate import make_collate_fn
from endo_model.data.dataset import EndometriosisDataset
from endo_model.data.samplers import PerStudyBatchSampler
from endo_model.data.vocabulary import GeneVocabulary, generate_synthetic_vocabulary
from endo_model.evaluation.baselines import (
    AllGenes, BenchmarkReport, CorpusMatrix, EmbeddingScorer, IntersectionGenes,
    PCABaseline, ScorerSettings, extract_corpus_matrix,
)
from endo_model.evaluation.baselines.scorer import (
    HARMONISED_LINEAGE, add_harmonised_labels, required_obs_columns,
)
from tests.test_data.test_dataset import make_synthetic_h5ad


@pytest.fixture()
def labelled_two_study(tmp_path):
    """Two studies whose h5ads carry the label columns the scorer wants."""
    import pandas as pd

    raw = generate_synthetic_vocabulary(n_genes=80, n_groups=4, seed=24)
    vocab = GeneVocabulary.from_dict(raw)
    genes = list(raw.keys())
    paths = []
    for sid, (n_cells, n_genes, col) in enumerate(
            ((48, 80, "AnnotationMain"), (40, 50, "lineage"))):
        path = make_synthetic_h5ad(tmp_path, f"s{sid}.h5ad", genes[:n_genes],
                                   n_cells=n_cells, n_patients=4, study_id=sid,
                                   seed=sid)
        # Plant a cell-family label in the column that study would carry.
        a = anndata.read_h5ad(path)
        a.obs[col] = pd.Categorical(
            ["epithelial" if i % 2 else "stromal" for i in range(n_cells)])
        a.write_h5ad(path)
        paths.append(path)
    configs = [{"id": i, "name": f"study{i}", "path": p}
               for i, p in enumerate(paths)]
    return vocab, configs


def _sampler_order(dataset, batch_size=7):
    """The sampler's order with drop_last, as the benchmark script reads it.

    Seven, not eight: the fixture holds 48 and 40 cells, both multiples of
    eight, and a batch size that divides every study leaves no tail to drop
    and nothing for the alignment check below to test. Seven leaves 6 and 5.
    """
    sampler = PerStudyBatchSampler(dataset.study_ids, batch_size=batch_size,
                                   shuffle=False, drop_last=True)
    return [i for batch in sampler for i in batch]


class TestSmoke:
    def test_dataset_to_report(self, tmp_path, labelled_two_study):
        vocab, configs = labelled_two_study
        names = {c["id"]: c["name"] for c in configs}
        gene_names = [vocab.index_to_gene(i) for i in range(vocab.vocab_size)]

        ds = EndometriosisDataset(configs, vocab, split="train", seed=24,
                                  extra_obs_columns=required_obs_columns())
        order = _sampler_order(ds)
        assert 0 < len(order) < len(ds), "drop_last should have removed a tail"

        corpus = extract_corpus_matrix(ds, gene_names, names, indices=order)
        assert corpus.n_cells == len(order)
        assert np.array_equal(corpus.source_index, np.asarray(order))
        add_harmonised_labels(corpus)
        labelled = sum(v is not None for v in corpus.meta[HARMONISED_LINEAGE])
        assert labelled == corpus.n_cells, "every cell was given a family"

        # Labels landed on the right rows. The fixture planted the family in
        # a different source column per study; after harmonisation every row
        # must hold one of the two families, and each study's rows must carry
        # exactly what its own h5ad said.
        for row, idx in enumerate(order):
            got = corpus.meta[HARMONISED_LINEAGE][row]
            assert got in ("epithelial", "stromal")
            src = "AnnotationMain" if ds[idx]["study_id"] == 0 else "lineage"
            assert corpus.meta[src][row] == got

        report = BenchmarkReport()
        scorer = EmbeddingScorer(ScorerSettings(k=5, k_cross_donor=10,
                                                permutations=5))
        for policy in (AllGenes(), IntersectionGenes()):
            restricted = policy.apply(corpus)
            emb = PCABaseline(n_components=4).embed(restricted)
            report.add(scorer.score(emb, restricted, gene_policy=policy.name))
            report.record_settings("pca", policy.name, emb.settings)

        assert report.methods() == ["pca/all", "pca/intersection"]
        csv_path = report.write_csv(tmp_path / "b.csv")
        md_path = report.write_markdown(tmp_path / "b.md")
        assert csv_path.exists() and md_path.exists()
        table = report.comparison_table()
        assert "pca/all" in table and "pca/intersection" in table
        # The intersection policy must have removed study 1's missing genes.
        assert IntersectionGenes().apply(corpus).n_genes == 50
