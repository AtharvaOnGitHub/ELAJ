"""End to end: real model + real Dataset -> every stage -> scorer -> tables.

The unit tests prove the readers, the collector, the probe and the tables
one at a time. This proves they fit, on an untrained model over a synthetic
two-study corpus: the collected rows line up with the corpus built over the
sampler order, every stage gets a row, the `latent` stage is the same
matrix the benchmark would have scored, and the tables render.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("anndata")

from torch.utils.data import DataLoader  # noqa: E402

from endo_model.configs.model_config import (  # noqa: E402
    AggregationEncoderConfig, GroupEncoderConfig, ModelConfig,
)
from endo_model.data.collate import make_collate_fn  # noqa: E402
from endo_model.data.dataset import EndometriosisDataset  # noqa: E402
from endo_model.data.samplers import PerStudyBatchSampler  # noqa: E402
from endo_model.evaluation.baselines import (  # noqa: E402
    ScorerSettings, extract_corpus_matrix,
)
from endo_model.evaluation.baselines.scorer import (  # noqa: E402
    add_harmonised_labels, required_obs_columns,
)
from endo_model.evaluation.latent_validation import circular_coordinates  # noqa: E402
from endo_model.evaluation.stages import (  # noqa: E402
    GroupPCAReference, ProbeSettings, StageCollector, StageEmbeddings,
    StageProbe, StageTables, group_battery, group_masks_from_vocabulary,
    group_readers, pipeline_readers,
)
from endo_model.model.endo_model import EndoFoundationModel  # noqa: E402
from tests.test_evaluation.test_baselines_smoke import (  # noqa: E402, F401
    _sampler_order, labelled_two_study,
)


def _model(vocab):
    config = ModelConfig(
        d_token=8, d_agg=16, d_gauss=4, n_circular=2, d_dec=12,
        n_groups=len(vocab.group_names), n_studies=2,
        group_encoder=GroupEncoderConfig(n_layers=1, n_heads=2, dropout=0.0),
        aggregation_encoder=AggregationEncoderConfig(n_layers=1, n_heads=2,
                                                     dropout=0.0),
    )
    return EndoFoundationModel(vocab, config).eval()


class TestSmoke:
    def test_dataset_to_tables(self, tmp_path, labelled_two_study):
        vocab, configs = labelled_two_study
        names = {c["id"]: c["name"] for c in configs}
        gene_names = [vocab.index_to_gene(i) for i in range(vocab.vocab_size)]
        ds = EndometriosisDataset(configs, vocab, split="train", seed=24,
                                  extra_obs_columns=required_obs_columns())
        loader = DataLoader(
            ds, batch_sampler=PerStudyBatchSampler(ds.study_ids, batch_size=7,
                                                   shuffle=False, drop_last=True),
            collate_fn=make_collate_fn(vocab),
        )
        order = _sampler_order(ds)

        groups = list(vocab.group_names)
        readers = pipeline_readers(groups) + group_readers(groups)
        model = _model(vocab)
        collected = StageCollector(readers).collect(model, loader, torch.device("cpu"),
                                                    order_len=len(order))
        assert collected.n_cells == len(order)
        assert len(collected.names) == 4 + len(groups)
        assert collected.width("groups/concat") == 8 * len(groups)
        assert collected.width("latent") == 4 + 2 * 2

        # The latent stage is what validate_latent / the benchmark score.
        mu, angle = [], []
        with torch.no_grad():
            for batch in loader:
                out = model.encode(batch)
                mu.append(out["mu_gauss"].numpy())
                angle.append(out["mu_angle"].numpy())
        want = np.concatenate([np.concatenate(mu),
                               circular_coordinates(np.concatenate(angle))], -1)
        got = collected.matrices["latent"]
        # circular_coordinates' column layout is its own; compare as sets of
        # per-row values, which is what a Euclidean neighbour search sees.
        np.testing.assert_allclose(np.sort(got, axis=1), np.sort(want, axis=1),
                                   atol=1e-5)

        corpus = extract_corpus_matrix(ds, gene_names, names, indices=order)
        add_harmonised_labels(corpus)

        # The input references from the counts, ahead of the collected stages.
        masks = group_masks_from_vocabulary(vocab, corpus.n_genes)
        emb = StageEmbeddings()
        for transform in ("counts_raw", "counts_log"):
            emb.merge(GroupPCAReference(transform, n_components=8).build(corpus, masks))
        emb.merge(collected)
        assert emb.names[0] == "counts_raw/concat" or emb.names[0].startswith("counts_raw/")
        assert emb.width("counts_log/concat") == 8 * len(groups)
        assert len(emb.names) == 2 * (len(groups) + 1) + 4 + len(groups)

        base = ScorerSettings(k=5, k_cross_donor=10, permutations=5, scib_labels=())
        report = StageProbe(ProbeSettings(base, group_battery(base))).score(emb, corpus)
        assert report.methods() == emb.names

        tables = StageTables(report, emb.kinds, {n: emb.width(n) for n in emb.names})
        assert sorted(tables.groups()) == sorted(groups)
        path = tables.write_markdown(tmp_path / "stages.md")
        text = path.read_text()
        assert "| latent |" in text and f"| {groups[0]} |" in text
