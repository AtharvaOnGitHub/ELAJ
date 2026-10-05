"""The label probe scores the validation latent on family, every epoch, and
can be the number that ranks checkpoints.

Planted data first: a latent that encodes family scores near 1, one that does
not scores near 0, unlabelled cells are ignored, and a row count that does not
match the labels is refused. Then the trainer: `val_family_knn` lands in the
history and, with `stop_metric: family_knn`, the checkpoint ranking follows it
negated.

The planted half needs no torch of its own, but `endo_model.training` imports
the checkpoint manager, which does, so the skip is module-wide rather than on
the trainer tests alone.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from endo_model.training.label_probe import FAMILY_METRIC, LabelProbe  # noqa: E402


def _labels(n, unlabelled_every=5):
    fam = np.array(["epithelial" if i % 2 else "stromal" for i in range(n)], dtype=object)
    fam[::unlabelled_every] = None
    return fam


def _latent_from(fam, noise, seed=0):
    rng = np.random.default_rng(seed)
    centre = np.where(fam == "epithelial", 1.0, -1.0).astype(float)[:, None]
    mu = centre * np.ones((1, 4)) + rng.normal(0, noise, (len(fam), 4))
    angle = rng.uniform(-np.pi, np.pi, (len(fam), 2))
    return mu.astype(np.float32), angle.astype(np.float32)


class TestLabelProbe:
    def test_family_latent_scores_high_and_noise_scores_low(self):
        fam = _labels(400)
        probe = LabelProbe(fam, k=10)
        assert probe.n_labelled == 320
        mu, angle = _latent_from(fam, noise=0.05)
        good = probe.score(mu, angle)
        assert good[FAMILY_METRIC] > 0.9 and good[f"{FAMILY_METRIC}_n"] == 320
        rng = np.random.default_rng(1)
        bad = probe.score(rng.normal(size=mu.shape).astype(np.float32), angle)
        assert abs(bad[FAMILY_METRIC]) < 0.2

    def test_row_mismatch_is_refused(self):
        probe = LabelProbe(_labels(50), k=5)
        with pytest.raises(ValueError, match="sampler order"):
            probe.score(np.zeros((49, 4), np.float32), np.zeros((49, 2), np.float32))

    def test_too_few_labelled_cells_is_nan_not_an_error(self):
        fam = np.full(20, None, dtype=object)
        fam[:3] = "stromal"
        out = LabelProbe(fam, k=5).score(np.zeros((20, 4), np.float32),
                                         np.zeros((20, 2), np.float32))
        assert np.isnan(out[FAMILY_METRIC]) and out[f"{FAMILY_METRIC}_n"] == 3

    def test_max_cells_subsamples_the_labelled_rows(self):
        fam = _labels(400)
        mu, angle = _latent_from(fam, noise=0.05)
        out = LabelProbe(fam, k=10, max_cells=100).score(mu, angle)
        assert out[f"{FAMILY_METRIC}_n"] == 100


class TestFamilyLabelsInOrder:
    def test_reads_the_harmonised_family_in_sampler_order(self, tmp_path):
        pytest.importorskip("anndata")
        import anndata
        import pandas as pd
        import scipy.sparse

        from endo_model.data.dataset import EndometriosisDataset
        from endo_model.data.vocabulary import GeneVocabulary, generate_synthetic_vocabulary
        from endo_model.training.label_probe import family_labels_in_order

        raw = generate_synthetic_vocabulary(n_genes=20, n_groups=2, seed=1)
        vocab = GeneVocabulary.from_dict(raw)
        n = 12
        obs = pd.DataFrame({
            "patient_id": [f"P{i % 3}" for i in range(n)],
            "disease_status": ["eutopic"] * n,
            "AnnotationMain": pd.Categorical(
                ["Epithelial" if i % 2 else "Stromal" for i in range(n)]),
        }, index=[f"c{i}" for i in range(n)])
        a = anndata.AnnData(X=scipy.sparse.csr_matrix(np.ones((n, 20), np.float32)), obs=obs)
        a.var_names = list(raw)
        path = tmp_path / "s.h5ad"
        a.write_h5ad(path)
        from endo_model.evaluation.label_harmonisation import LINEAGE_SOURCE_COLUMNS
        ds = EndometriosisDataset([{"id": 0, "name": "s", "path": str(path)}], vocab,
                                  split="train", seed=24, val_fraction=0.0,
                                  test_fraction=0.0,
                                  extra_obs_columns=list(LINEAGE_SOURCE_COLUMNS))
        order = list(range(len(ds)))[::-1]
        labels = family_labels_in_order(ds, order)
        assert len(labels) == len(ds)
        raw_labels = ds.extra_obs["AnnotationMain"]
        for pos, idx in enumerate(order):
            expected = "epithelial" if str(raw_labels[idx]).lower().startswith("epi") else "stromal"
            assert labels[pos] == expected


class TestTrainerIntegration:
    def test_probe_number_reaches_history_and_ranks_checkpoints(self, tmp_path):
        torch = pytest.importorskip("torch")
        from endo_model.configs.training_config import TrainingConfig
        from endo_model.model.endo_model import EndoFoundationModel
        from endo_model.model.objectives.composite import CompositeLoss
        from endo_model.training.trainer import Trainer
        from tests.test_training.test_trainer import (
            B, D_GAUSS, N_CIRC, N_STUDIES, _BatchList, make_batch, make_config, make_vocab,
        )

        vocab = make_vocab()
        model = EndoFoundationModel(vocab, make_config())
        loss = CompositeLoss(n_studies=N_STUDIES, d_gauss=D_GAUSS, n_circular=N_CIRC,
                             w_kl_gauss=0.01, w_kl_circ=0.01, w_dab=0.01, w_cce=0.0)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(loss.parameters()),
                                     lr=1e-3)
        n_val_batches = 6
        batches = _BatchList([make_batch(vocab) for _ in range(n_val_batches)])
        fam = np.array(["epithelial" if i % 2 else "stromal"
                        for i in range(B * n_val_batches)], dtype=object)
        probe = LabelProbe(fam, k=2)
        trainer = Trainer(model=model, train_loader=batches, val_loader=batches,
                          composite_loss=loss, optimizer=optimizer, scheduler=None,
                          config=TrainingConfig(max_epochs=2, early_stop_patience=100,
                                                stop_metric=FAMILY_METRIC),
                          checkpoint_dir=str(tmp_path), label_probe=probe)
        history = trainer.fit()
        assert f"val_{FAMILY_METRIC}" in history
        assert len(history[f"val_{FAMILY_METRIC}"]) == 2
        # The checkpoint's recorded val_loss is the NEGATED probe number.
        payload = torch.load(sorted(tmp_path.glob("epoch_*.pt"))[-1],
                             map_location="cpu", weights_only=False)
        assert payload["val_loss"] == pytest.approx(-history[f"val_{FAMILY_METRIC}"][-1])

    def test_nan_probe_falls_back_to_nb(self):
        from endo_model.configs.training_config import TrainingConfig
        from endo_model.training.trainer import Trainer

        from endo_model.training.checkpoint_ranking import CheckpointRanking

        # The stub needs `ranking` as well as `config`: `_stop_metric` was
        # inlined in the Trainer when this test was written and now delegates
        # to CheckpointRanking, which is built in __init__ and logged there so
        # the record says what the run selected for. A stub carrying only
        # `config` raised AttributeError rather than testing the fallback.
        class _T(Trainer):
            def __init__(self, metric):
                self.config = TrainingConfig(stop_metric=metric)
                self.ranking = CheckpointRanking(metric=metric)

        t = _T(FAMILY_METRIC)
        assert t._stop_metric({"nb": 0.4, FAMILY_METRIC: float("nan")}, 1.0) == 0.4
        assert t._stop_metric({"nb": 0.4, FAMILY_METRIC: 0.8}, 1.0) == -0.8
        t = _T("nb")
        assert t._stop_metric({"nb": 0.4, FAMILY_METRIC: 0.8}, 1.0) == 0.4
