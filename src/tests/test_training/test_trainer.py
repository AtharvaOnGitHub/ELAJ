"""Integration test: training loop runs one epoch without errors."""

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset

from endo_model.configs.model_config import (
    AggregationEncoderConfig,
    GroupEncoderConfig,
    ModelConfig,
)
from endo_model.configs.training_config import TrainingConfig
from endo_model.data.vocabulary import GeneVocabulary, generate_synthetic_vocabulary
from endo_model.model.endo_model import EndoFoundationModel
from endo_model.model.objectives.composite import CompositeLoss
from endo_model.training.callbacks import EarlyStopping, MetricLogger
from endo_model.training.checkpointing import CheckpointManager
from endo_model.training.trainer import Trainer

N_GENES = 30
N_GROUPS = 3
N_STUDIES = 2
B = 2
D_TOKEN = 8
D_AGG = 16
D_GAUSS = 8
N_VMF = 2


def make_vocab():
    raw = generate_synthetic_vocabulary(n_genes=N_GENES, n_groups=N_GROUPS, seed=24)
    return GeneVocabulary.from_dict(raw)


def make_config() -> ModelConfig:
    return ModelConfig(
        d_token=D_TOKEN, d_agg=D_AGG, d_gauss=D_GAUSS, n_vMF=N_VMF,
        d_dec=8, n_studies=N_STUDIES,
        group_encoder=GroupEncoderConfig(n_layers=1, n_heads=2, dropout=0.0),
        aggregation_encoder=AggregationEncoderConfig(n_layers=1, n_heads=2, dropout=0.0),
    )


def make_batch(vocab: GeneVocabulary, n: int = B):
    vocab_size = vocab.vocab_size
    group_names = vocab.group_names
    counts = torch.randint(1, 50, (n, vocab_size)).float()
    is_dns = torch.zeros(n, vocab_size, dtype=torch.bool)

    group_indices: dict[str, torch.Tensor] = {}
    group_dns_mask: dict[str, torch.Tensor] = {}
    for g in group_names:
        gi = torch.tensor(vocab.global_indices_for_group(g), dtype=torch.long)
        group_indices[g] = gi.unsqueeze(0).expand(n, -1)
        group_dns_mask[g] = torch.zeros(n, gi.shape[0], dtype=torch.bool)

    return {
        "counts": counts,
        "is_dns": is_dns,
        "library_size": counts.sum(dim=1),
        "study_id": torch.zeros(n, dtype=torch.long),
        "group_indices": group_indices,
        "group_dns_mask": group_dns_mask,
        "tissue_levels": torch.zeros(n, 4, dtype=torch.long),
        "age": torch.full((n,), 30.0),
        "disease_status": torch.zeros(n, dtype=torch.long),
    }


class _BatchList:
    """Fake DataLoader that yields a fixed list of batch dicts."""
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTrainer:
    def setup_method(self):
        self.vocab = make_vocab()
        self.config = make_config()
        self.model = EndoFoundationModel(self.vocab, self.config)
        self.comp_loss = CompositeLoss(
            n_studies=N_STUDIES, d_gauss=D_GAUSS, n_vMF=N_VMF,
            w_kl_gauss=0.01, w_kl_vmf=0.01, w_dab=0.01, w_cce=0.0,
        )
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.comp_loss.parameters()),
            lr=1e-3
        )
        batch = make_batch(self.vocab)
        self.train_loader = _BatchList([batch])
        self.val_loader = _BatchList([batch])
        self.train_config = TrainingConfig(max_epochs=2, early_stop_patience=100)

    def _make_trainer(self, **kwargs):
        return Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            composite_loss=self.comp_loss,
            optimizer=self.optimizer,
            scheduler=None,
            config=self.train_config,
            **kwargs,
        )

    def test_one_step_no_error(self):
        trainer = self._make_trainer()
        history = trainer.fit()
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2  # 2 epochs

    def test_val_loss_in_history(self):
        trainer = self._make_trainer()
        history = trainer.fit()
        assert "val_loss" in history
        assert len(history["val_loss"]) == 2

    def test_history_losses_finite(self):
        trainer = self._make_trainer()
        history = trainer.fit()
        for loss in history["train_loss"]:
            assert loss < float("inf") and loss > -float("inf")


class TestEarlyStopping:
    def test_no_stop_with_improving_loss(self):
        es = EarlyStopping(patience=3)
        for val in [1.0, 0.9, 0.8, 0.7]:
            assert not es.step(val)

    def test_stop_after_patience_exceeded(self):
        es = EarlyStopping(patience=2)
        es.step(1.0)  # sets best
        es.step(1.1)  # no improvement, count=1
        stopped = es.step(1.2)  # count=2 >= patience=2
        assert stopped

    def test_reset_clears_state(self):
        es = EarlyStopping(patience=2)
        es.step(1.0)
        es.step(2.0)
        es.step(3.0)
        es.reset()
        assert not es.should_stop
        assert es._counter == 0


class TestMetricLogger:
    def test_log_and_average(self):
        logger = MetricLogger()
        logger.log({"loss": 1.0, "kl": 0.5})
        logger.log({"loss": 3.0, "kl": 0.5})
        avgs = logger.epoch_averages()
        assert abs(avgs["loss"] - 2.0) < 1e-6
        assert abs(avgs["kl"] - 0.5) < 1e-6

    def test_reset_clears_data(self):
        logger = MetricLogger()
        logger.log({"loss": 1.0})
        logger.reset()
        avgs = logger.epoch_averages()
        assert "loss" not in avgs


class TestCheckpointManager:
    def test_save_and_load(self, tmp_path):
        vocab = make_vocab()
        config = make_config()
        model = EndoFoundationModel(vocab, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ckpt = CheckpointManager(str(tmp_path), max_to_keep=2)
        path = ckpt.save(model, optimizer, epoch=0, step=10, val_loss=1.5)
        assert path.exists()

        # Load into a fresh model
        model2 = EndoFoundationModel(vocab, config)
        payload = CheckpointManager.load(str(path), model2, optimizer)
        assert payload["epoch"] == 0
        assert payload["val_loss"] == 1.5

    def test_max_to_keep_prunes(self, tmp_path):
        vocab = make_vocab()
        config = make_config()
        model = EndoFoundationModel(vocab, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ckpt = CheckpointManager(str(tmp_path), max_to_keep=2)
        for i, loss in enumerate([1.0, 0.8, 0.6, 0.4]):
            ckpt.save(model, optimizer, epoch=i, step=i * 10, val_loss=loss)

        # Only 2 files should remain (the best 2: 0.4 and 0.6)
        saved = list(tmp_path.glob("*.pt"))
        assert len(saved) == 2
