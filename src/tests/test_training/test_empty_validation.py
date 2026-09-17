"""An epoch whose validation loader yields nothing must not end the run.

Findings A3: `_val_epoch` on no batches returns only timing, and the
trainer read `val_metrics["total"]` from it. The split is by donor, so a
study, or at a bad seed a whole split, can be empty. The epoch is now
skipped for scoring and the run continues, which is what a warning is for.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.configs.training_config import TrainingConfig  # noqa: E402
from endo_model.model.endo_model import EndoFoundationModel  # noqa: E402
from endo_model.model.objectives.composite import CompositeLoss  # noqa: E402
from endo_model.training.trainer import Trainer  # noqa: E402
from tests.test_training.test_trainer import (  # noqa: E402
    D_GAUSS, N_STUDIES, N_CIRC, _BatchList, make_batch, make_config, make_vocab,
)


def test_empty_validation_loader_warns_and_continues(tmp_path, caplog):
    vocab = make_vocab()
    model = EndoFoundationModel(vocab, make_config())
    loss = CompositeLoss(n_studies=N_STUDIES, d_gauss=D_GAUSS, n_circular=N_CIRC,
                         w_kl_gauss=0.01, w_kl_circ=0.01, w_dab=0.01, w_cce=0.0)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss.parameters()),
                                 lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    trainer = Trainer(model=model, train_loader=_BatchList([make_batch(vocab)]),
                      val_loader=_BatchList([]), composite_loss=loss,
                      optimizer=optimizer, scheduler=scheduler,
                      config=TrainingConfig(max_epochs=2, early_stop_patience=100),
                      checkpoint_dir=str(tmp_path))
    history = trainer.fit()
    assert len(history["train_loss"]) == 2
    assert history["val_loss"] == []
    assert "validation produced no batches" in caplog.text
    assert not list(tmp_path.glob("epoch_*.pt"))
    assert scheduler.last_epoch == 2
