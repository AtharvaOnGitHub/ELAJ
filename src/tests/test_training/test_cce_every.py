"""CCE's second encoder pass runs on every k-th step, and only then."""

import pytest

torch = pytest.importorskip("torch")

from endo_model.configs.model_config import CCEConfig  # noqa: E402
from endo_model.configs.training_config import TrainingConfig  # noqa: E402
from endo_model.model.endo_model import EndoFoundationModel  # noqa: E402
from endo_model.model.objectives.composite import CompositeLoss  # noqa: E402
from endo_model.training.trainer import Trainer  # noqa: E402
from tests.test_training.test_trainer import (  # noqa: E402
    D_GAUSS, N_STUDIES, N_CIRC, _BatchList, make_batch, make_config, make_vocab,
)


def _run(cce_every: int, steps: int = 6):
    """Number of `encode` calls beyond the one every forward pass makes:
    `forward` goes through `encode` too, so the count is steps + CCE passes."""
    vocab = make_vocab()
    model = EndoFoundationModel(vocab, make_config())
    loss = CompositeLoss(n_studies=N_STUDIES, d_gauss=D_GAUSS, n_circular=N_CIRC,
                         w_kl_gauss=0.01, w_kl_circ=0.01, w_dab=0.01, w_cce=0.5)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss.parameters()),
                                 lr=1e-3)
    calls = []
    original = model.encode

    def spy(batch):
        calls.append(True)
        return original(batch)

    model.encode = spy
    batches = _BatchList([make_batch(vocab)] * steps)
    trainer = Trainer(model=model, train_loader=batches, val_loader=_BatchList([]),
                      composite_loss=loss, optimizer=optimizer, scheduler=None,
                      config=TrainingConfig(max_epochs=1, early_stop_patience=100),
                      cce_enabled=True, cce_every=cce_every)
    trainer.fit()
    return len(calls) - steps


def test_every_step_by_default():
    assert _run(1) == 6


def test_every_third_step():
    assert _run(3) == 2          # steps 0 and 3


def test_config_refuses_zero():
    with pytest.raises(ValueError):
        CCEConfig.from_dict({"every": 0})
    assert CCEConfig.from_dict({}).every == 1
    assert CCEConfig.from_dict({"every": 4}).every == 4
