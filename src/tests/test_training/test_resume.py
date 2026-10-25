"""A resumed run continues the run it was; it does not start a new one.

Findings A2: the first resume restored the epoch and nothing else, so the
KL warm-up and the DAB ramp (functions of `global_step`) restarted from
zero, the cosine LR went back to its peak, early stopping forgot its best,
and metrics.csv began again empty. Each of those is now saved in every
checkpoint and restored by `Trainer.restore`, and each is asserted here by
training a run in two halves and comparing it with the same run in one
piece.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.configs.training_config import TrainingConfig  # noqa: E402
from endo_model.model.endo_model import EndoFoundationModel  # noqa: E402
from endo_model.model.objectives.composite import CompositeLoss  # noqa: E402
from endo_model.training.callbacks import EarlyStopping  # noqa: E402
from endo_model.training.checkpointing import CheckpointManager  # noqa: E402
from endo_model.training.trainer import Trainer  # noqa: E402
from tests.test_training.test_trainer import (  # noqa: E402
    D_GAUSS, N_STUDIES, N_CIRC, _BatchList, make_batch, make_config, make_vocab,
)


def _setup(seed: int, max_epochs: int, ckpt_dir):
    torch.manual_seed(seed)
    vocab = make_vocab()
    model = EndoFoundationModel(vocab, make_config())
    loss = CompositeLoss(n_studies=N_STUDIES, d_gauss=D_GAUSS, n_circular=N_CIRC,
                         w_kl_gauss=0.01, w_kl_circ=0.01, w_dab=0.01, w_cce=0.0)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss.parameters()),
                                 lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    torch.manual_seed(seed)
    batches = _BatchList([make_batch(vocab)])
    config = TrainingConfig(max_epochs=max_epochs, early_stop_patience=100)
    trainer = Trainer(model=model, train_loader=batches, val_loader=batches,
                      composite_loss=loss, optimizer=optimizer, scheduler=scheduler,
                      config=config, checkpoint_dir=str(ckpt_dir))
    return trainer, model, optimizer


class TestEarlyStoppingState:
    def test_round_trip(self):
        es = EarlyStopping(patience=3, warmup_epochs=1)
        for v in (1.0, 0.5, 0.7):
            es.step(v)
        state = es.state_dict()
        again = EarlyStopping(patience=3, warmup_epochs=1)
        again.load_state_dict(state)
        assert again.state_dict() == state
        assert again._best == 0.5 and again._counter == 1 and again._seen == 3


class TestCheckpointManagerRehydrate:
    def test_adopts_existing_checkpoints(self, tmp_path):
        """A fresh manager must adopt every checkpoint ON DISK and rank them.

        THE COUNT IS 3, AND IT HAS NOW BEEN WRONG IN BOTH DIRECTIONS. It
        asserted 3 when retention was top-k, was corrected to 2 on the
        reasoning that `scheduled` excludes epoch 0, and `RetentionPolicy`
        has since been changed to KEEP epoch 0 deliberately -- "as a
        diagnostic and not as a candidate", with the correlation figures in
        retention.py explaining why an undertrained epoch 0 must be visible
        but never selectable. So of epochs 0, 1 and 2: epoch 0 survives as
        the diagnostic, epoch 1 as the best, epoch 2 as the last.

        The code is right and the test was stale. Counted off the filesystem
        rather than hard-coded, so the assertion names what retention
        actually did rather than what someone expected it to do.
        """
        trainer, model, optimizer = _setup(0, 3, tmp_path)
        for epoch, val in ((0, 0.9), (1, 0.4), (2, 0.6)):
            trainer.ckpt_manager.save(model, optimizer, epoch, epoch * 10, val)
        on_disk = sorted(p.name for p in tmp_path.glob("epoch_*.pt"))
        assert len(on_disk) == 3, on_disk
        assert on_disk[0].startswith("epoch_0000_"), on_disk
        fresh = CheckpointManager(str(tmp_path))
        # Epoch 0 is kept but must never be chosen: epoch 1 has the best
        # metric and it is epoch 1 the manager has to rank first.
        assert fresh.best_path.name.startswith("epoch_0001_")
        assert len(fresh._history) == len(on_disk)

    def test_prunes_across_a_restart(self, tmp_path):
        trainer, model, optimizer = _setup(0, 3, tmp_path)
        for epoch, val in ((0, 0.9), (1, 0.4), (2, 0.6)):
            trainer.ckpt_manager.save(model, optimizer, epoch, epoch * 10, val)
        fresh = CheckpointManager(str(tmp_path), max_to_keep=3)
        fresh.save(model, optimizer, 3, 30, 0.5)
        kept = sorted(p.name[:10] for p in tmp_path.glob("epoch_*.pt"))
        # The worst of the four (0.9 at epoch 0) is gone, not the oldest new one.
        assert kept == ["epoch_0001", "epoch_0002", "epoch_0003"]


class TestResume:
    def test_split_run_matches_single_run(self, tmp_path):
        """Four epochs in one go against two, a checkpoint, then two more."""
        whole, _, _ = _setup(7, 4, tmp_path / "whole")
        history_whole = whole.fit()

        first, model, optimizer = _setup(7, 4, tmp_path / "split")
        first.config.max_epochs = 2
        first.fit()
        last = sorted((tmp_path / "split").glob("epoch_*.pt"))[-1]

        second, model2, optimizer2 = _setup(7, 4, tmp_path / "split")
        payload = CheckpointManager.load(str(last), model2, optimizer2)
        start = second.restore(payload)
        assert start == 2
        assert second.global_step == first.global_step
        assert second.early_stopping.state_dict() == first.early_stopping.state_dict()
        assert second.scheduler.last_epoch == first.scheduler.last_epoch
        history_split = second.fit()

        assert len(history_split["train_loss"]) == 4
        assert history_split["train_loss"][:2] == pytest.approx(
            history_whole["train_loss"][:2])
        assert second.global_step == whole.global_step
        assert second.scheduler.get_last_lr() == pytest.approx(whole.scheduler.get_last_lr())

    def test_restore_without_state_warns_and_continues(self, tmp_path, caplog):
        trainer, _, _ = _setup(1, 3, tmp_path)
        start = trainer.restore({"epoch": 4})
        assert start == 5 and trainer.global_step == 0
        assert "not restored" in caplog.text


class TestResolveResume:
    def test_latest_picks_the_newest_epoch(self, tmp_path):
        from scripts.train import resolve_resume

        ck = tmp_path / "run" / "checkpoints"
        ck.mkdir(parents=True)
        for name in ("epoch_0003_step_0000030.pt", "epoch_0011_step_0000110.pt",
                     "epoch_0007_step_0000070.pt"):
            (ck / name).write_bytes(b"")
        assert resolve_resume(str(tmp_path), tmp_path / "run", "latest").name.startswith("epoch_0011")
        assert resolve_resume(str(tmp_path), tmp_path / "run", None) is None
        assert resolve_resume(str(tmp_path), tmp_path / "run", "x/y.pt") == tmp_path / "x/y.pt"

    def test_latest_with_no_checkpoints_is_an_error(self, tmp_path):
        from scripts.train import resolve_resume

        (tmp_path / "run" / "checkpoints").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            resolve_resume(str(tmp_path), tmp_path / "run", "latest")


class TestRetiredModuleNames:
    def test_checkpoint_with_vonmises_keys_loads(self, tmp_path):
        """Every checkpoint before September 2026 names the circular branch
        `latent_space.vonmises`; the loader translates it."""
        trainer, model, optimizer = _setup(3, 2, tmp_path)
        state = {("latent_space.vonmises." + k[len("latent_space.circular."):]
                  if k.startswith("latent_space.circular.") else k): v
                 for k, v in model.state_dict().items()}
        assert any(k.startswith("latent_space.vonmises.") for k in state)
        path = tmp_path / "old.pt"
        torch.save({"epoch": 0, "step": 0, "val_loss": 1.0, "model_state": state,
                    "optimizer_state": optimizer.state_dict()}, path)
        fresh, model2, optimizer2 = _setup(4, 2, tmp_path / "fresh")
        CheckpointManager.load(str(path), model2, optimizer2)
        for k, v in model.state_dict().items():
            assert torch.equal(model2.state_dict()[k], v), k

    def test_checkpoint_with_value_encoder_keys_loads(self, tmp_path):
        """`ValueEncoder` became one ValuePart among several, and its linear
        map moved down into `SharedDirectionValue.scalar_map`. The weights are
        unchanged, so r1 through c1_ref must still load."""
        # The module is registered twice, at the top level and inside the
        # token constructor, and the two retired spellings differ: the token
        # constructor's attribute was `value_encoder` then and is `value_part`
        # now. Both have to translate.
        retired = {
            "value_encoder.scalar_map.linear.": "value_encoder.linear.",
            "token_constructor.value_part.scalar_map.linear.":
                "token_constructor.value_encoder.linear.",
        }
        trainer, model, optimizer = _setup(3, 2, tmp_path)
        state = {}
        for key, value in model.state_dict().items():
            for new, old in retired.items():
                if key.startswith(new):
                    key = old + key[len(new):]
                    break
            state[key] = value
        assert "value_encoder.linear.weight" in state
        assert "token_constructor.value_encoder.linear.weight" in state
        assert not any("scalar_map" in k for k in state)
        assert not any("scalar_map" in k for k in state)
        path = tmp_path / "old_value_encoder.pt"
        torch.save({"epoch": 0, "step": 0, "val_loss": 1.0, "model_state": state,
                    "optimizer_state": optimizer.state_dict()}, path)
        fresh, model2, optimizer2 = _setup(4, 2, tmp_path / "fresh_ve")
        CheckpointManager.load(str(path), model2, optimizer2)
        for k, v in model.state_dict().items():
            assert torch.equal(model2.state_dict()[k], v), k


class TestLossStateSurvivesAResume:
    """The study discriminator lives on CompositeLoss, not on the model, and
    until October 2026 nothing saved it. The optimiser state WAS saved, and
    the optimiser is built over both sets of parameters, so a resume
    restored Adam's moments for an adversary that had just been
    re-initialised at random. Every train_dab reading after a resume was a
    measurement of a fresh classifier."""

    def test_adversary_weights_are_written_to_the_checkpoint(self, tmp_path):
        trainer, model, optimizer = _setup(3, 2, tmp_path)
        trainer.fit()
        path = sorted(tmp_path.glob("epoch_*.pt"))[-1]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert "loss_state" in payload
        assert any(k.startswith("dab.") for k in payload["loss_state"])

    def test_resume_restores_them_exactly(self, tmp_path):
        trainer, model, optimizer = _setup(3, 2, tmp_path)
        trainer.fit()
        path = sorted(tmp_path.glob("epoch_*.pt"))[-1]
        trained = {k: v.clone() for k, v in trainer.composite_loss.state_dict().items()}

        fresh, model2, optimizer2 = _setup(99, 2, tmp_path / "fresh")
        # A different seed, so the fresh adversary cannot coincide with the
        # trained one and a no-op load would fail this test.
        assert not torch.equal(
            fresh.composite_loss.state_dict()["dab.classifier.0.weight"],
            trained["dab.classifier.0.weight"])
        CheckpointManager.load(str(path), model2, optimizer2,
                               loss=fresh.composite_loss)
        for key, value in trained.items():
            assert torch.equal(fresh.composite_loss.state_dict()[key], value), key

    def test_a_checkpoint_without_loss_state_still_loads(self, tmp_path):
        """Every v-series and m-series checkpoint predates the field, and a
        resume of one must warn rather than fail."""
        trainer, model, optimizer = _setup(3, 2, tmp_path)
        path = tmp_path / "no_loss_state.pt"
        torch.save({"epoch": 0, "step": 0, "val_loss": 1.0,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict()}, path)
        fresh, model2, optimizer2 = _setup(4, 2, tmp_path / "fresh_no_ls")
        CheckpointManager.load(str(path), model2, optimizer2,
                               loss=fresh.composite_loss)
