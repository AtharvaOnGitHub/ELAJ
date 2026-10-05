"""CheckpointManager: save and load model/optimizer state.

Checkpoints are stored as:
    {checkpoint_dir}/epoch_{epoch:04d}_step_{step:07d}.pt

Only the best N checkpoints are retained (by validation loss) to avoid
filling disk.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# Submodule names retired in September 2026 -> their current names. Every
# checkpoint from r1 to c_ref carries the old ones; renaming the attribute
# without this would make all of them unloadable.
_RENAMED_MODULES = {
    "latent_space.vonmises.": "latent_space.circular.",
    # ValueEncoder became one ValuePart among several, and the linear map it
    # was moved a level down into SharedDirectionValue.scalar_map. The weights
    # are unchanged, so every checkpoint from r1 to c1_ref still loads.
    "value_encoder.linear.": "value_encoder.scalar_map.linear.",
    # The token constructor's attribute was renamed too: it holds a ValuePart
    # now, of which the linear map is one, so it is `value_part` there.
    "token_constructor.value_encoder.linear.":
        "token_constructor.value_part.scalar_map.linear.",
}


def _renamed_keys(state: dict) -> dict:
    """A state_dict with retired submodule prefixes translated."""
    out = {}
    for key, value in state.items():
        for old, new in _RENAMED_MODULES.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        out[key] = value
    return out


def _widened_dsbn_statistics(state: dict, model: "torch.nn.Module") -> dict:
    """Give an old checkpoint's single batch-norm statistics one row per study.

    DSBN used to hold one `nn.BatchNorm1d` with one running mean and variance
    for every study; it now holds one row per study. Every checkpoint from r1
    to the v-series carries the old shape, and `load_state_dict` would reject
    it on shape rather than name, so `_renamed_keys` cannot help.

    Copying the pooled statistics into every row reproduces exactly what the
    old checkpoint did at evaluation: every study normalised by the corpus
    average. That is the bug, faithfully. It is the right migration because
    the alternative is inventing per-study numbers that were never measured,
    and because a run resumed from here specialises the rows as soon as it
    trains. A checkpoint loaded only to be SCORED still shows the old
    behaviour, which is what makes old and new scores comparable.
    """
    target = model.state_dict()
    out = {}
    for key, value in state.items():
        want = target.get(key)
        if (want is not None and hasattr(value, "shape")
                and value.shape != want.shape
                and key.endswith(("running_mean", "running_var",
                                  "num_batches_tracked"))):
            logger.info("Widening %s from %s to %s: one row per study.",
                        key, tuple(value.shape), tuple(want.shape))
            value = value.expand_as(want).clone() if value.dim() else \
                value.repeat(want.shape[0])
        out[key] = value
    return out


def _migrated(state: dict, model: "torch.nn.Module") -> dict:
    """Every migration an old checkpoint needs, in one place."""
    return _widened_dsbn_statistics(_renamed_keys(state), model)


def _epoch_of(path: Path) -> int:
    """The epoch a checkpoint filename records, or -1 if it does not.

    Filenames are `epoch_{epoch:04d}_step_{step:07d}.pt`, so the epoch is
    recoverable without opening the file. -1 for anything unparseable, which
    the retention policy will never schedule and so will delete, matching the
    old behaviour for files it did not write.
    """
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return -1


class CheckpointManager:
    """Saves model checkpoints and retains only the best few.

    Args:
        checkpoint_dir: Directory to write checkpoints.
        max_to_keep:    Maximum number of checkpoints to retain.
    """

    def __init__(self, checkpoint_dir: str, max_to_keep: int = 3,
                 policy: "RetentionPolicy | None" = None) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        # When a policy is given it replaces top-k entirely. See
        # training/retention.py for why top-k is the wrong rule when the
        # ranking metric is itself under investigation.
        self.policy = policy
        self._history: list[tuple[float, Path]] = []  # (val_loss, path)
        self.rehydrate()

    def rehydrate(self) -> int:
        """Adopt the checkpoints already in the directory.

        A resumed run starts with a fresh manager. Without this it would keep
        `max_to_keep` NEW checkpoints beside every old one, never prune the
        old ones, and report a `best_path` that ignores them. Each file
        carries its own val_loss, so the ranking is rebuilt from disk.
        Returns the number adopted.
        """
        found = []
        for path in sorted(self.checkpoint_dir.glob("epoch_*.pt")):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                found.append((float(payload["val_loss"]), path))
            except Exception as exc:                            # noqa: BLE001
                logger.warning("Ignoring unreadable checkpoint %s (%s: %s)",
                               path.name, type(exc).__name__, exc)
        self._history = sorted(found, key=lambda x: x[0])
        if found:
            logger.info("Checkpoint manager adopted %d existing checkpoint(s); "
                        "best %s", len(found), self._history[0][1].name)
        return len(found)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        val_loss: float,
        extra: dict | None = None,
        loss: "torch.nn.Module | None" = None,
    ) -> Path:
        """Save a checkpoint and prune old ones.

        Args:
            model:     Model whose state_dict to save.
            optimizer: Optimizer whose state_dict to save.
            epoch:     Current epoch (for naming and metadata).
            step:      Global step (for naming and metadata).
            val_loss:  Validation loss at this checkpoint.
            extra:     Additional metadata to include.
            loss:      CompositeLoss, which owns parameters of its own.

        Returns:
            Path to the saved checkpoint file.

        `loss_state` WAS MISSING UNTIL OCTOBER 2026, AND IT MATTERED.
        CompositeLoss owns the study discriminator, and it is not part of
        `model`, so it was never saved. The phase head is on the model and
        always was; the adversary is the one that went missing. The
        OPTIMISER state was saved, and the optimiser is built over both
        sets of parameters, so a resumed run restored Adam's moments for an
        adversary whose weights had just been re-initialised at random. The
        adversary restarted at chance while the encoder carried on, which
        makes every `train_dab` reading after a resume a measurement of a
        fresh classifier rather than of the latent.

        Old checkpoints have no `loss_state`. `load` therefore treats it as
        optional and says so in the log rather than failing, which keeps the
        v-series and m-series checkpoints loadable.
        """
        fname = self.checkpoint_dir / f"epoch_{epoch:04d}_step_{step:07d}.pt"
        payload = {
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        if loss is not None:
            payload["loss_state"] = loss.state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, fname)

        self._history.append((val_loss, fname))
        self._history.sort(key=lambda x: x[0])  # ascending by val_loss
        self.prune()
        return fname

    def prune(self) -> None:
        """Delete the checkpoints the policy does not keep.

        With no policy this is the original rule: the best `max_to_keep` by
        metric. With one, the metric decides only which single checkpoint is
        "the best"; the schedule and the last epoch are kept regardless of
        what it says about them.
        """
        if self.policy is None:
            while len(self._history) > self.max_to_keep:
                _, old_path = self._history.pop()
                if old_path.exists():
                    old_path.unlink()
            return

        entries = [(_epoch_of(path), loss) for loss, path in self._history]
        keep = self.policy.survivors(entries)
        remaining = []
        for loss, path in self._history:
            if _epoch_of(path) in keep:
                remaining.append((loss, path))
            elif path.exists():
                path.unlink()
        self._history = remaining

    @staticmethod
    def load_weights(
        path: str,
        model: torch.nn.Module,
        map_location: str = "cpu",
    ) -> dict:
        """Load a checkpoint's weights only, and start a fresh run from them.

        Different from `load`, which continues a run: this reads no optimiser
        state, and the caller keeps its own epoch, step, schedule, and
        history. It is what `training.init_from_checkpoint` does, and what a
        frozen-encoder run needs, since such a run trains a different set of
        parameters from the one that wrote the checkpoint and its optimiser
        state would not apply.

        `strict=True` deliberately. A frozen-encoder run's whole premise is
        that the encoder carries trained weights, so a key that fails to match
        means it does not, and that must stop the run rather than leave one
        module at its random initialisation.

        Returns:
            The payload, minus the states, so the caller can log where the
            weights came from.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(_migrated(payload["model_state"], model))
        logger.info("Loaded weights from %s (its epoch %s, step %s)", path,
                    payload.get("epoch", "?"), payload.get("step", "?"))
        return {k: v for k, v in payload.items()
                if k not in ("model_state", "optimizer_state")}

    @staticmethod
    def load(
        path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str = "cpu",
        loss: "torch.nn.Module | None" = None,
    ) -> dict:
        """Load a checkpoint into model (and optionally optimizer and loss).

        Args:
            path:         Path to the .pt checkpoint file.
            model:        Model to load state into.
            optimizer:    Optimizer to restore (optional).
            map_location: torch device string for map_location.
            loss:         CompositeLoss to restore (optional). See `save` for
                          what a resume did without it.

        Returns:
            The full checkpoint payload dict.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(_migrated(payload["model_state"], model))
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        CheckpointManager._restore_loss(loss, payload, path)
        return payload

    @staticmethod
    def _restore_loss(loss, payload: dict, path: str) -> None:
        """Put the study discriminator back, if the file carries it.

        Loud about the three cases, because the silent one was the bug: a
        resume that re-initialised the adversary looked exactly like a resume
        that restored it, apart from a train_dab that jumped back to chance,
        which is also what a working adversary looks like.
        """
        if loss is None:
            return
        state = payload.get("loss_state")
        if state is None:
            logger.warning(
                "%s has no loss_state: it predates October 2026. The study "
                "discriminator starts from a fresh initialisation, so "
                "train_dab will read near ln(n_studies) for the first "
                "epochs regardless of what the latent holds.",
                path)
            return
        loss.load_state_dict(state)
        logger.info("Restored the loss module's own parameters from %s", path)

    @property
    def best_path(self) -> Path | None:
        """Path to the checkpoint with the lowest validation loss."""
        if not self._history:
            return None
        return self._history[0][1]
