"""CheckpointManager: save and load model/optimizer state.

Checkpoints are stored as:
    {checkpoint_dir}/epoch_{epoch:04d}_step_{step:07d}.pt

Only the best N checkpoints are retained (by validation loss) to avoid
filling disk.
"""

import os
from pathlib import Path

import torch


class CheckpointManager:
    """Saves model checkpoints and retains only the best few.

    Args:
        checkpoint_dir: Directory to write checkpoints.
        max_to_keep:    Maximum number of checkpoints to retain.
    """

    def __init__(self, checkpoint_dir: str, max_to_keep: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self._history: list[tuple[float, Path]] = []  # (val_loss, path)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        val_loss: float,
        extra: dict | None = None,
    ) -> Path:
        """Save a checkpoint and prune old ones.

        Args:
            model:     Model whose state_dict to save.
            optimizer: Optimizer whose state_dict to save.
            epoch:     Current epoch (for naming and metadata).
            step:      Global step (for naming and metadata).
            val_loss:  Validation loss at this checkpoint.
            extra:     Additional metadata to include.

        Returns:
            Path to the saved checkpoint file.
        """
        fname = self.checkpoint_dir / f"epoch_{epoch:04d}_step_{step:07d}.pt"
        payload = {
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, fname)

        self._history.append((val_loss, fname))
        self._history.sort(key=lambda x: x[0])  # ascending by val_loss

        # Remove worst checkpoints beyond max_to_keep
        while len(self._history) > self.max_to_keep:
            _, old_path = self._history.pop()
            if old_path.exists():
                old_path.unlink()

        return fname

    @staticmethod
    def load(
        path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str = "cpu",
    ) -> dict:
        """Load a checkpoint into model (and optionally optimizer).

        Args:
            path:         Path to the .pt checkpoint file.
            model:        Model to load state into.
            optimizer:    Optimizer to restore (optional).
            map_location: torch device string for map_location.

        Returns:
            The full checkpoint payload dict.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        return payload

    @property
    def best_path(self) -> Path | None:
        """Path to the checkpoint with the lowest validation loss."""
        if not self._history:
            return None
        return self._history[0][1]
