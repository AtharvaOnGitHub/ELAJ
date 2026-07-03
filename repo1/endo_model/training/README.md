# endo_model/training — Training Loop

## Modules

| File | Class | Purpose |
|------|-------|---------|
| `trainer.py` | `Trainer` | Epoch/batch loop, gradient clipping, CCE double-pass, val loop |
| `callbacks.py` | `EarlyStopping` | Stop when val_loss stagnates for `patience` epochs |
| `callbacks.py` | `MetricLogger` | Accumulates per-step metrics, computes epoch averages |
| `checkpointing.py` | `CheckpointManager` | Save/load `.pt` checkpoints; retain only the best N by val_loss |

## Usage sketch

```python
from endo_model.model.endo_model import EndoFoundationModel
from endo_model.model.objectives.composite import CompositeLoss
from endo_model.training.trainer import Trainer
from endo_model.configs.training_config import TrainingConfig

model = EndoFoundationModel(vocab, model_config)
composite_loss = CompositeLoss(n_studies=n_studies, d_gauss=32, n_vMF=8)
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(composite_loss.parameters()),
    lr=3e-4,
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    composite_loss=composite_loss,
    optimizer=optimizer,
    scheduler=None,           # or a torch LRScheduler
    config=TrainingConfig(),
    checkpoint_dir="checkpoints/",
    cce_enabled=False,        # set True to enable double forward pass
)

history = trainer.fit()
```

## CCE (double forward pass)

When `cce_enabled=True`, the Trainer runs a second forward pass of each batch under a different dropout mask and passes both outputs to `CompositeLoss.forward(model_output2=...)`.  The InfoNCE loss then pushes the two `mu_gauss` embeddings for the same cell to agree.

## Checkpointing

`CheckpointManager` saves `epoch_{:04d}_step_{:07d}.pt` files containing model state, optimizer state, epoch/step numbers, and val_loss.  Only the best `max_to_keep` checkpoints (lowest val_loss) are retained.

```python
# Load best checkpoint
CheckpointManager.load(ckpt_manager.best_path, model, optimizer)
```
