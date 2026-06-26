import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all RNG sources for full reproducibility.

    Sets Python's random, NumPy, PyTorch CPU and CUDA seeds, and forces
    cuDNN into deterministic mode.  Call this once at the start of every
    training run before any weights are initialised.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
