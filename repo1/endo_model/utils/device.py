from __future__ import annotations

import torch


class DeviceManager:
    """Centralised device selection.

    All model components call DeviceManager.get_device() rather than querying
    torch.cuda.is_available() directly.  This makes it easy to add MPS or
    multi-GPU support in one place later.
    """

    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def device_info() -> dict:
        """Return a dict describing the active device (useful for logging)."""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "type": "cuda",
                "name": props.name,
                "count": torch.cuda.device_count(),
                "memory_gb": round(props.total_memory / 1e9, 2),
            }
        return {"type": "cpu"}
