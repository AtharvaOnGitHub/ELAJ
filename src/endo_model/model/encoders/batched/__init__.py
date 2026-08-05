"""Grouped execution of the per-group encoders.

The model keeps one independent transformer per gene group. Running them as 45
Python calls left an A100 idle 30% of the time and at ~7% of peak while busy,
because every kernel was too small to fill the card. These modules hold the same
45 weight sets stacked along a leading K axis and drive them with grouped
matmuls, so the work arrives as a few large kernels instead of thousands of tiny
ones.

Same parameters, same independence, same outputs — verified against
`nn.TransformerEncoderLayer` in tests/test_model/test_batched_encoder.py.
"""

from .bucketing import BucketedGroupEncoder, SizeBucket, plan_buckets
from .encoder import BatchedEncoderLayer, BatchedGroupEncoder, BatchedSelfAttention
from .primitives import BatchedLayerNorm, BatchedLinear

__all__ = [
    "BatchedEncoderLayer",
    "BatchedGroupEncoder",
    "BatchedLayerNorm",
    "BatchedLinear",
    "BatchedSelfAttention",
    "BucketedGroupEncoder",
    "SizeBucket",
    "plan_buckets",
]
