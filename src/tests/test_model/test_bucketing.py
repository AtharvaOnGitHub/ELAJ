"""Bucketing is a scheduling change, so the model must not notice it.

The whole claim is that sorting groups into size buckets changes only how the
work reaches the GPU. If a bucketed encoder and an unbucketed one holding the
same weights disagree, or if the parameter count moves, the speedup was bought
with a silent modelling change.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.encoders.batched import (  # noqa: E402
    BatchedGroupEncoder,
    BucketedGroupEncoder,
    plan_buckets,
)

D_TOKEN, N_HEADS, N_LAYERS, B = 8, 2, 2, 3
# Deliberately lopsided: an even split would bucket these badly, and two groups
# share a size so the uniform-bucket path gets exercised.
SIZES = {"a": 4, "b": 4, "c": 9, "d": 10, "e": 30, "f": 31}
NAMES = list(SIZES)


def make_tokens() -> dict:
    torch.manual_seed(1)
    return {name: torch.randn(B, size, D_TOKEN) for name, size in SIZES.items()}


# --- the partition ----------------------------------------------------------


def test_every_group_lands_in_exactly_one_bucket():
    buckets = plan_buckets(SIZES, n_buckets=3)
    seen = [name for bucket in buckets for name in bucket.names]
    assert sorted(seen) == sorted(NAMES)


def test_bucket_seq_len_covers_its_largest_group():
    for bucket in plan_buckets(SIZES, n_buckets=3):
        largest = max(SIZES[name] for name in bucket.names)
        assert bucket.seq_len == largest + 1, "a group would be truncated"


def test_bucketing_never_increases_padded_work():
    """More buckets can only help — the 1-bucket case is the old behaviour."""
    single = plan_buckets(SIZES, n_buckets=1)
    baseline = sum(b.attention_work for b in single)
    for n in range(2, len(SIZES) + 1):
        got = sum(b.attention_work for b in plan_buckets(SIZES, n))
        assert got <= baseline, f"{n} buckets cost more than one"


def test_partition_beats_an_even_split():
    """The point of the DP: an even split is not optimal on lopsided sizes."""
    seq = sorted(s + 1 for s in SIZES.values())
    half = len(seq) // 2
    even = len(seq[:half]) * max(seq[:half]) ** 2 + \
        len(seq[half:]) * max(seq[half:]) ** 2
    optimal = sum(b.attention_work for b in plan_buckets(SIZES, 2))
    assert optimal <= even


def test_more_buckets_than_groups_is_clamped():
    buckets = plan_buckets(SIZES, n_buckets=999)
    assert len(buckets) == len(SIZES)
    assert all(len(b.names) == 1 for b in buckets)


# --- equivalence to the unbucketed encoder ----------------------------------


def _copy_weights(dst: BucketedGroupEncoder, src: BatchedGroupEncoder) -> None:
    """Give every bucketed group the weights its unbucketed twin holds."""
    with torch.no_grad():
        for encoder in dst.encoders:
            for local, name in enumerate(encoder.group_names):
                k = src.group_names.index(name)
                encoder.cls_token[local].copy_(src.cls_token[k])
                for ref, own in zip(src.layers, encoder.layers):
                    for attr in ("norm1", "norm2"):
                        getattr(own, attr).weight[local].copy_(
                            getattr(ref, attr).weight[k]
                        )
                        getattr(own, attr).bias[local].copy_(
                            getattr(ref, attr).bias[k]
                        )
                    for attr in ("linear1", "linear2"):
                        getattr(own, attr).weight[local].copy_(
                            getattr(ref, attr).weight[k]
                        )
                        getattr(own, attr).bias[local].copy_(
                            getattr(ref, attr).bias[k]
                        )
                    own.self_attn.in_proj.weight[local].copy_(
                        ref.self_attn.in_proj.weight[k]
                    )
                    own.self_attn.in_proj.bias[local].copy_(
                        ref.self_attn.in_proj.bias[k]
                    )
                    own.self_attn.out_proj.weight[local].copy_(
                        ref.self_attn.out_proj.weight[k]
                    )
                    own.self_attn.out_proj.bias[local].copy_(
                        ref.self_attn.out_proj.bias[k]
                    )


def test_matches_the_unbucketed_encoder():
    """The claim bucketing stands or falls on."""
    torch.manual_seed(0)
    unbucketed = BatchedGroupEncoder(
        NAMES, SIZES, D_TOKEN, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=0.0
    ).eval()
    bucketed = BucketedGroupEncoder(
        NAMES, SIZES, D_TOKEN, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=0.0, n_buckets=3,
    ).eval()
    _copy_weights(bucketed, unbucketed)

    tokens = make_tokens()
    with torch.no_grad():
        expected = unbucketed(tokens)
        got = bucketed(tokens)

    assert set(got) == set(expected)
    for name in NAMES:
        assert torch.allclose(got[name], expected[name], atol=1e-5), (
            f"group {name!r} diverged: "
            f"max |diff| = {(got[name] - expected[name]).abs().max().item():.3e}"
        )


def test_parameter_count_is_unchanged():
    torch.manual_seed(0)
    unbucketed = BatchedGroupEncoder(
        NAMES, SIZES, D_TOKEN, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=0.0
    )
    for n in (1, 2, 3, len(SIZES)):
        bucketed = BucketedGroupEncoder(
            NAMES, SIZES, D_TOKEN, n_heads=N_HEADS, n_layers=N_LAYERS,
            dropout=0.0, n_buckets=n,
        )
        assert sum(p.numel() for p in bucketed.parameters()) == sum(
            p.numel() for p in unbucketed.parameters()
        ), f"{n} buckets changed the parameter count"


# --- the mask, which is what unlocks FlashAttention -------------------------


def test_uniform_bucket_carries_no_mask():
    """No padding means no mask, and no mask is what reaches flash in bf16."""
    encoder = BatchedGroupEncoder(
        ["a", "b"], {"a": 4, "b": 4}, D_TOKEN,
        n_heads=N_HEADS, n_layers=N_LAYERS, dropout=0.0,
    )
    assert encoder.key_valid is None


def test_ragged_bucket_keeps_its_mask():
    """Dropping the mask where sizes differ would let padding into the CLS."""
    encoder = BatchedGroupEncoder(
        ["a", "c"], {"a": 4, "c": 9}, D_TOKEN,
        n_heads=N_HEADS, n_layers=N_LAYERS, dropout=0.0,
    )
    assert encoder.key_valid is not None
    assert not encoder.key_valid[0, 5:].any(), "group a's padding is not masked"


def test_one_group_per_bucket_drops_every_mask():
    bucketed = BucketedGroupEncoder(
        NAMES, SIZES, D_TOKEN, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=0.0, n_buckets=len(SIZES),
    )
    assert all(enc.key_valid is None for enc in bucketed.encoders)


def test_gradients_reach_every_group():
    bucketed = BucketedGroupEncoder(
        NAMES, SIZES, D_TOKEN, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=0.0, n_buckets=3,
    )
    out = bucketed(make_tokens())
    sum(v.sum() for v in out.values()).backward()
    for name, param in bucketed.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert param.grad.abs().sum() > 0, f"{name} gradient is all zero"
