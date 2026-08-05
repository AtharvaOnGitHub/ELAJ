"""BatchedGroupEncoder must be the same model, not merely a faster one.

Option B is a performance change with no modelling content: the 45 per-group
encoders keep their own weights and their own independence, they are just
executed as grouped matmuls instead of 45 Python calls. That claim is only worth
anything if it is checked, so the central test here copies weights into 45
genuine `nn.TransformerEncoderLayer` stacks and compares outputs elementwise.

The other tests cover what batching can quietly break: padding leaking between
groups of different sizes, and groups influencing each other.
"""

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from endo_model.model.encoders.batched import BatchedGroupEncoder  # noqa: E402

D_TOKEN, N_HEADS, N_LAYERS, B = 8, 2, 2, 3
SIZES = {"alpha": 5, "beta": 9, "gamma": 4}   # deliberately unequal
NAMES = list(SIZES)


def make_batched(dropout: float = 0.0) -> BatchedGroupEncoder:
    torch.manual_seed(0)
    enc = BatchedGroupEncoder(
        NAMES, SIZES, D_TOKEN, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=dropout
    )
    # Random non-trivial affine params: LayerNorm defaults to weight=1/bias=0,
    # which would hide a mis-indexed per-group affine.
    with torch.no_grad():
        for layer in enc.layers:
            for norm in (layer.norm1, layer.norm2):
                norm.weight.uniform_(0.5, 1.5)
                norm.bias.uniform_(-0.5, 0.5)
    return enc.eval()


def reference_stack(enc: BatchedGroupEncoder, k: int) -> nn.TransformerEncoder:
    """Build a real nn.TransformerEncoder holding group k's weights.

    torch stores Linear weights as (out, in) and packs q/k/v into a single
    `in_proj_weight`; BatchedLinear stores (in, out) because that is the
    orientation baddbmm wants. Hence the transposes.
    """
    layer = nn.TransformerEncoderLayer(
        d_model=D_TOKEN, nhead=N_HEADS, dim_feedforward=4 * D_TOKEN,
        dropout=0.0, batch_first=True, norm_first=True,
    )
    stack = nn.TransformerEncoder(
        layer, num_layers=N_LAYERS, enable_nested_tensor=False
    ).eval()

    with torch.no_grad():
        for ref, src in zip(stack.layers, enc.layers):
            ref.norm1.weight.copy_(src.norm1.weight[k])
            ref.norm1.bias.copy_(src.norm1.bias[k])
            ref.norm2.weight.copy_(src.norm2.weight[k])
            ref.norm2.bias.copy_(src.norm2.bias[k])

            ref.self_attn.in_proj_weight.copy_(src.self_attn.in_proj.weight[k].T)
            ref.self_attn.in_proj_bias.copy_(src.self_attn.in_proj.bias[k])
            ref.self_attn.out_proj.weight.copy_(src.self_attn.out_proj.weight[k].T)
            ref.self_attn.out_proj.bias.copy_(src.self_attn.out_proj.bias[k])

            ref.linear1.weight.copy_(src.linear1.weight[k].T)
            ref.linear1.bias.copy_(src.linear1.bias[k])
            ref.linear2.weight.copy_(src.linear2.weight[k].T)
            ref.linear2.bias.copy_(src.linear2.bias[k])
    return stack


def make_tokens() -> dict:
    torch.manual_seed(1)
    return {name: torch.randn(B, size, D_TOKEN) for name, size in SIZES.items()}


# --- The equivalence claim --------------------------------------------------


def test_matches_independent_transformer_encoders():
    """The test Option B stands or falls on."""
    enc = make_batched()
    tokens = make_tokens()

    with torch.no_grad():
        got = enc(tokens)

    for k, name in enumerate(NAMES):
        stack = reference_stack(enc, k)
        seq = torch.cat(
            [enc.cls_token[k].unsqueeze(0).expand(B, 1, D_TOKEN), tokens[name]],
            dim=1,
        )
        with torch.no_grad():
            expected = stack(seq)[:, 0, :]
        assert torch.allclose(got[name], expected, atol=1e-5), (
            f"group {name!r} diverged from nn.TransformerEncoder: "
            f"max |diff| = {(got[name] - expected).abs().max().item():.3e}"
        )


# --- What batching can quietly break ----------------------------------------


def test_padding_does_not_leak_into_shorter_groups():
    """The largest group sets the padded length; shorter ones must not notice.

    Feeding the shortest group alone through a batch sized by the longest is
    exactly the case a wrong or missing key mask would corrupt.
    """
    enc = make_batched()
    tokens = make_tokens()
    with torch.no_grad():
        baseline = enc(tokens)["gamma"]          # the 4-gene group

    # Change only the longest group. gamma is padded to beta's length, so if the
    # mask were wrong, gamma's CLS could pick up beta's padding region.
    tokens["beta"] = torch.randn(B, SIZES["beta"], D_TOKEN)
    with torch.no_grad():
        after = enc(tokens)["gamma"]
    assert torch.equal(baseline, after), "a group's output changed with another group's input"


def test_groups_are_independent():
    """Perturbing one group's tokens must move that group and no other."""
    enc = make_batched()
    tokens = make_tokens()
    with torch.no_grad():
        baseline = enc(tokens)

    for changed in NAMES:
        perturbed = dict(tokens)
        perturbed[changed] = perturbed[changed] + 3.0
        with torch.no_grad():
            out = enc(perturbed)
        assert not torch.allclose(out[changed], baseline[changed]), (
            f"{changed} ignored its own tokens"
        )
        for other in NAMES:
            if other != changed:
                assert torch.equal(out[other], baseline[other]), (
                    f"changing {changed} moved {other}"
                )


def test_output_shapes_and_keys():
    enc = make_batched()
    out = enc(make_tokens())
    assert set(out) == set(NAMES)
    for name in NAMES:
        assert out[name].shape == (B, D_TOKEN)


def test_parameter_count_matches_independent_encoders():
    """Same model means same parameter count, not merely the same interface."""
    enc = make_batched()
    batched = sum(p.numel() for p in enc.parameters())

    reference = 0
    for _ in NAMES:
        layer = nn.TransformerEncoderLayer(
            d_model=D_TOKEN, nhead=N_HEADS, dim_feedforward=4 * D_TOKEN,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        stack = nn.TransformerEncoder(
            layer, num_layers=N_LAYERS, enable_nested_tensor=False
        )
        reference += sum(p.numel() for p in stack.parameters())
        reference += D_TOKEN            # that group's CLS token

    assert batched == reference, f"batched {batched} vs independent {reference}"


def test_gradients_reach_every_group():
    enc = make_batched()
    out = enc(make_tokens())
    sum(v.sum() for v in out.values()).backward()
    for name, param in enc.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert param.grad.abs().sum() > 0, f"{name} gradient is all zero"
