"""The embedding layout CCE compares cells in.

The arithmetic these tests pin down is the reason r5_cce's circular branch
collapsed: a (cos, sin) block has norm sqrt(n_vMF) no matter what it holds, so
under a plain concatenation the Gaussian block can crowd it out of the cosine
similarity simply by growing.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.objectives.cce_embedding import cce_embedding, unit_circular
from endo_model.model.latent.circular import angles_to_cartesian

B, D_GAUSS, N_VMF = 4, 8, 4


def angles(scale: float = 1.0) -> torch.Tensor:
    torch.manual_seed(24)
    return (torch.rand(B, N_VMF) * 2 - 1) * math.pi * scale


# ── the fixed norm that causes the problem ───────────────────────────────────

def test_circular_block_norm_is_sqrt_n_vmf_whatever_the_angles():
    """The premise of the whole fix. If this were data-dependent, the Gaussian
    block could not reliably crowd the circular one out."""
    for scale in (0.01, 0.5, 1.0):
        block = angles_to_cartesian(angles(scale))
        assert torch.allclose(
            block.norm(dim=-1), torch.full((B,), math.sqrt(N_VMF)), atol=1e-5
        )


def test_unit_circular_rescales_that_block_to_one():
    block = unit_circular(angles_to_cartesian(angles()))
    assert torch.allclose(block.norm(dim=-1), torch.ones(B), atol=1e-5)


def test_unit_circular_tolerates_a_model_with_no_circular_dimensions():
    empty = torch.zeros(B, 0)
    assert unit_circular(empty).shape == (B, 0)


# ── shape is unchanged either way ────────────────────────────────────────────

@pytest.mark.parametrize("block_normalise", [False, True])
def test_width_is_d_gauss_plus_two_n_vmf(block_normalise):
    out = cce_embedding(torch.randn(B, D_GAUSS), angles(),
                        block_normalise=block_normalise)
    assert out.shape == (B, D_GAUSS + 2 * N_VMF)


def test_default_reproduces_the_plain_concatenation():
    """Every pre-R6 config relies on this default, so it is what keeps those
    configs reproducing the runs they name."""
    mu = torch.randn(B, D_GAUSS)
    a = angles()
    expected = torch.cat([mu, angles_to_cartesian(a)], dim=-1)
    assert torch.allclose(cce_embedding(mu, a), expected)


# ── the failure mode, and that the fix removes it ────────────────────────────

def _circular_share(mu_gauss, a, **kw):
    """Circular block's share of the L2-normalised embedding — what InfoNCE
    actually compares, since it normalises before the cosine similarity."""
    emb = cce_embedding(mu_gauss, a, **kw)
    circ = emb[:, D_GAUSS:]
    return (circ.norm(dim=-1) / emb.norm(dim=-1)).mean().item()


def test_without_the_fix_a_large_gaussian_block_crowds_the_circular_one_out():
    """The r5_cce failure, reproduced. Growing mu_gauss costs the model nothing
    -- the decoder's linear layer rescales it away -- so this is the cheapest
    route to reducing the circular block's influence on the contrastive loss."""
    a = angles()
    small = _circular_share(torch.randn(B, D_GAUSS) * 0.1, a)
    large = _circular_share(torch.randn(B, D_GAUSS) * 100.0, a)
    assert small > 0.5, "circular block should dominate when mu_gauss is small"
    assert large < 0.02, "circular block should be crowded out when it is large"


def test_with_the_fix_the_share_does_not_move_with_the_gaussian_scale():
    a = angles()
    small = _circular_share(torch.randn(B, D_GAUSS) * 0.1, a, block_normalise=True)
    large = _circular_share(torch.randn(B, D_GAUSS) * 100.0, a, block_normalise=True)
    assert small == pytest.approx(large, abs=1e-5)


def test_equal_weight_splits_the_embedding_evenly():
    """At circular_weight 1.0 both blocks are unit length, so each takes
    1/sqrt(2) of the normalised embedding."""
    got = _circular_share(torch.randn(B, D_GAUSS), angles(),
                          block_normalise=True, circular_weight=1.0)
    assert got == pytest.approx(1 / math.sqrt(2), abs=1e-5)


@pytest.mark.parametrize("w", [0.25, 1.0, 4.0])
def test_circular_weight_sets_the_share_to_w_over_sqrt_one_plus_w_squared(w):
    got = _circular_share(torch.randn(B, D_GAUSS), angles(),
                          block_normalise=True, circular_weight=w)
    assert got == pytest.approx(w / math.sqrt(1 + w * w), abs=1e-5)


def test_circular_weight_is_ignored_without_block_normalise():
    """It scales a block that is about to be renormalised anyway, so reading it
    in the plain layout would silently change a run that did not ask for it."""
    mu, a = torch.randn(B, D_GAUSS), angles()
    assert torch.allclose(cce_embedding(mu, a, circular_weight=7.0),
                          cce_embedding(mu, a, circular_weight=1.0))


def test_gradient_reaches_both_blocks_through_the_fix():
    """The fix must not detach the circular branch — that would 'stop the
    collapse' by removing the branch from the loss entirely."""
    mu = torch.randn(B, D_GAUSS, requires_grad=True)
    a = angles().requires_grad_(True)
    cce_embedding(mu, a, block_normalise=True).sum().backward()
    assert mu.grad is not None and mu.grad.abs().sum() > 0
    assert a.grad is not None and a.grad.abs().sum() > 0
