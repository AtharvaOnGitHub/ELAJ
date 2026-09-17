"""The two wave R7 changes: gradient checkpointing, and the concentration floor.

Both are the kind of change that alters training silently if it is wrong —
checkpointing by not reproducing the un-checkpointed computation, the floor by
not actually binding — so both are pinned here rather than trusted.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.encoders.batched.encoder import BatchedGroupEncoder
from endo_model.model.latent.circular_vae import CircularBranch

GROUPS = ["g0", "g1", "g2"]
SIZES = {"g0": 6, "g1": 9, "g2": 9}      # uneven, so the padding mask is live
D_TOKEN, B = 8, 4


def tokens():
    torch.manual_seed(24)
    return {n: torch.randn(B, SIZES[n], D_TOKEN) for n in GROUPS}


def encoder(grad_checkpointing: bool, dropout: float = 0.0):
    torch.manual_seed(7)                  # same init for both variants
    return BatchedGroupEncoder(
        GROUPS, SIZES, D_TOKEN, n_heads=2, n_layers=2,
        dropout=dropout, grad_checkpointing=grad_checkpointing,
    )


# ── gradient checkpointing ───────────────────────────────────────────────────

def test_checkpointing_gives_the_same_forward_output():
    """It must be a memory/time trade and nothing else. A different output
    means the recomputation is not reproducing the original pass."""
    x = tokens()
    plain, ckpt = encoder(False), encoder(True)
    plain.train(), ckpt.train()
    a, b = plain(x), ckpt(x)
    for name in GROUPS:
        assert torch.allclose(a[name], b[name], atol=1e-6), name


def test_checkpointing_gives_the_same_gradients():
    """The forward matching is not enough — checkpointing rebuilds the graph,
    and it is the backward that training actually consumes."""
    grads = {}
    for flag in (False, True):
        enc = encoder(flag)
        enc.train()
        x = {n: t.clone().requires_grad_(True) for n, t in tokens().items()}
        sum(enc(x)[n].square().sum() for n in GROUPS).backward()
        grads[flag] = {n: x[n].grad.clone() for n in GROUPS}
    for name in GROUPS:
        assert torch.allclose(grads[False][name], grads[True][name], atol=1e-6), name


def test_checkpointing_is_off_in_eval_so_inference_pays_nothing():
    """Nothing is recomputed when there is no backward to recompute for."""
    enc = encoder(True)
    enc.eval()
    with torch.no_grad():
        out = enc(tokens())
    assert set(out) == set(GROUPS)


def test_checkpointing_defaults_off():
    """Every pre-R7 config relies on this to keep reproducing its own run."""
    assert encoder(False).grad_checkpointing is False
    assert BatchedGroupEncoder(GROUPS, SIZES, D_TOKEN).grad_checkpointing is False


# ── the concentration floor ──────────────────────────────────────────────────

def test_kappa_never_falls_below_the_floor():
    """The point of the floor: kappa cannot reach the flat region where
    dKL/dkappa ~ kappa/2 vanishes and a dimension can no longer climb back."""
    branch = CircularBranch(d_agg=16, n_circular=4, kappa_floor=0.5)
    torch.manual_seed(0)
    # Drive the head hard negative — softplus underflows to ~0, so whatever
    # survives is the floor and nothing else.
    with torch.no_grad():
        branch.kappa_head.weight.fill_(0.0)
        branch.kappa_head.bias.fill_(-50.0)
    _, kappa, _ = branch(torch.randn(32, 16))
    assert (kappa >= 0.5).all()
    # .item(), not the bare tensor: kappa carries grad here (see the
    # differentiability test below), and pytest.approx converts via numpy,
    # which refuses a tensor that requires grad.
    assert kappa.max().item() == pytest.approx(0.5, abs=1e-5)


def test_the_floor_is_a_floor_not_a_clamp_on_large_values():
    """It must not cap concentration — a dimension carrying real signal needs
    to be able to run to high kappa."""
    branch = CircularBranch(d_agg=16, n_circular=4, kappa_floor=0.5)
    with torch.no_grad():
        branch.kappa_head.weight.fill_(0.0)
        branch.kappa_head.bias.fill_(10.0)
    _, kappa, _ = branch(torch.randn(32, 16))
    assert (kappa > 10.0).all()


def test_default_floor_is_the_pre_r7_value():
    """1e-4, so every earlier config still trains the model it names — the R7
    configs set 0.5 explicitly."""
    assert CircularBranch(d_agg=16, n_circular=4).kappa_floor == pytest.approx(1e-4)


def test_kappa_stays_differentiable_through_the_floor():
    """Adding a constant must not detach the head; a floor that stopped the
    gradient would freeze concentration rather than protect it."""
    branch = CircularBranch(d_agg=16, n_circular=4, kappa_floor=0.5)
    _, kappa, _ = branch(torch.randn(8, 16))
    kappa.sum().backward()
    assert branch.kappa_head.weight.grad is not None
    assert branch.kappa_head.weight.grad.abs().sum() > 0


# ── the R8 circular parameterisation ─────────────────────────────────────────

class TestCircularParameterisation:
    """Predicting the mean DIRECTION rather than the mean ANGLE.

    Up to R7 the head emitted one unbounded number per dimension and cos/sin
    turned it into a point. cos and sin repeat every 2*pi, so the head could
    drift to any magnitude and alias — and nothing opposed the drift, because
    the circular KL is a function of kappa alone and the angle is absent from
    it. Predicting two numbers and dividing by their length discards the
    magnitude instead of wrapping it.
    """

    D_AGG, N_CIRC, B = 16, 4, 32

    def branch(self, parameterisation: str):
        torch.manual_seed(24)
        return CircularBranch(
            d_agg=self.D_AGG, n_circular=self.N_CIRC, kappa_floor=1.0,
            parameterisation=parameterisation,
        )

    def test_head_width_doubles_for_direction(self):
        assert self.branch("angle").mu_head.out_features == self.N_CIRC
        assert self.branch("direction").mu_head.out_features == 2 * self.N_CIRC

    def test_latent_shapes_are_unchanged_either_way(self):
        """The latent is still two numbers per circular dimension, so nothing
        downstream — DAB, CCE, the decoder — sees a different shape."""
        for p in ("angle", "direction"):
            mu_angle, kappa, z = self.branch(p)(torch.randn(self.B, self.D_AGG))
            assert mu_angle.shape == (self.B, self.N_CIRC)
            assert kappa.shape == (self.B, self.N_CIRC)
            assert z.shape == (self.B, 2 * self.N_CIRC)

    def test_direction_angles_are_bounded_to_one_turn(self):
        """atan2 returns [-pi, pi], so wrapping is impossible by construction.
        This is the property the whole wave is for."""
        branch = self.branch("direction")
        # A hostile input: huge activations, which is exactly what drift
        # produces and what breaks the angle head.
        mu_angle, _, _ = branch(torch.randn(self.B, self.D_AGG) * 1000.0)
        assert mu_angle.abs().max().item() <= math.pi + 1e-5

    def test_angle_parameterisation_does_wrap_on_the_same_input(self):
        """The failure being fixed, demonstrated rather than asserted."""
        branch = self.branch("angle")
        with torch.no_grad():
            branch.mu_head.weight.mul_(10.0)
        mu_angle, _, _ = branch(torch.randn(self.B, self.D_AGG) * 100.0)
        assert mu_angle.abs().max().item() > 2 * math.pi, (
            "the angle head should run past one turn on a large input — if it "
            "cannot, this test no longer demonstrates the problem"
        )

    def test_scaling_the_head_output_does_not_move_the_direction(self):
        """The property that makes drift harmless: only the ratio matters, so
        a head whose outputs grow by 100x predicts the same points."""
        branch = self.branch("direction")
        x = torch.randn(self.B, self.D_AGG)
        small, _, _ = branch(x)
        with torch.no_grad():                 # scale the whole affine map
            branch.mu_head.weight.mul_(100.0)
            branch.mu_head.bias.mul_(100.0)
        large, _, _ = branch(x)
        assert torch.allclose(small, large, atol=1e-4)

    def test_direction_head_is_differentiable_including_near_the_origin(self):
        """clamp_min guards the (0,0) pair, where atan2's gradient is otherwise
        undefined — a NaN this project has met before in the sampler."""
        branch = self.branch("direction")
        with torch.no_grad():
            branch.mu_head.weight.zero_()
            branch.mu_head.bias.zero_()       # every pair exactly (0, 0)
        mu_angle, _, _ = branch(torch.randn(self.B, self.D_AGG))
        assert torch.isfinite(mu_angle).all()
        mu_angle.sum().backward()
        assert torch.isfinite(branch.mu_head.weight.grad).all()

    def test_unknown_parameterisation_is_rejected_at_construction(self):
        """A typo in a config must fail loudly, not silently train the old
        model under a new name."""
        with pytest.raises(ValueError, match="angle.*direction|direction.*angle"):
            CircularBranch(d_agg=self.D_AGG, n_circular=self.N_CIRC,
                           parameterisation="directional")

    def test_default_is_the_pre_r8_head(self):
        assert CircularBranch(d_agg=self.D_AGG, n_circular=self.N_CIRC).parameterisation \
            == "angle"
