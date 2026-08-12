"""The decoder must reach a cell only through the latent.

Run 49817807 trained for 11 epochs and learned nothing after the first, because
`gene_reps` was built from the encoder's per-cell group summaries. The decoder
reconstructed from those and ignored the latent, so both latent branches
collapsed — `kl_vmf` reached 5.2e-7, meaning zero information.

Nothing in that failure raised an error, and the reconstruction loss looked
plausible throughout. It was only visible by reading the KL terms. So these
tests assert the property structurally rather than by watching a loss:

    the gene side of the decoder has no batch dimension,

which makes it impossible for per-cell information to arrive by that route.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.configs.model_config import (  # noqa: E402
    AggregationEncoderConfig,
    GroupEncoderConfig,
    ModelConfig,
)
from endo_model.data.vocabulary import (  # noqa: E402
    GeneVocabulary,
    generate_synthetic_vocabulary,
)
from endo_model.model.endo_model import EndoFoundationModel  # noqa: E402

N_GENES, N_GROUPS, B = 90, 5, 4
D_TOKEN, D_AGG, D_DEC = 8, 16, 12


def make_model():
    vocab = GeneVocabulary.from_dict(
        generate_synthetic_vocabulary(n_genes=N_GENES, n_groups=N_GROUPS, seed=7)
    )
    config = ModelConfig(
        d_token=D_TOKEN, d_agg=D_AGG, d_gauss=8, n_vMF=2, d_dec=D_DEC,
        n_groups=len(vocab.group_names), n_studies=2,
        group_encoder=GroupEncoderConfig(n_layers=1, n_heads=2, dropout=0.0),
        aggregation_encoder=AggregationEncoderConfig(
            n_layers=1, n_heads=2, dropout=0.0
        ),
    )
    return vocab, EndoFoundationModel(vocab, config).eval()


def make_batch(vocab, counts=None, study_id=0):
    vocab_size = vocab.vocab_size
    if counts is None:
        counts = torch.randint(0, 50, (B, vocab_size)).float()
    is_dns = torch.zeros(B, vocab_size, dtype=torch.bool)
    group_indices, group_dns_mask = {}, {}
    for name in vocab.group_names:
        idxs = vocab.global_indices_for_group(name)
        group_indices[name] = (
            torch.tensor(idxs, dtype=torch.long).unsqueeze(0).expand(B, -1)
        )
        group_dns_mask[name] = torch.zeros(B, len(idxs), dtype=torch.bool)
    return {
        "counts": counts,
        "is_dns": is_dns,
        "library_size": counts.sum(dim=1),
        "study_id": torch.full((B,), study_id, dtype=torch.long),
        "group_indices": group_indices,
        "group_dns_mask": group_dns_mask,
        "tissue_levels": torch.zeros(B, 4, dtype=torch.long),
        "age": torch.full((B,), 40.0),
        "disease_status": torch.zeros(B, dtype=torch.long),
    }


# --- The structural guarantee ----------------------------------------------


def test_gene_queries_have_no_batch_dimension():
    """(G_meas, d_dec), not (B, G_meas, ...). This is the whole invariant."""
    vocab, model = make_model()
    with torch.no_grad():
        queries, measured = model._build_gene_queries(make_batch(vocab))
    assert queries.dim() == 2, f"expected (G_meas, d_dec), got {tuple(queries.shape)}"
    assert queries.shape == (measured.numel(), D_DEC)


def test_gene_queries_ignore_the_counts():
    """Two batches, same study, completely different expression."""
    vocab, model = make_model()
    quiet = make_batch(vocab, counts=torch.ones(B, vocab.vocab_size))
    loud = make_batch(vocab, counts=torch.full((B, vocab.vocab_size), 500.0))
    with torch.no_grad():
        a, _ = model._build_gene_queries(quiet)
        b, _ = model._build_gene_queries(loud)
    assert torch.equal(a, b), "gene queries changed with the counts — a bypass exists"


def test_gene_queries_take_no_gradient_from_the_encoder():
    """No path from the encoders to the gene side of the decoder.

    Catches a bypass reintroduced through any tensor, not just the one that
    caused the original failure: if the encoders could influence the queries,
    they would receive gradient from them.
    """
    vocab, model = make_model()
    queries, _ = model._build_gene_queries(make_batch(vocab))
    queries.sum().backward()

    for name, param in model.named_parameters():
        if name.startswith(("hierarchical_encoder", "aggregation_encoder",
                            "value_encoder", "metadata_encoder")):
            assert param.grad is None or not param.grad.any(), (
                f"{name} received gradient from the decoder's gene side"
            )


# --- The latent must actually be used ---------------------------------------


def test_reconstruction_depends_on_the_latent():
    """With the bypass closed, z is the only route — so perturbing it must move
    the prediction. If this fails, the decoder is ignoring the latent."""
    vocab, model = make_model()
    batch = make_batch(vocab)
    with torch.no_grad():
        queries, _ = model._build_gene_queries(batch)
        z = model.latent_space(
            model.aggregation_encoder(
                model.hierarchical_encoder(model.token_constructor(batch)),
                model.metadata_encoder(batch["age"]),
            )
        )["z_proj"]
        # Tissue is held fixed across both calls: the decoder is conditioned on
        # it, so varying it too would leave the test unable to say which input
        # moved the output.
        tissue = model.decoder_tissue(batch["tissue_levels"])
        base = model.decoder(queries, z, tissue)
        moved = model.decoder(queries, z + 1.0, tissue)
    assert not torch.allclose(base, moved), "decoder output is independent of z"


def test_different_cells_get_different_predictions():
    """The only way this can hold now is via the latent."""
    vocab, model = make_model()
    counts = torch.randint(0, 50, (B, vocab.vocab_size)).float()
    counts[0] = 1.0
    counts[1] = 400.0
    with torch.no_grad():
        mu = model(make_batch(vocab, counts=counts))["mu_hat"]
    fractions = mu / mu.sum(dim=-1, keepdim=True)
    assert not torch.allclose(fractions[0], fractions[1], atol=1e-6), (
        "two very different cells got the same predicted composition"
    )
