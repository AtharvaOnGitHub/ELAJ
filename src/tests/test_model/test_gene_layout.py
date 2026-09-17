"""Tests for the per-study measured-gene layout cache.

The layout used to be recomputed every forward pass. Caching it is only safe if
the key is right, so the tests that matter here are the ones that would catch a
*wrong* cache hit: two studies with different did-not-sample patterns must get
different layouts, and revisiting a study must not return its neighbour's.

The reference implementation below is the Python loop the cache replaced, kept
verbatim so "faster" is checked against "same".
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
from endo_model.model.gene_layout import MeasuredGeneLayoutCache  # noqa: E402

N_GENES, N_GROUPS, B = 120, 6, 3


def make_vocab() -> GeneVocabulary:
    return GeneVocabulary.from_dict(
        generate_synthetic_vocabulary(n_genes=N_GENES, n_groups=N_GROUPS, seed=24)
    )


def reference_layout(vocab: GeneVocabulary, is_dns_row):
    """The Python loop the cache replaced. Returns (measured, {name: (...)})."""
    measured = (~is_dns_row).nonzero(as_tuple=True)[0].tolist()
    position_of = {g: i for i, g in enumerate(measured)}
    groups = {}
    for seq_pos, name in enumerate(vocab.group_names):
        within, positions = [], []
        for within_i, global_i in enumerate(vocab.global_indices_for_group(name)):
            if global_i in position_of:
                within.append(within_i)
                positions.append(position_of[global_i])
        if within:
            groups[name] = (seq_pos, within, positions)
    return measured, groups


def dns_row(vocab: GeneVocabulary, drop_every: int):
    """A DNS pattern that drops every `drop_every`-th gene. 0 drops nothing."""
    row = torch.zeros(vocab.vocab_size, dtype=torch.bool)
    if drop_every:
        row[::drop_every] = True
    return row


def make_cache(vocab: GeneVocabulary) -> MeasuredGeneLayoutCache:
    return MeasuredGeneLayoutCache(
        list(vocab.group_names),
        {n: vocab.global_indices_for_group(n) for n in vocab.group_names},
    )


# --- Layout matches the loop it replaced -----------------------------------


@pytest.mark.parametrize("drop_every", [0, 2, 3, 7])
def test_layout_matches_reference(drop_every):
    vocab = make_vocab()
    row = dns_row(vocab, drop_every)
    layout = make_cache(vocab).get(0, row)
    measured, groups = reference_layout(vocab, row)

    assert layout.measured_global_idxs.tolist() == measured
    assert [s.name for s in layout.slices] == list(groups)
    for slice_ in layout.slices:
        seq_pos, within, positions = groups[slice_.name]
        assert slice_.seq_pos == seq_pos
        assert slice_.within_idxs.tolist() == within
        assert slice_.meas_positions.tolist() == positions


def test_groups_with_no_measured_genes_are_omitted():
    vocab = make_vocab()
    row = torch.zeros(vocab.vocab_size, dtype=torch.bool)
    dropped = vocab.group_names[0]
    row[vocab.global_indices_for_group(dropped)] = True

    layout = make_cache(vocab).get(0, row)
    assert dropped not in [s.name for s in layout.slices]
    assert len(layout.slices) == len(vocab.group_names) - 1


# --- The cache must key on the study, not just return the first thing -------


def test_different_studies_get_different_layouts():
    vocab = make_vocab()
    cache = make_cache(vocab)
    a = cache.get(0, dns_row(vocab, 0))
    b = cache.get(1, dns_row(vocab, 3))
    assert a.n_measured != b.n_measured
    assert a.measured_global_idxs.tolist() != b.measured_global_idxs.tolist()


def test_revisiting_a_study_returns_its_own_layout():
    """The wrong-cache-hit regression: study 0 after study 1 is still study 0."""
    vocab = make_vocab()
    cache = make_cache(vocab)
    first = cache.get(0, dns_row(vocab, 0))
    cache.get(1, dns_row(vocab, 3))
    again = cache.get(0, dns_row(vocab, 0))

    assert again is first  # served from cache, not rebuilt
    assert again.measured_global_idxs.tolist() == reference_layout(
        vocab, dns_row(vocab, 0)
    )[0]


# --- End to end through the model ------------------------------------------


def make_batch(vocab: GeneVocabulary, study_id: int, drop_every: int):
    counts = torch.randint(0, 50, (B, vocab.vocab_size)).float()
    row = dns_row(vocab, drop_every)
    is_dns = row.unsqueeze(0).expand(B, -1).contiguous()
    group_indices, group_dns_mask = {}, {}
    for name in vocab.group_names:
        idxs = vocab.global_indices_for_group(name)
        group_indices[name] = (
            torch.tensor(idxs, dtype=torch.long).unsqueeze(0).expand(B, -1)
        )
        group_dns_mask[name] = is_dns[:, idxs]
    return {
        "counts": counts,
        "is_dns": is_dns,
        "library_size": counts.sum(dim=1),
        "study_id": torch.full((B,), study_id, dtype=torch.long),
        "group_indices": group_indices,
        "group_dns_mask": group_dns_mask,
        "tissue_levels": torch.zeros(B, 4, dtype=torch.long),
        "age": torch.tensor([30.0, 45.0, float("nan")]),
        "disease_status": torch.zeros(B, dtype=torch.long),
    }


def test_alternating_studies_keep_their_own_gene_counts():
    """A stale layout would give a study the wrong number of measured genes."""
    vocab = make_vocab()
    config = ModelConfig(
        d_token=8, d_agg=16, d_gauss=8, n_circular=2, d_dec=8,
        n_groups=len(vocab.group_names), n_studies=2,
        group_encoder=GroupEncoderConfig(n_layers=1, n_heads=2, dropout=0.0),
        aggregation_encoder=AggregationEncoderConfig(n_layers=1, n_heads=2, dropout=0.0),
    )
    model = EndoFoundationModel(vocab, config).eval()

    batch_a = make_batch(vocab, study_id=0, drop_every=0)
    batch_b = make_batch(vocab, study_id=1, drop_every=3)
    expected_a = int((~batch_a["is_dns"][0]).sum())
    expected_b = int((~batch_b["is_dns"][0]).sum())
    assert expected_a != expected_b

    with torch.no_grad():
        for _ in range(3):  # exercise miss, miss, then hit on both
            assert model(batch_a)["mu_hat"].shape[1] == expected_a
            assert model(batch_b)["mu_hat"].shape[1] == expected_b
