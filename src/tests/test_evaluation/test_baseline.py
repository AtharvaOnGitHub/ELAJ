"""The baseline has to be beatable, comparable, and honest about DNS genes.

A baseline that is accidentally too weak makes any model look good, and one
computed a slightly different way from the model's own loss is not a comparison
at all. Both failures are silent, so they are what these tests check.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.evaluation.baseline import MeanProfileBaseline  # noqa: E402

D, N_STUDIES, B = 40, 2, 8


def make_batch(study: int, counts: torch.Tensor, dns: torch.Tensor | None = None):
    is_dns = (
        dns if dns is not None else torch.zeros(D, dtype=torch.bool)
    ).unsqueeze(0).expand(counts.shape[0], -1).contiguous()
    counts = counts.clone()
    counts[:, is_dns[0]] = -1.0            # the sentinel the pipeline writes
    return {
        "counts": counts,
        "is_dns": is_dns,
        "library_size": counts.clamp_min(0).sum(dim=1),
        "study_id": torch.full((counts.shape[0],), study, dtype=torch.long),
    }


def study_profile(study: int) -> torch.Tensor:
    """A study's expression profile — distinct in *shape*, not merely scale.

    Two studies differing by a constant factor have *identical* proportion
    profiles, because dividing by the library size removes the factor. Building
    the fixture that way makes a contamination test vacuous: it would compare
    two things that are correctly equal. Reversing the profile makes them
    genuinely different.
    """
    base = torch.arange(1, D + 1).float()
    return base if study == 0 else base.flip(0)


def constant_corpus(seed: int = 0, dns: torch.Tensor | None = None) -> list:
    """Two studies with distinct, cell-invariant expression profiles."""
    torch.manual_seed(seed)
    batches = []
    for study in range(N_STUDIES):
        counts = study_profile(study).unsqueeze(0).expand(B, -1).contiguous()
        batches.append(make_batch(study, counts, dns))
    return batches


# --- Fitting ----------------------------------------------------------------


def test_profile_recovers_a_constant_corpus():
    """If every cell in a study is identical, the profile is that cell."""
    batches = constant_corpus()
    base = MeanProfileBaseline(N_STUDIES, D, n_hvg=5).fit(batches)
    for study in range(N_STUDIES):
        expected = study_profile(study)
        expected = expected / expected.sum()
        got = base._proportions[study]
        assert torch.allclose(got, expected, atol=1e-5), f"study {study}"


def test_studies_do_not_contaminate_each_other():
    """Distinct study profiles must stay distinct.

    Note the profiles differ in shape, not scale: proportions are
    scale-invariant, so a fixture whose studies differ by a constant factor
    would compare two correctly-equal vectors and prove nothing.
    """
    base = MeanProfileBaseline(N_STUDIES, D, n_hvg=5).fit(constant_corpus())
    assert not torch.allclose(base._proportions[0], base._proportions[1])


def test_dns_genes_are_excluded_not_counted_as_zero():
    """DNS holds -1, not 0. Summing it blind would make proportions negative.

    Silent if unchecked: a negative proportion becomes a negative predicted
    mean, and the NB loss on that is nonsense rather than an error.
    """
    dns = torch.zeros(D, dtype=torch.bool)
    dns[:10] = True
    base = MeanProfileBaseline(N_STUDIES, D, n_hvg=5).fit(constant_corpus(dns=dns))
    assert (base._proportions >= 0).all(), "a DNS sentinel leaked into the profile"
    assert (base._proportions[:, :10] == 0).all(), "DNS genes should hold no mass"


def test_theta_is_positive_and_finite():
    """Method of moments divides by (var - mean), which is <= 0 for a constant
    corpus — the degenerate case must yield a large theta, not a negative one."""
    base = MeanProfileBaseline(N_STUDIES, D, n_hvg=5).fit(constant_corpus())
    assert torch.isfinite(base._theta).all()
    assert (base._theta > 0).all()


# --- Scoring ----------------------------------------------------------------


def test_score_is_finite_and_reports_cells():
    batches = constant_corpus()
    base = MeanProfileBaseline(N_STUDIES, D, n_hvg=5).fit(batches)
    scores = base.score(batches)
    assert scores.n_cells_scored == B * N_STUDIES
    assert scores.nb_all_genes == pytest.approx(scores.nb_all_genes)  # not NaN
    assert scores.n_hvg == 5


def test_a_variable_corpus_scores_worse_than_a_constant_one():
    """The baseline predicts one profile per study, so it must do better on a
    study whose cells are identical than on one whose cells are not. If this
    fails the baseline is not actually using the profile."""
    torch.manual_seed(0)
    constant = constant_corpus()

    variable = []
    for study in range(N_STUDIES):
        profile = study_profile(study)
        noisy = (
            profile.unsqueeze(0) * torch.rand(B, D).mul(4.0)
        ).round().clamp_min(0.0)
        variable.append(make_batch(study, noisy))

    base_c = MeanProfileBaseline(N_STUDIES, D, n_hvg=5).fit(constant)
    base_v = MeanProfileBaseline(N_STUDIES, D, n_hvg=5).fit(variable)
    assert base_c.score(constant).nb_all_genes < base_v.score(variable).nb_all_genes


def test_hvg_mask_size_and_measured_restriction():
    dns = torch.zeros(D, dtype=torch.bool)
    dns[:10] = True
    # Genes 0-9 are missing from *every* study here, so they cannot be ranked.
    base = MeanProfileBaseline(N_STUDIES, D, n_hvg=5).fit(constant_corpus(dns=dns))
    mask = base.hvg_mask
    assert int(mask.sum()) <= 5
    assert not mask[:10].any(), "a gene no study measured was called highly variable"


# --- HVG metric -------------------------------------------------------------


def test_hvg_metric_matches_the_baseline_gene_set():
    """The model must be scored on exactly the genes the baseline was.

    Two curves on the same axes computed over different gene sets would be a
    comparison in appearance only, and nothing in the figure would say so.
    """
    from endo_model.evaluation.metrics import HvgReconstructionMetric

    batches = constant_corpus()
    base = MeanProfileBaseline(N_STUDIES, D, n_hvg=5).fit(batches)
    metric = HvgReconstructionMetric(base.hvg_mask)

    batch = batches[0]
    measured = (~batch["is_dns"][0]).nonzero(as_tuple=True)[0]
    output = {
        "measured_global_idxs": measured,
        "mu_hat": batch["counts"][:, measured].clamp_min(0.1).float(),
        "theta": torch.full((measured.numel(),), 5.0),
    }
    result = metric(output, batch)
    assert "nb_hvg" in result
    assert torch.isfinite(torch.tensor(result["nb_hvg"]))


def test_hvg_metric_returns_empty_when_no_hvgs_measured():
    """Empty, not zero: a zero would pull the epoch average down and read as
    excellent reconstruction from a study that contributed nothing."""
    from endo_model.evaluation.metrics import HvgReconstructionMetric

    batches = constant_corpus()
    base = MeanProfileBaseline(N_STUDIES, D, n_hvg=5).fit(batches)
    metric = HvgReconstructionMetric(base.hvg_mask)

    # A study measuring only genes the HVG set excludes.
    excluded = (~base.hvg_mask).nonzero(as_tuple=True)[0][:3]
    batch = batches[0]
    output = {
        "measured_global_idxs": excluded,
        "mu_hat": torch.ones(B, excluded.numel()),
        "theta": torch.full((excluded.numel(),), 5.0),
    }
    assert metric(output, batch) == {}
