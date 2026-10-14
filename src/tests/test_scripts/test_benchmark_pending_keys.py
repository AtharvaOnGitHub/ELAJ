"""pending_keys: what a benchmark run will actually compute.

WHY THIS IS TESTED. --fit-split builds a 1,161,472-cell CorpusMatrix, and on
12 October 2026 a job built it alongside --reuse-baselines supplying every
requested row, reached 111.9 GB of 128G, and was OOM-killed over a corpus no
method was going to read. These keys are what decides whether it is built.
"""

from __future__ import annotations

from argparse import Namespace

from scripts.benchmark_baselines import pending_keys


def args_for(methods, policies=("all",), fit_split="train"):
    return Namespace(methods=list(methods), gene_policies=list(policies),
                     fit_split=fit_split)


class FakeAxis:
    """A WideGeneAxis stand-in: only `tag` is consulted."""

    def __init__(self, n_genes: int) -> None:
        self.n_genes = n_genes

    def tag(self, policy_name: str) -> str:
        return f"{policy_name}{self.n_genes}"


class TestPendingKeys:

    def test_nothing_reused_means_everything_pending(self):
        keys = pending_keys(args_for(["pca", "scvi"]), None, set())
        assert keys == ["pca/all/train", "scvi/all/train"]

    def test_a_reused_row_is_not_pending(self):
        keys = pending_keys(args_for(["pca", "scvi"]), None, {"pca/all/train"})
        assert keys == ["scvi/all/train"]

    def test_everything_reused_means_nothing_pending(self):
        """The OOM case: the fit corpus must not be built for this."""
        already = {"pca/all/train", "scvi/all/train", "scanvi/all/train"}
        assert pending_keys(args_for(["pca", "scvi", "scanvi"]), None,
                            already) == []

    def test_a_wide_axis_changes_the_keys_so_narrow_rows_do_not_match(self):
        """A 15,496-gene reuse file must not satisfy a 30,962-gene request."""
        already = {"pca/all15496/train"}
        keys = pending_keys(args_for(["pca"]), FakeAxis(30962), already)
        assert keys == ["pca/all30962/train"]

    def test_a_wide_axis_row_is_reused_when_the_width_matches(self):
        already = {"pca/all30962/train"}
        assert pending_keys(args_for(["pca"]), FakeAxis(30962), already) == []

    def test_transductive_keys_carry_no_fit_tag(self):
        keys = pending_keys(args_for(["pca"], fit_split=None), None, set())
        assert keys == ["pca/all"]

    def test_an_inductive_reuse_file_does_not_satisfy_a_transductive_request(self):
        """The two modes are different rows and must not be confused."""
        keys = pending_keys(args_for(["pca"], fit_split=None), None,
                            {"pca/all/train"})
        assert keys == ["pca/all"]

    def test_every_policy_is_crossed_with_every_method(self):
        keys = pending_keys(args_for(["pca", "scvi"], ("all", "hvg2000")),
                            None, set())
        assert keys == ["pca/all/train", "scvi/all/train",
                        "pca/hvg2000/train", "scvi/hvg2000/train"]


class TestNoMethods:
    """`--methods none`: the reuse-plus-external pass, with nothing to fit.

    WHY IT EXISTS. A pass that only scores an external latent beside a reuse
    file used to have to NAME the methods it was not going to run, so that
    pending_keys could tick them off against the file. On 14 October 2026 the
    row tags did not match the file -- the command omitted
    --baseline-vocabulary, so keys came out `pca/all/train` against the file's
    `pca/all30962/train` -- all three went pending, the 1,162,786-cell fit
    corpus was built for nobody, and the job was OOM-killed.
    """

    def test_none_means_nothing_is_pending(self):
        assert pending_keys(args_for(["none"]), None, set()) == []

    def test_none_is_case_insensitive_and_stripped(self):
        assert pending_keys(args_for([" NONE "]), None, set()) == []

    def test_nothing_is_pending_even_with_several_policies(self):
        assert pending_keys(args_for(["none"], ("all", "hvg2000")),
                            None, set()) == []

    def test_nothing_is_pending_even_with_a_wide_axis(self):
        """The tag cannot mismatch when no method is named."""
        assert pending_keys(args_for(["none"]), FakeAxis(30962), set()) == []

    def test_a_method_literally_called_none_is_not_special_cased_away(self):
        """Only the SOLE value 'none' is the sentinel, so a list containing it
        beside real methods is a user error rather than a silent no-op."""
        keys = pending_keys(args_for(["none", "pca"]), None, set())
        assert keys == ["none/all/train", "pca/all/train"]
