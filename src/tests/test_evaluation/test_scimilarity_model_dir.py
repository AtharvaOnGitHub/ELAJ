"""Finding SCimilarity's model directory, which their archive nests.

WHY. The Zenodo archive extracts into a subdirectory whose name is not
documented and differs between records: the 116 MB encoder archive from record
15729925 unpacks to `scimilarity/`. Pointing CellEmbedding at the parent fails
with FileNotFoundError on gene_order.tsv, which is how a GPU job was spent on
15 October 2026.
"""

from __future__ import annotations

import pytest

from endo_model.evaluation.baselines.pretrained.scimilarity import (
    ScimilarityEmbedder,
)

LANDMARK = ScimilarityEmbedder.LANDMARK


class TestResolveModelDir:

    def test_the_landmark_at_the_root_resolves_to_the_root(self, tmp_path):
        (tmp_path / LANDMARK).write_text("A\tB\n")
        assert ScimilarityEmbedder.resolve_model_dir(tmp_path) == tmp_path

    def test_it_descends_one_level(self, tmp_path):
        """The real layout: the archive unpacks to scimilarity/."""
        nested = tmp_path / "scimilarity"
        nested.mkdir()
        (nested / LANDMARK).write_text("A\tB\n")
        assert ScimilarityEmbedder.resolve_model_dir(tmp_path) == nested

    def test_it_descends_two_levels(self, tmp_path):
        nested = tmp_path / "model_v1.1" / "scimilarity"
        nested.mkdir(parents=True)
        (nested / LANDMARK).write_text("A\tB\n")
        assert ScimilarityEmbedder.resolve_model_dir(tmp_path) == nested

    def test_a_missing_landmark_names_the_archive_to_use(self, tmp_path):
        (tmp_path / "README.md").write_text("nothing useful")
        with pytest.raises(FileNotFoundError, match="15729925"):
            ScimilarityEmbedder.resolve_model_dir(tmp_path)

    def test_a_missing_landmark_lists_what_is_there(self, tmp_path):
        (tmp_path / "wrong_file.bin").write_text("x")
        with pytest.raises(FileNotFoundError, match="wrong_file.bin"):
            ScimilarityEmbedder.resolve_model_dir(tmp_path)

    def test_two_candidates_are_refused_rather_than_guessed(self, tmp_path):
        for name in ("a", "b"):
            sub = tmp_path / name
            sub.mkdir()
            (sub / LANDMARK).write_text("A\tB\n")
        with pytest.raises(ValueError, match="ambiguous"):
            ScimilarityEmbedder.resolve_model_dir(tmp_path)

    def test_a_missing_directory_points_at_the_installer(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="install_foundation"):
            ScimilarityEmbedder.resolve_model_dir(tmp_path / "absent")
