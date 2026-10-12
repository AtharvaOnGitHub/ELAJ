"""build_gene_axis: a vocabulary that is only a gene axis and drops nothing.

WHY IT EXISTS. A wide vocabulary built through the pathway pipeline with
--drop-unassigned off still came out at 15,496 of 30,962 genes, because
UnionCutStrategy discards any gene with no Reactome or GO hierarchy entry.
The one property these tests are for is that nothing here drops a gene.
"""

from __future__ import annotations

import pytest

from endo_model.data.vocabulary import GeneVocabulary
from scripts.build_gene_axis import axis_vocabulary, read_gene_list


class TestReadGeneList:

    def test_it_reads_one_symbol_per_line(self, tmp_path):
        path = tmp_path / "genes.txt"
        path.write_text("VEGFA\nESR1\nPGR\n")
        assert read_gene_list(path) == ["VEGFA", "ESR1", "PGR"]

    def test_blank_lines_and_whitespace_are_ignored(self, tmp_path):
        path = tmp_path / "genes.txt"
        path.write_text("VEGFA\n\n  ESR1  \n\n")
        assert read_gene_list(path) == ["VEGFA", "ESR1"]

    def test_duplicates_are_removed_keeping_first_position(self, tmp_path):
        path = tmp_path / "genes.txt"
        path.write_text("VEGFA\nESR1\nVEGFA\n")
        assert read_gene_list(path) == ["VEGFA", "ESR1"]

    def test_order_is_preserved_because_it_becomes_column_order(self, tmp_path):
        path = tmp_path / "genes.txt"
        path.write_text("ZZZ3\nAAAS\nMID1\n")
        assert read_gene_list(path) == ["ZZZ3", "AAAS", "MID1"]

    def test_an_empty_file_is_refused(self, tmp_path):
        path = tmp_path / "genes.txt"
        path.write_text("\n\n")
        with pytest.raises(ValueError, match="no gene symbols"):
            read_gene_list(path)


class TestAxisVocabulary:

    def test_every_gene_survives(self):
        """The whole point: no gene is dropped for lacking an annotation."""
        genes = [f"G{i}" for i in range(1234)]
        assert len(axis_vocabulary(genes)) == 1234

    def test_global_idx_is_the_position_in_the_gene_list(self):
        vocabulary = axis_vocabulary(["A", "B", "C"])
        assert [vocabulary[g]["global_idx"] for g in ("A", "B", "C")] == [0, 1, 2]

    def test_groups_are_chunks_of_the_requested_size(self):
        vocabulary = axis_vocabulary([f"G{i}" for i in range(7)], chunk_size=3)
        assert vocabulary["G0"]["group"] == "axis_0000"
        assert vocabulary["G2"]["group"] == "axis_0000"
        assert vocabulary["G3"]["group"] == "axis_0001"
        assert vocabulary["G6"]["group"] == "axis_0002"

    def test_group_idx_is_the_position_within_the_chunk(self):
        vocabulary = axis_vocabulary([f"G{i}" for i in range(7)], chunk_size=3)
        assert vocabulary["G4"]["group_idx"] == 1

    def test_chromosome_is_left_blank_rather_than_invented(self):
        assert axis_vocabulary(["A"])["A"]["chromosome"] == ""

    def test_a_non_positive_chunk_size_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            axis_vocabulary(["A"], chunk_size=0)

    def test_gene_vocabulary_loads_it_and_round_trips_the_axis(self):
        """It has to satisfy the consumer, not just look right as a dict."""
        genes = [f"G{i}" for i in range(1050)]
        loaded = GeneVocabulary.from_dict(axis_vocabulary(genes, chunk_size=500))
        assert loaded.vocab_size == 1050
        assert len(loaded.group_names) == 3
        assert [loaded.index_to_gene(i) for i in range(3)] == ["G0", "G1", "G2"]
        assert loaded.gene_to_index("G1049") == 1049
