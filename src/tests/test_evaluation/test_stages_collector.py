"""The readers read the right tensor and the collector keeps rows in order.

A stub model stands in for ELAJ: its `encode_stages` returns tensors whose
values encode the batch and row they came from, so the collected matrices
can be checked row by row against the order the loader yielded them in.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from endo_model.evaluation.stages import (  # noqa: E402
    GROUP, PIPELINE, CellClsReader, ConcatReader, GroupClsReader, LatentReader,
    MuGaussReader, StageCollector, StageEmbeddings, TokenMeanReader,
    group_readers, masked_token_mean, pipeline_readers,
)
from endo_model.model.encoder_stages import EncoderStages  # noqa: E402

GROUPS = ("g0", "g1")
G_K = {"g0": 3, "g1": 2}
D = 4


def _stages(tag: float, b: int) -> EncoderStages:
    """Tensors whose every entry is `tag + row/10`, so provenance is legible."""
    rows = (tag + torch.arange(b).float() / 10).view(b, 1)
    return EncoderStages(
        group_tokens={g: rows.view(b, 1, 1).expand(b, G_K[g], D).clone()
                      for g in GROUPS},
        group_cls={g: rows.expand(b, D).clone() for g in GROUPS},
        cell_cls=rows.expand(b, 6).clone(),
        latent={"mu_gauss": rows.expand(b, 2).clone(),
                "mu_angle": torch.zeros(b, 1)},
    )


def _batch(tag: float, b: int, dns=None) -> dict:
    return {
        "tag": tag,
        "study_id": torch.zeros(b, dtype=torch.long),
        "group_dns_mask": {g: (dns[g] if dns and g in dns
                               else torch.zeros(b, G_K[g], dtype=torch.bool))
                           for g in GROUPS},
    }


class _StubModel:
    def eval(self):
        return self

    def encode_stages(self, batch):
        return _stages(batch["tag"], len(batch["study_id"]))


class TestReaders:
    def test_names_and_kinds(self):
        assert TokenMeanReader("g0").name == "tokens/g0"
        assert GroupClsReader("g0").name == "group/g0"
        assert TokenMeanReader("g0").kind == GROUP
        assert CellClsReader().kind == PIPELINE
        names = [r.name for r in pipeline_readers(list(GROUPS))]
        assert names == ["tokens/concat", "groups/concat", "cell_cls",
                         "mu_gauss", "latent"]
        assert len(group_readers(list(GROUPS))) == 2 * len(GROUPS)

    def test_masked_token_mean_ignores_dns_genes(self):
        tokens = torch.tensor([[[1.0], [2.0], [100.0]]])        # (1, 3, 1)
        dns = torch.tensor([[False, False, True]])
        assert masked_token_mean(tokens, dns).item() == pytest.approx(1.5)

    def test_masked_token_mean_with_nothing_measured_is_zero_not_nan(self):
        tokens = torch.ones(1, 2, 1)
        dns = torch.ones(1, 2, dtype=torch.bool)
        assert masked_token_mean(tokens, dns).item() == 0.0

    def test_each_reader_reads_its_stage(self):
        st, batch = _stages(0.0, 2), _batch(0.0, 2)
        assert GroupClsReader("g1").read(st, batch).shape == (2, D)
        assert TokenMeanReader("g1").read(st, batch).shape == (2, D)
        assert CellClsReader().read(st, batch).shape == (2, 6)
        assert MuGaussReader().read(st, batch).shape == (2, 2)
        latent = LatentReader().read(st, batch)
        assert latent.shape == (2, 2 + 2)                  # mu + (cos, sin)
        assert torch.allclose(latent[:, 2], torch.ones(2))  # cos 0
        assert torch.allclose(latent[:, 3], torch.zeros(2)) # sin 0

    def test_concat_widths_add(self):
        st, batch = _stages(0.0, 2), _batch(0.0, 2)
        cat = ConcatReader("x", [GroupClsReader(g) for g in GROUPS])
        assert cat.read(st, batch).shape == (2, D * len(GROUPS))

    def test_concat_needs_parts(self):
        with pytest.raises(ValueError):
            ConcatReader("x", [])


class TestStageCollector:
    def test_rows_follow_the_loader_order(self):
        loader = [_batch(10.0, 3), _batch(20.0, 2)]
        emb = StageCollector(pipeline_readers(list(GROUPS))).collect(
            _StubModel(), loader, torch.device("cpu"), order_len=5)
        assert emb.n_cells == 5
        want = np.array([10.0, 10.1, 10.2, 20.0, 20.1], dtype=np.float32)
        np.testing.assert_allclose(emb.matrices["cell_cls"][:, 0], want, rtol=1e-6)
        np.testing.assert_allclose(emb.matrices["groups/concat"][:, 0], want, rtol=1e-6)
        assert emb.width("groups/concat") == D * len(GROUPS)
        assert emb.kinds["cell_cls"] == PIPELINE

    def test_length_mismatch_raises(self):
        with pytest.raises(RuntimeError):
            StageCollector([CellClsReader()]).collect(
                _StubModel(), [_batch(0.0, 3)], torch.device("cpu"), order_len=4)

    def test_duplicate_names_refused(self):
        with pytest.raises(ValueError):
            StageCollector([CellClsReader(), CellClsReader()])

    def test_matrices_are_float32_and_contiguous(self):
        emb = StageCollector([MuGaussReader()]).collect(
            _StubModel(), [_batch(0.0, 2)], torch.device("cpu"), order_len=2)
        m = emb.matrices["mu_gauss"]
        assert m.dtype == np.float32 and m.flags["C_CONTIGUOUS"]


class TestStageEmbeddings:
    def test_concatenate_stacks_rows(self):
        a = StageEmbeddings({"s": np.zeros((2, 3), np.float32)}, {"s": PIPELINE})
        b = StageEmbeddings({"s": np.ones((3, 3), np.float32)}, {"s": PIPELINE})
        c = StageEmbeddings.concatenate([a, StageEmbeddings(), b])
        assert c.n_cells == 5 and c.kinds == {"s": PIPELINE}

    def test_concatenate_refuses_differing_stage_sets(self):
        a = StageEmbeddings({"s": np.zeros((2, 3), np.float32)}, {"s": PIPELINE})
        b = StageEmbeddings({"t": np.zeros((2, 3), np.float32)}, {"t": PIPELINE})
        with pytest.raises(ValueError):
            StageEmbeddings.concatenate([a, b])

    def test_empty(self):
        e = StageEmbeddings.concatenate([])
        assert e.n_cells == 0 and e.names == [] and e.nbytes() == 0
