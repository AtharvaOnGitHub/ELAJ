"""The selector in diagnose_bn_mode must see PerStudyBatchNorm.

WHY THIS TEST EXISTS. On 12 October 2026 the selector was an isinstance check
against `nn.modules.batchnorm._BatchNorm` alone. `PerStudyBatchNorm` subclasses
`nn.Module` directly, because it holds its statistics as one buffer indexed by
study rather than as N child BatchNorm1d modules. So the selector matched
nothing on m6_drain, both passes ran the identical computation, and the script
reported that 0% of the study separation came from the BatchNorm mode. A
diagnostic that silently measures nothing is worse than one that crashes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from endo_model.model.embeddings.per_study_batch_norm import PerStudyBatchNorm
from scripts.diagnose_bn_mode import normalisation_layers, set_batchnorm_training


class Wrapped(nn.Module):
    """A model shaped like ELAJ's: PerStudyBatchNorm nested under a parent."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(PerStudyBatchNorm(3, 4), nn.Linear(4, 4))


class LayerNormOnly(nn.Module):
    """Normalisation that holds no running statistics, so toggling it is a no-op."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)


class TestNormalisationLayers:

    def test_it_finds_a_nested_per_study_batch_norm(self):
        found = normalisation_layers(Wrapped())
        assert len(found) == 1
        assert isinstance(found[0], PerStudyBatchNorm)

    def test_it_still_finds_a_stock_batch_norm(self):
        model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4))
        assert len(normalisation_layers(model)) == 1

    def test_it_finds_both_kinds_at_once(self):
        model = nn.Sequential(nn.BatchNorm1d(4), PerStudyBatchNorm(3, 4))
        assert len(normalisation_layers(model)) == 2

    def test_layer_norm_is_not_counted(self):
        """It behaves identically in both modes, so counting it would mislead."""
        assert normalisation_layers(LayerNormOnly()) == []


class TestSetBatchnormTraining:

    def test_it_toggles_and_reports_the_count(self):
        model = Wrapped()
        model.eval()
        assert set_batchnorm_training(model, True) == 1
        assert model.inner[0].training is True
        assert set_batchnorm_training(model, False) == 1
        assert model.inner[0].training is False

    def test_it_leaves_other_modules_alone(self):
        """Dropout must stay off, which is why model.train() is not used."""
        model = Wrapped()
        model.eval()
        set_batchnorm_training(model, True)
        assert model.inner[1].training is False

    def test_it_refuses_a_model_with_nothing_to_toggle(self):
        """The +0.000 bug: no layers found must crash, not report no effect."""
        with pytest.raises(RuntimeError, match="nothing to toggle"):
            set_batchnorm_training(LayerNormOnly(), True)
