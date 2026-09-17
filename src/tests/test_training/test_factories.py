"""optimizer.name and scheduler.name are honoured, and unknown names refused."""

import pytest

torch = pytest.importorskip("torch")

from endo_model.configs.training_config import OptimizerConfig  # noqa: E402
from endo_model.training.factories import build_optimizer, build_scheduler  # noqa: E402


def _params():
    return [torch.nn.Parameter(torch.zeros(2))]


@pytest.mark.parametrize("name,cls", [("AdamW", torch.optim.AdamW),
                                      ("Adam", torch.optim.Adam),
                                      ("SGD", torch.optim.SGD)])
def test_optimizer_by_name(name, cls):
    opt = build_optimizer(_params(), OptimizerConfig(name=name, lr=0.01))
    assert isinstance(opt, cls)
    assert opt.param_groups[0]["lr"] == 0.01


def test_unknown_optimizer_is_refused():
    with pytest.raises(ValueError, match="RMSprop"):
        build_optimizer(_params(), OptimizerConfig(name="RMSprop"))


def test_scheduler_by_name():
    opt = build_optimizer(_params(), OptimizerConfig())
    sched = build_scheduler("CosineAnnealingLR", opt, 22)
    assert sched.T_max == 22
    assert build_scheduler("none", opt, 22) is None
    with pytest.raises(ValueError, match="StepLR"):
        build_scheduler("StepLR", opt, 22)
