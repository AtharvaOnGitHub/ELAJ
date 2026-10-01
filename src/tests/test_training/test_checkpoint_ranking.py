"""The rule that picks the model, pinned before the run that uses it.

This is the test that matters most in the wave: with menstrual phase as the
target, the checkpoint the ranking chooses IS the result. If the guard can
be bypassed, the run reports a model whose reconstruction had already gone
past the mean-profile baseline, and nothing downstream would say so.

No torch here. It is arithmetic over a metrics dict, and it should stay
runnable on a machine with no GPU so the rule can be checked anywhere.
"""

import importlib.util
import math
import pathlib
import sys

# Loaded by path, not by `from endo_model.training...`: that package's
# __init__ imports CheckpointManager, which imports torch, and this rule is
# arithmetic that must stay checkable on a machine without it. The module
# itself has no torch dependency, which is part of what is being pinned.
_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "endo_model" / "training" / "checkpoint_ranking.py")
_spec = importlib.util.spec_from_file_location("_checkpoint_ranking", _PATH)
_module = importlib.util.module_from_spec(_spec)
# Registered before exec: the module uses `from __future__ import
# annotations`, and @dataclass resolves those string annotations through
# sys.modules[cls.__module__]. Without this line that lookup returns None.
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

BREACH_PENALTY = _module.BREACH_PENALTY
CheckpointRanking = _module.CheckpointRanking

BASELINE_NB = 0.4100


def metrics(nb: float, phase: float | None = None) -> dict:
    out = {"nb": nb}
    if phase is not None:
        out["phase_knn_xdonor"] = phase
    return out


class TestTheDefaultIsUnchanged:
    def test_plain_nb_ranking_minimises_nb(self):
        rank = CheckpointRanking()
        assert rank.rank(metrics(0.38), 9.9) == 0.38
        assert rank.rank(metrics(0.41), 9.9) > rank.rank(metrics(0.38), 9.9)

    def test_val_loss_is_the_fallback_when_nb_is_absent(self):
        assert CheckpointRanking().rank({}, 0.77) == 0.77


class TestAnAgreementMetricIsNegated:
    def test_higher_agreement_ranks_better(self):
        rank = CheckpointRanking(metric="phase_knn_xdonor")
        better = rank.rank(metrics(0.38, 0.72), 9.9)
        worse = rank.rank(metrics(0.38, 0.65), 9.9)
        assert better < worse
        assert better == -0.72

    def test_an_unscored_selector_falls_back_to_nb(self):
        """A probe can be absent for reasons that say nothing about the
        model. Ranking that checkpoint last would discard a good epoch."""
        rank = CheckpointRanking(metric="phase_knn_xdonor")
        assert rank.rank(metrics(0.39), 9.9) == 0.39

    def test_a_nan_selector_falls_back_to_nb(self):
        rank = CheckpointRanking(metric="phase_knn_xdonor")
        assert rank.rank(metrics(0.39, float("nan")), 9.9) == 0.39


class TestTheGuard:
    def test_a_breach_ranks_behind_every_clean_checkpoint(self):
        """The case this class exists for: a checkpoint with the best phase
        score in the run and reconstruction past baseline must not win."""
        rank = CheckpointRanking(metric="phase_knn_xdonor",
                                 guard="nb", guard_max=BASELINE_NB)
        spectacular_but_broken = rank.rank(metrics(0.4600, 0.81), 9.9)
        modest_but_sound = rank.rank(metrics(0.3900, 0.68), 9.9)
        assert modest_but_sound < spectacular_but_broken

    def test_equal_to_the_ceiling_is_not_a_breach(self):
        rank = CheckpointRanking(guard="nb", guard_max=BASELINE_NB)
        assert not rank.breached(metrics(BASELINE_NB))
        assert rank.breached(metrics(BASELINE_NB + 1e-9))

    def test_breached_checkpoints_still_order_among_themselves(self):
        """A run that breaches everywhere must still hand back its least bad
        checkpoint rather than an arbitrary one."""
        rank = CheckpointRanking(metric="phase_knn_xdonor",
                                 guard="nb", guard_max=BASELINE_NB)
        a = rank.rank(metrics(0.48, 0.70), 9.9)
        b = rank.rank(metrics(0.48, 0.60), 9.9)
        assert a < b
        assert a > BREACH_PENALTY - 1.0

    def test_an_unmeasured_guard_is_not_a_breach(self):
        """The guard rejects what we can see is bad, not what we could not
        measure. Treating absence as a breach would penalise every epoch of
        a run whose probe is misconfigured."""
        rank = CheckpointRanking(guard="nb", guard_max=0.1)
        assert not rank.breached({})
        assert not rank.breached({"nb": float("nan")})

    def test_no_guard_by_default(self):
        rank = CheckpointRanking(metric="phase_knn_xdonor")
        assert rank.guard is None
        assert rank.guard_max == math.inf
        assert not rank.breached(metrics(99.0, 0.5))


class TestItSaysWhatItIsDoing:
    def test_the_description_names_the_rule(self):
        text = CheckpointRanking(metric="phase_knn_xdonor", guard="nb",
                                 guard_max=BASELINE_NB).describe()
        assert "phase_knn_xdonor" in text and "maximise" in text
        assert "nb" in text and "0.4100" in text

    def test_the_default_description_says_no_guard(self):
        assert "no guard" in CheckpointRanking().describe()
