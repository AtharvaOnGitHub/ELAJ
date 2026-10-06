"""Which checkpoints survive, and the two that must never be deleted.

Top-k by metric is only sound when the metric is a trusted ranking of what
you care about. Here the ranking metric is val_nb and whether it tracks
latent quality is still being measured, so destroying everything it ranked
low would lose the evidence needed to check it.

Arithmetic over epoch numbers. No torch, no filesystem.
"""

import importlib.util
import pathlib
import sys

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "endo_model" / "training" / "retention.py")
_spec = importlib.util.spec_from_file_location("_retention", _PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

RetentionPolicy = _module.RetentionPolicy


class TestTheSchedule:
    def test_every_fifth_epoch_is_kept(self):
        p = RetentionPolicy(every=5)
        entries = [(e, 1.0) for e in range(1, 26)]
        keep = p.survivors(entries)
        assert {5, 10, 15, 20, 25} <= keep

    def test_epoch_zero_is_kept(self):
        """Retained as a DIAGNOSTIC, not as a candidate.

        Epoch 0 is the most donor-blind checkpoint of every m-series arm and
        also the worst on cross-donor phase, because the two rise together at
        r = +0.86 to +0.94. One file is cheap and the start of the trajectory
        is where that is visible; see RetentionPolicy.scheduled.
        """
        assert RetentionPolicy(every=5).scheduled(0)
        assert 0 in RetentionPolicy(every=5).survivors(
            [(e, 1.0 - 0.01 * e) for e in range(12)])

    def test_every_zero_schedules_nothing(self):
        p = RetentionPolicy(every=0, keep_best=False, keep_last=False)
        assert p.survivors([(e, float(e)) for e in range(10)]) == set()


class TestTheBestAndTheLast:
    def test_the_best_survives_off_schedule(self):
        """The case that motivates the class: a good epoch at an unscheduled
        number used to be deleted the moment three better-scoring ones
        existed."""
        p = RetentionPolicy(every=5)
        entries = [(e, 1.0) for e in range(1, 12)]
        entries[6] = (7, 0.1)                    # epoch 7 is clearly best
        assert 7 in p.survivors(entries)

    def test_the_last_survives_even_when_worst(self):
        """A run killed by the wall has to be resumable and scoreable from
        where it stopped."""
        p = RetentionPolicy(every=5)
        entries = [(e, 0.1) for e in range(1, 12)]
        entries[-1] = (11, 9.9)                  # epoch 11 is clearly worst
        assert 11 in p.survivors(entries)

    def test_ties_on_the_metric_keep_the_earlier_epoch(self):
        p = RetentionPolicy(every=0, keep_last=False)
        assert p.survivors([(3, 0.5), (8, 0.5)]) == {3}


class TestTheWholeRule:
    def test_a_26_epoch_run_keeps_six_or_seven(self):
        p = RetentionPolicy(every=5)
        entries = [(e, 1.0 - e * 0.001) for e in range(1, 27)]  # best is last
        keep = p.survivors(entries)
        assert keep == {5, 10, 15, 20, 25, 26}

        entries[6] = (7, 0.0)                    # best now off schedule
        keep = p.survivors(entries)
        assert keep == {5, 7, 10, 15, 20, 25, 26}

    def test_nothing_is_kept_from_an_empty_run(self):
        assert RetentionPolicy().survivors([]) == set()

    def test_the_policy_says_what_it_does(self):
        text = RetentionPolicy(every=5).describe()
        assert "every 5 epochs" in text and "best" in text and "last" in text
