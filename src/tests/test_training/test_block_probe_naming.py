"""Each latent block's donor metrics must be their own series.

`patient_knn` and `study_knn` describe the LATENT, not the probe's label, so
every probe computes them under the same two names. Merging three probes'
dictionaries left whichever ran last. In m1 that was the circular-block probe
on even epochs and the full-latent probe on odd ones, so one column carried
two interleaved series: 0.23, 0.53, 0.27, 0.56 for sixteen epochs. Nothing
raised, and the column looked like an oscillating measurement.

`named` is pure dictionary work, so this needs no torch and runs anywhere.
"""

import importlib.util
import pathlib
import sys

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "endo_model" / "training" / "label_probe.py")
_spec = importlib.util.spec_from_file_location("_label_probe", _PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

BlockProbe = _module.BlockProbe
IDENTITY_METRICS = _module.IDENTITY_METRICS


class TestIdentityMetricsAreNamespaced:
    def test_patient_and_study_take_the_probe_s_name(self):
        out = BlockProbe.named(
            {"patient_knn": 0.23, "study_knn": 0.60}, "phase_knn_circ")
        assert out == {"phase_knn_circ_patient_knn": 0.23,
                       "phase_knn_circ_study_knn": 0.60}

    def test_the_label_metrics_keep_their_names(self):
        """Only the latent-level metrics collide. Prefixing the label ones
        too would rename every column the configs select on."""
        out = BlockProbe.named(
            {"phase_knn_circ": 0.70, "phase_knn_circ_xdonor": 0.65,
             "patient_knn": 0.23}, "phase_knn_circ")
        assert "phase_knn_circ" in out
        assert "phase_knn_circ_xdonor" in out
        assert out["phase_knn_circ_patient_knn"] == 0.23

    def test_three_blocks_no_longer_overwrite_each_other(self):
        """The m1 failure, reproduced and then prevented."""
        merged = {}
        for probe, value in (("phase_knn", 0.53),
                             ("phase_knn_gauss", 0.54),
                             ("phase_knn_circ", 0.23)):
            merged.update(BlockProbe.named({"patient_knn": value}, probe))
        assert len(merged) == 3
        assert merged["phase_knn_patient_knn"] == 0.53
        assert merged["phase_knn_circ_patient_knn"] == 0.23

    def test_without_naming_they_collapse_to_one(self):
        """The guard is only worth having if the thing it guards is real."""
        merged = {}
        for value in (0.53, 0.54, 0.23):
            merged.update({"patient_knn": value})
        assert merged == {"patient_knn": 0.23}, "last writer wins, as in m1"

    def test_the_identity_list_is_what_it_claims(self):
        assert IDENTITY_METRICS == ("patient_knn", "study_knn")
