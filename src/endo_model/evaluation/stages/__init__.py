"""Stage probe: where along the encoder is the biology lost?

    readers     one StageReader per point in the encoder (five pipeline
                points, and input/output of each of the 45 group encoders)
    collector   StageCollector runs the encoder and keeps every reader's
                matrix, aligned to the sampler order
    references  GroupPCAReference: per-group PCA of the counts, raw and
                log-normalised, the linear reading of the encoder's input
    probe       StageProbe scores every matrix with the benchmark scorer
    report      StageTables arranges the rows into the two tables a person
                reads
"""

from endo_model.evaluation.stages.collector import StageCollector, StageEmbeddings
from endo_model.evaluation.stages.probe import ProbeSettings, StageProbe, group_battery
from endo_model.evaluation.stages.readers import (
    GROUP, PIPELINE, CellClsReader, ConcatReader, GroupClsReader, LatentReader,
    MuGaussReader, StageReader, TokenMeanReader, group_readers,
    masked_token_mean, pipeline_readers,
)
from endo_model.evaluation.stages.references import (
    GroupPCAReference, group_masks_from_vocabulary,
)
from endo_model.evaluation.stages.report import StageTables

__all__ = [
    "GROUP", "PIPELINE",
    "CellClsReader", "ConcatReader", "GroupClsReader", "LatentReader",
    "MuGaussReader", "StageReader", "TokenMeanReader",
    "group_readers", "masked_token_mean", "pipeline_readers",
    "StageCollector", "StageEmbeddings",
    "ProbeSettings", "StageProbe", "group_battery",
    "StageTables",
    "GroupPCAReference", "group_masks_from_vocabulary",
]
