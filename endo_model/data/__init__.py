"""Data pipeline: vocabulary, dataset, sampler, and collate.

Torch-dependent modules (collate, schema) are not imported here at package
load time — import them directly when needed.  This allows vocabulary, sampler,
and constants to be used without torch installed (e.g. in config-only CI).
"""

from endo_model.data.constants import (
    DISEASE_STATUS_TO_INT,
    DNS_SENTINEL,
    INT_TO_DISEASE_STATUS,
    N_DISEASE_STATUSES,
    N_TISSUE_LEVELS,
)
from endo_model.data.samplers import PerStudyBatchSampler
from endo_model.data.vocabulary import GeneVocabulary, generate_synthetic_vocabulary

__all__ = [
    "DNS_SENTINEL",
    "DISEASE_STATUS_TO_INT",
    "INT_TO_DISEASE_STATUS",
    "N_DISEASE_STATUSES",
    "N_TISSUE_LEVELS",
    "GeneVocabulary",
    "generate_synthetic_vocabulary",
    "PerStudyBatchSampler",
]
