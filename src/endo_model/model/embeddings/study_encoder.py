"""StudyEncoder: which study a cell came from, told to the decoder.

This goes to the **decoder**, never the encoder, for the reason
`platform_encoder.py` already sets out and then applies to everything except
the covariate it most obviously applies to.

THE CONTRADICTION THIS REMOVES. The decoder's context is
`d_agg + d_tissue + d_platform + d_disease`. Study is not in it. So the
negative-binomial term needs study-specific expression and the only route to it
is the latent, while DAB penalises the latent for carrying exactly that. The
encoder is handed two demands it cannot both satisfy, and it resolves them the
only way available: keep study in a form the decoder can use and the
2,827-parameter adversary cannot read.

That is what `v1_pool` did. At epoch 10 the study discriminator sat at chance,
P(true study) = 0.094 against 1/11 = 0.091, while the UMAP showed eleven
cleanly separated study islands and donor identity was at its highest of the
run. Across epochs 1 to 17 the correlation between the discriminator's loss and
donor kNN was **+0.232**: the blinder the adversary, the more donor identity
the latent held. Five waves read that as the adversary being mistuned and
walked lambda_dab from 0.2 down to zero.

Telling the decoder the study removes the reason to store it. This is the scVI
pattern, and `TissueHierarchyEncoder` and `PlatformEncoder` are already the
same shape sitting in the same context.

A LIGHTER VARIANT, NOT TAKEN HERE. DSBN already learns a per-study affine pair
in the encoder, so the decoder could be handed those parameters instead of a
fresh table. It costs nothing new and it is literally the study signal the
encoder has already isolated. It is not what this module does, because a shared
tensor makes the decoder's gradient reach back into the encoder's normalisation
and the two roles stop being separable. Worth trying as its own arm.

WHAT IT COSTS. The decoder now needs a study id at inference. That constrains
the DECODER on a genuinely unseen study; it does not touch the encoder or the
latent, which is what the paper claims and what the benchmark scores. scVI
carries the same constraint.

AND ONE CONFOUND IT SHARPENS. wang_2022 is the only study with adenomyosis and
the only one with myometrium, so a study covariate is also an adenomyosis
indicator for those 49,606 cells. OUTSTANDING.md 0.1 already rules out any
adenomyosis claim; this makes the reason sharper rather than new.
"""

import torch.nn as nn
from torch import Tensor


class StudyEncoder(nn.Module):
    """One learnable vector per study, for the decoder's context.

    A plain embedding table rather than a sum over sub-codes: unlike platform,
    which decomposes into assay, chemistry, sequencer and material, a study has
    no structure to share. Whatever a study does to expression that tissue and
    platform do not already explain is what this table has to hold.

    Args:
        n_studies: Number of studies in the corpus.
        d_study:   Output width, concatenated onto the decoder's context.
    """

    def __init__(self, n_studies: int, d_study: int) -> None:
        super().__init__()
        if n_studies < 1:
            raise ValueError(f"n_studies must be positive, got {n_studies}")
        self.n_studies = int(n_studies)
        self.d_study = int(d_study)
        self.embedding = nn.Embedding(self.n_studies, self.d_study)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, study_id: Tensor) -> Tensor:
        """Args:
            study_id: (B,) long — the study each cell came from.

        Returns:
            (B, d_study)
        """
        return self.embedding(study_id.long())
