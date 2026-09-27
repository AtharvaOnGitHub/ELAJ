"""No metadata reaches the encoder. This module records why, and stays empty.

Three fields were considered for the aggregation encoder's input. All three are
now out, and each removal changes what the trained model can be used to claim.

**disease_status.** The project's output is a disease atlas, and its headline
result is whether the latent organises cells by disease.  If the encoder is
*told* which cells are diseased it will separate them, and that separation is
evidence of nothing — the answer was an input.  Disease status is held out of
the model entirely and used only to colour the UMAP, which is the only
arrangement in which the picture supports a claim.

**tissue.** Moved to the decoder rather than dropped; see
decoder/bilinear_decoder.py.  Conditioning a decoder on a known covariate is
standard, and it changes what the latent represents: the latent no longer has
to spend capacity encoding "this cell came from myometrium", because the
decoder is told separately.  What remains in the latent is the variation tissue
does not explain.

**age.** Removed 27 September 2026, and it was a defect rather than a design
choice. Every study in the corpus reports age for **zero** cells: the column is
either absent or present and entirely empty in all twelve. The one study that
carries real ages, zhang_2022 with 345,469 cells, is excluded from the corpus.
So the age token was a learned constant plus a NaN path, occupying a position
in every cell's aggregation sequence and contributing nothing to any of them.

Two reasons removing it is worth the churn rather than leaving it inert. It sat
on the *encoder* side, and nothing about a donor's age belongs there: metadata
that is known and not to be discovered conditions the decoder, which is where
tissue and platform already are. And an input that is always missing is a
standing invitation to misread the architecture — the shape map showed a
metadata token feeding the cell-level transformer, which is a claim about the
model that was not true in any run.

N_METADATA is kept at 0 so that a caller which still counts metadata positions
gets the right answer rather than an AttributeError.
"""

N_METADATA = 0
