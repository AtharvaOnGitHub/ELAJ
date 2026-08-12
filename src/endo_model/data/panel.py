"""StudyPanel: sparse per-study count storage, densified one cell at a time.

Why this exists
---------------
A cell's model input is a dense ``(vocab_size,)`` vector in which genes outside
the study's gene panel carry ``DNS_SENTINEL``.  Materialising that vector for
every cell at load time costs ``vocab_size * 4`` bytes per cell — roughly 76 KB
at a 19 K vocabulary, or ~90 GB over a million-cell corpus.

Single-cell count matrices are ~90 % zeros, so the same information fits in a
CSR matrix at roughly a sixth of the cost.  ``StudyPanel`` holds that CSR form
and rebuilds the dense vector on demand in ``counts()``, which the Dataset calls
from ``__getitem__`` — i.e. once per cell per epoch, inside the DataLoader
worker, off the critical path.

This changes *storage only*.  The vector handed to the collate function is
byte-for-byte what the eager implementation produced, so nothing downstream —
collate, the model, the losses — is aware of the difference.  The model itself
remains entirely dense; ``torch.sparse`` has no autograd path for attention,
LayerNorm, or embedding gathers.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from endo_model.data.constants import DNS_SENTINEL


class StudyPanel:
    """Sparse counts for the cells of one study, restricted to vocabulary genes.

    Args:
        matrix:        (n_cells, n_panel_genes) counts, already restricted to
                       the study's vocabulary-mapped genes and to the cells of
                       one split.  Any scipy sparse format or a dense ndarray.
        global_idx:    (n_panel_genes,) global vocabulary index of each column.
        vocab_size:    Width of the dense vector returned by ``counts()``.
        study_name:    Human-readable name, for error messages only.
    """

    def __init__(
        self,
        matrix,
        global_idx: np.ndarray,
        vocab_size: int,
        study_name: str = "",
    ) -> None:
        csr = scipy.sparse.csr_matrix(matrix, dtype=np.float32)
        if csr.shape[1] != len(global_idx):
            raise ValueError(
                f"Study '{study_name}': matrix has {csr.shape[1]} columns but "
                f"{len(global_idx)} global indices were supplied."
            )

        self._data = csr.data
        self._indices = csr.indices
        self._indptr = csr.indptr
        self._n_cells = csr.shape[0]
        self._global_idx = np.asarray(global_idx, dtype=np.int64)
        self._vocab_size = int(vocab_size)
        self._study_name = study_name

    def counts(self, row: int) -> np.ndarray:
        """Return the dense ``(vocab_size,)`` count vector for one cell.

        Genes outside this study's panel carry ``DNS_SENTINEL``; panel genes
        with no recorded count are a measured zero.  That distinction is the
        whole point of DNS, so the two scatters below must stay in this order.
        """
        if not 0 <= row < self._n_cells:
            raise IndexError(
                f"Study '{self._study_name}': row {row} out of range "
                f"[0, {self._n_cells})."
            )

        out = np.full(self._vocab_size, float(DNS_SENTINEL), dtype=np.float32)
        out[self._global_idx] = 0.0          # panel genes: measured zero

        start, end = self._indptr[row], self._indptr[row + 1]
        cols = self._global_idx[self._indices[start:end]]
        out[cols] = self._data[start:end]    # panel genes: measured non-zero
        return out

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def nbytes(self) -> int:
        """Bytes held by the sparse arrays — used for the load-time size log."""
        return int(self._data.nbytes + self._indices.nbytes + self._indptr.nbytes)
