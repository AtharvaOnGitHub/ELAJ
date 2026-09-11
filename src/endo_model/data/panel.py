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

    def as_csr(self) -> scipy.sparse.csr_matrix:
        """The whole panel as one CSR matrix in GLOBAL vocabulary coordinates.

        Shape ``(n_cells, vocab_size)``. This is the representation every
        competing method reads -- scVI, Harmony, Scanorama, PCA -- and none of
        them can hold a "not measured" marker that differs from an implicit
        zero, so genes outside the panel are structurally absent here rather
        than ``DNS_SENTINEL``. The distinction is not lost: ``measured_mask``
        says which genes this study measured, so a caller that wants the
        intersection of every study's panel can build it.

        ``counts()`` remains the path the model reads and is byte-for-byte
        unchanged. This exists so the baseline comparison never has to densify
        a million rows to get a matrix it will immediately re-sparsify.

        The data is copied. The panel's arrays are shared with every
        ``counts()`` call, and a caller scaling or log-transforming the
        returned matrix in place would otherwise corrupt what the model sees.
        """
        matrix = scipy.sparse.csr_matrix(
            (self._data.copy(), self._global_idx[self._indices], self._indptr.copy()),
            shape=(self._n_cells, self._vocab_size),
        )
        matrix.sort_indices()
        return matrix

    @property
    def measured_mask(self) -> np.ndarray:
        """``(vocab_size,)`` bool: which vocabulary genes this panel measured.

        A gene that is True here and zero in a cell is a measured zero. A gene
        that is False was never on this study's panel. ``counts()`` encodes the
        second case as ``DNS_SENTINEL``; ``as_csr()`` cannot, so this carries it.
        """
        mask = np.zeros(self._vocab_size, dtype=bool)
        mask[self._global_idx] = True
        return mask

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def nbytes(self) -> int:
        """Bytes held by the sparse arrays — used for the load-time size log."""
        return int(self._data.nbytes + self._indices.nbytes + self._indptr.nbytes)
