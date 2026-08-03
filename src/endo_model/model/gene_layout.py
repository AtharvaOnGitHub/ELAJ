"""Per-study layout of measured genes, cached across steps.

`_build_gene_reps` needs, for every gene group, which of that group's genes were
measured in this study and where each one sits in the packed `(G_meas,)` output.
That mapping depends on nothing but the study's did-not-sample pattern, and
`PerStudyBatchSampler` guarantees one study per batch — so there are exactly as
many distinct layouts as there are studies (11), not one per step.

Recomputing it every forward pass cost two things that do not show up in a FLOP
count:

  * a device synchronisation — `.tolist()` on a ~15 000-element tensor drains
    every queued kernel in the middle of the forward pass, once per step;
  * one Python iteration per gene in the vocabulary (~15 495), building a dict
    the next step would throw away.

Both are now paid once per study. The build itself is also done with tensor ops
rather than a Python loop, so the one time it does run it stays on the device.
"""

from __future__ import annotations

import torch
from torch import Tensor


class GroupSlice:
    """Where one group's measured genes sit, in both index spaces.

    Args:
        name:           Group name, for the embedding table lookup.
        seq_pos:        The group's position in the aggregation output's K axis.
        within_idxs:    (n_sel,) indices *within the group*, for the group's
                        embedding table.
        meas_positions: (n_sel,) matching positions in the packed `(G_meas,)`
                        axis of `gene_reps`.
    """

    __slots__ = ("name", "seq_pos", "within_idxs", "meas_positions")

    def __init__(
        self, name: str, seq_pos: int, within_idxs: Tensor, meas_positions: Tensor
    ) -> None:
        self.name = name
        self.seq_pos = seq_pos
        self.within_idxs = within_idxs
        self.meas_positions = meas_positions

    @property
    def n_selected(self) -> int:
        return int(self.within_idxs.shape[0])


class StudyLayout:
    """The measured-gene layout for one study.

    Args:
        measured_global_idxs: (G_meas,) global vocabulary indices that this
                              study actually assayed, ascending.
        slices:               One `GroupSlice` per group that contributed at
                              least one measured gene. Groups with none are
                              omitted rather than carried as empty, so the
                              consumer needs no per-group emptiness check.
    """

    __slots__ = ("measured_global_idxs", "slices")

    def __init__(
        self, measured_global_idxs: Tensor, slices: list[GroupSlice]
    ) -> None:
        self.measured_global_idxs = measured_global_idxs
        self.slices = slices

    @property
    def n_measured(self) -> int:
        return int(self.measured_global_idxs.shape[0])


class MeasuredGeneLayoutCache:
    """Builds and caches one `StudyLayout` per (study, device).

    Args:
        group_names:       Vocabulary group names, in aggregation-sequence order.
        group_global_idxs: group name → its genes' global vocabulary indices.
    """

    def __init__(
        self, group_names: list[str], group_global_idxs: dict[str, list[int]]
    ) -> None:
        self._group_names = list(group_names)
        self._group_global_idxs = group_global_idxs
        self._seq_pos = {name: i for i, name in enumerate(self._group_names)}
        self._index_tensors: dict[tuple[str, torch.device], Tensor] = {}
        self._cache: dict[tuple[int, torch.device], StudyLayout] = {}

    def get(self, study_id: int, is_dns_row: Tensor) -> StudyLayout:
        """Return the layout for `study_id`, building it on first use.

        Args:
            study_id:   Which study this batch came from.
            is_dns_row: (vocab_size,) bool — one row of `batch["is_dns"]`. Every
                        cell in a batch shares it, so row 0 is the whole study.

        The device is part of the key because the cached tensors are indices
        that get used directly in device-side gathers; a layout built on CPU
        would silently fail against a CUDA batch.
        """
        key = (study_id, is_dns_row.device)
        layout = self._cache.get(key)
        if layout is None:
            layout = self._build(is_dns_row)
            self._cache[key] = layout
        return layout

    def _build(self, is_dns_row: Tensor) -> StudyLayout:
        """Compute a study's layout entirely with tensor ops, no Python loop."""
        device = is_dns_row.device
        measured_global_idxs = (~is_dns_row).nonzero(as_tuple=True)[0]
        n_measured = measured_global_idxs.shape[0]

        # Scatter the inverse map once: global index → packed position, with -1
        # marking a gene this study never assayed. One vocabulary-sized vector
        # replaces the per-gene dict lookup this used to do.
        position_of = is_dns_row.new_full(
            (is_dns_row.shape[0],), -1, dtype=torch.long
        )
        position_of[measured_global_idxs] = torch.arange(n_measured, device=device)

        slices: list[GroupSlice] = []
        for name in self._group_names:
            global_idxs = self._global_index_tensor(name, device)
            positions = position_of[global_idxs]           # (G_k,)
            selected = positions >= 0
            if not bool(selected.any()):
                continue
            slices.append(
                GroupSlice(
                    name=name,
                    seq_pos=self._seq_pos[name],
                    within_idxs=selected.nonzero(as_tuple=True)[0],
                    meas_positions=positions[selected],
                )
            )

        return StudyLayout(measured_global_idxs, slices)

    def _global_index_tensor(self, name: str, device: torch.device) -> Tensor:
        """The group's global indices as a device tensor, built once per device."""
        key = (name, device)
        tensor = self._index_tensors.get(key)
        if tensor is None:
            tensor = torch.tensor(
                self._group_global_idxs[name], dtype=torch.long, device=device
            )
            self._index_tensors[key] = tensor
        return tensor
