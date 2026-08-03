"""NaNGuard: finds the module that *originates* a non-finite activation.

The error a NaN eventually raises names the victim, not the cause.  DSBN
normalises over the flattened (B x G) axis, so a single bad value anywhere in
a batch makes every activation after it non-finite; by the time the latent
sampler rejects its parameters the evidence is several hundred modules
downstream.

This attaches a forward hook to every named module and records which ones emit
non-finite output.  The module whose inputs were all finite but whose output is
not is the origin.  Everything after it is a consequence.

Installing the hooks costs one device synchronisation per module per forward,
so this is not something to leave on for a whole run — TrainingMonitor installs
it only to replay a step that has already failed.
"""

from __future__ import annotations

import torch.nn as nn

from endo_model.utils.tensors import describe, is_all_finite, nonfinite_count


class ModuleEvent:
    """One module's contribution to a non-finite forward pass.

    Args:
        name:     Dotted module path, as given by ``named_modules()``.
        module:   The module itself, for its type name.
        n_in:     Non-finite elements across all positional inputs.
        n_out:    Non-finite elements in the output.
        in_desc:  Shape/range summary of the inputs.
        out_desc: Shape/range summary of the output.
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        n_in: int,
        n_out: int,
        in_desc: str,
        out_desc: str,
    ) -> None:
        self.name = name
        self.type_name = type(module).__name__
        self.n_in = n_in
        self.n_out = n_out
        self.in_desc = in_desc
        self.out_desc = out_desc

    @property
    def is_origin(self) -> bool:
        """True when this module manufactured the fault rather than inherited it."""
        return self.n_in == 0

    def as_row(self) -> str:
        return (
            f"{self.name:<52} {self.type_name:<22} "
            f"in={self.n_in:>8} out={self.n_out:>10}"
        )


class NaNGuard:
    """Context manager recording, in execution order, which modules emit NaN.

    Args:
        model: The module tree to instrument.  Every descendant is hooked; the
               root is skipped because its "output" is the whole result dict,
               which would report as non-finite for any fault anywhere.

    Usage:
        with NaNGuard(model) as guard:
            model(batch)
        origin = guard.origin()
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._handles: list = []
        self.events: list[ModuleEvent] = []

    def __enter__(self) -> "NaNGuard":
        for name, module in self._model.named_modules():
            if name:
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        """Build the hook closure that records `name`'s output finiteness."""

        def hook(module, inputs, output):
            if is_all_finite(output):
                return
            self.events.append(
                ModuleEvent(
                    name=name,
                    module=module,
                    n_in=nonfinite_count(inputs),
                    n_out=nonfinite_count(output),
                    in_desc=", ".join(describe(i) for i in inputs),
                    out_desc=describe(output),
                )
            )

        return hook

    def origin(self) -> ModuleEvent | None:
        """First module with finite inputs and non-finite output, if any.

        None means either that nothing went non-finite, or that the fault
        entered with the batch itself rather than being manufactured by a
        module — the caller distinguishes those by whether `events` is empty.
        """
        for event in self.events:
            if event.is_origin:
                return event
        return None

    def lines(self, limit: int = 15) -> list[str]:
        """Human-readable trace, origin first, truncated to `limit` rows."""
        if not self.events:
            return ["    no non-finite values in any module output"]

        origin = self.origin()
        out = [f"    {len(self.events)} module(s) emitted non-finite output", ""]
        if origin is not None:
            out += [
                f"    ORIGIN: {origin.name}  ({origin.type_name})",
                f"      inputs  : all finite — {origin.in_desc}",
                f"      output  : {origin.n_out} non-finite — {origin.out_desc}",
                "",
                "    Everything below is downstream of it:",
            ]
        else:
            out += [
                "    No module had finite inputs and non-finite output, so the "
                "fault entered with the batch or the parameters — see above.",
            ]

        for event in self.events[:limit]:
            marker = "  <-- ORIGIN" if event is origin else ""
            out.append(f"      {event.as_row()}{marker}")
        if len(self.events) > limit:
            out.append(f"      ... and {len(self.events) - limit} more")
        return out
