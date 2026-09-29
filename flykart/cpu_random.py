"""CPU stimulation draws with explicit reproducibility modes.

Sparse mode preserves the independent uniform input distribution while drawing
only for stimulated neurons. Full mode preserves the original CPU draw sequence.
"""

from __future__ import annotations

import numpy as np
import torch


class CPUStimulusRNG:
    def __init__(self, n: int, mode: str = "sparse"):
        if mode not in ("sparse", "full"):
            raise ValueError("cpu_rng must be 'sparse' or 'full'")
        self.n = n
        self.mode = mode
        self.rows = np.arange(n, dtype=np.int32) if mode == "full" else np.full(n, -1, dtype=np.int32)
        self.active_count = n if mode == "full" else 0

    def set_rates(self, rate_p: torch.Tensor) -> None:
        if tuple(rate_p.shape) != (self.n,):
            raise ValueError(f"expected {self.n} stimulation probabilities")
        if self.mode == "sparse":
            active = np.flatnonzero(rate_p.numpy() > 0)
            self.rows.fill(-1)
            self.rows[active] = np.arange(len(active), dtype=np.int32)
            self.active_count = len(active)

    def draw(self, steps: int, generator: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == "full":
            # Draw in the original time-major order, then expose a view for the
            # neuron-major native loop. No transpose copy is necessary.
            values = torch.rand(steps, self.n, generator=generator, dtype=torch.float32).numpy().T
        else:
            values = torch.rand(self.active_count, steps, generator=generator, dtype=torch.float32).numpy()
        return values, self.rows
