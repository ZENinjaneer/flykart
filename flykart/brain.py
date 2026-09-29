"""Whole-CNS leaky integrate-and-fire model, run on GPU or compiled CPU kernels.

Same equations and parameters as the reference Brian2 model of Shiu et al.
(Nature 2024, github.com/philshiu/Drosophila_brain_model):

    dv/dt = (v0 - v + g) / tau_m        (neither v nor g integrate while refractory)
    dg/dt = -g / tau_s
    v > v_th  ->  spike; v = v_reset, g = 0, refractory for t_ref
    a spike adds w_syn * (#synapses) * sign(presynaptic NT) to g, t_delay later
    Poisson stimulation adds w_syn * 250 = 68.75 mV to v (i.e. forces a spike)

Speed trick: the synaptic delay is exactly D = 18 time steps, so nothing a spike
does can matter until D steps later. One "block" therefore advances every neuron
D steps independently (one fused kernel that also packs each neuron's spikes
into an 18-bit mask), then one propagation kernel turns those masks into the
next block's synaptic input. Most inputs are silent, so most edges cost a
single cached gather. On an RTX 5090 Laptop GPU a block takes ~0.43 ms, i.e.
the whole male CNS runs about 4x faster than real time.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only installs
    HAVE_TRITON = False


@dataclass(frozen=True)
class LIFParams:
    dt: float = 0.1  # ms
    v0: float = -52.0  # mV, resting potential
    v_reset: float = -52.0  # mV
    v_th: float = -45.0  # mV
    tau_m: float = 20.0  # ms, membrane
    tau_s: float = 5.0  # ms, synaptic
    t_ref: float = 2.2  # ms
    t_delay: float = 1.8  # ms
    w_syn: float = 0.275  # mV per synapse
    f_poisson: float = 250.0  # Poisson kick = w_syn * f_poisson

    @property
    def delay_steps(self) -> int:
        return round(self.t_delay / self.dt)

    @property
    def ref_steps(self) -> int:
        return round(self.t_ref / self.dt)

    @property
    def poisson_kick(self) -> float:
        return self.w_syn * self.f_poisson

    def propagators(self) -> tuple[float, float, float]:
        """Exact one-step solution of the linear (v, g) system."""
        dm = math.exp(-self.dt / self.tau_m)
        ds = math.exp(-self.dt / self.tau_s)
        pgv = (dm * self.dt / self.tau_m if self.tau_s == self.tau_m else
               self.tau_s / (self.tau_s - self.tau_m) * (ds - dm))
        return dm, ds, pgv


if HAVE_TRITON:

    @triton.jit
    def _lif_block_kernel(
        v_ptr, g_ptr, ref_ptr, rate_p_ptr, kick_ptr, inp_ptr, bits_ptr, cnt_ptr,
        n, seed,
        dm, ds, pgv, v0, v_reset, v_th, ref_steps,
        D: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(v_ptr + offs, mask=mask, other=0.0)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        ref = tl.load(ref_ptr + offs, mask=mask, other=0)
        p = tl.load(rate_p_ptr + offs, mask=mask, other=0.0)
        kick = tl.load(kick_ptr + offs, mask=mask, other=0.0)
        cnt = tl.load(cnt_ptr + offs, mask=mask, other=0)
        bits = tl.zeros([BLOCK], dtype=tl.int32)
        for k in range(D):
            active = ref <= 0
            v = tl.where(active, v0 + (v - v0) * dm + g * pgv, v)
            g = tl.where(active, g * ds, g)
            r = tl.rand(seed, offs * D + k)
            v = tl.where(r < p, v + kick, v)
            spk = active & (v > v_th)
            g += tl.load(inp_ptr + k * n + offs, mask=mask, other=0.0)
            v = tl.where(spk, v_reset, v)
            g = tl.where(spk, 0.0, g)
            ref = tl.where(spk, ref_steps, tl.maximum(ref - 1, 0))
            bits = bits | (spk.to(tl.int32) << k)
            cnt += spk.to(tl.int32)
        tl.store(v_ptr + offs, v, mask=mask)
        tl.store(g_ptr + offs, g, mask=mask)
        tl.store(ref_ptr + offs, ref, mask=mask)
        tl.store(cnt_ptr + offs, cnt, mask=mask)
        tl.store(bits_ptr + offs, bits, mask=mask)

    @triton.jit
    def _propagate_kernel(indptr_ptr, indices_ptr, w_ptr, bits_ptr, out_ptr, n,
                          D: tl.constexpr, DP: tl.constexpr, CHUNK: tl.constexpr):
        # One program per postsynaptic neuron: out[k, row] = sum_e w[e] * spiked(pre[e], k).
        # Presynaptic spikes are a bitmask per neuron (bit k = spiked at step k), so a
        # chunk of edges whose inputs were all silent costs one small gather.
        row = tl.program_id(0)
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        kk = tl.arange(0, DP)
        acc = tl.zeros([DP], dtype=tl.float32)
        for e0 in range(start, end, CHUNK):
            offs = e0 + tl.arange(0, CHUNK)
            m = offs < end
            pre = tl.load(indices_ptr + offs, mask=m, other=0)
            b = tl.load(bits_ptr + pre, mask=m, other=0)
            if tl.max(b, axis=0) > 0:
                ww = tl.load(w_ptr + offs, mask=m, other=0.0)
                sel = ((b[:, None] >> kk[None, :]) & 1).to(tl.float32)
                acc += tl.sum(ww[:, None] * sel, axis=0)
        tl.store(out_ptr + kk * n + row, acc, mask=kk < D)


class Brain:
    """State for the whole connectome plus the block-stepping loop.

    W is CSR with rows = postsynaptic, cols = presynaptic, values already equal
    to w_syn * synapse_count * sign(presynaptic neuron).
    """

    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        n: int,
        params: LIFParams = LIFParams(),
        device: str | None = None,
        use_triton: bool | None = None,
        seed: int = 0,
        cpu_threads: int | None = None,
        cpu_rng: str = "sparse",
    ):
        if params.dt <= 0 or params.tau_m <= 0 or params.tau_s <= 0:
            raise ValueError("dt, tau_m, and tau_s must be positive")
        if params.t_ref < 0 or params.delay_steps < 1:
            raise ValueError("t_ref must be nonnegative and delay must span at least one step")
        if n < 1:
            raise ValueError("n must be positive")
        self.p = params
        self.n = n
        self.D = params.delay_steps
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        on_gpu = self.device.type == "cuda"
        self.use_triton = (HAVE_TRITON and on_gpu) if use_triton is None else use_triton
        if self.use_triton and (not HAVE_TRITON or not on_gpu):
            raise ValueError("Triton requires an available CUDA device and Triton installation")
        if self.use_triton and self.D > 31:
            raise ValueError("The Triton spike mask supports at most 31 delay steps")
        self.dm, self.ds, self.pgv = params.propagators()

        dev = self.device
        if dev.type == "cpu" and not self.use_triton:
            from numba import config
            from . import cpu_kernels

            self._cpu_kernels = cpu_kernels
            requested = cpu_threads if cpu_threads is not None else int(
                os.environ.get("FLYKART_CPU_THREADS", min(4, config.NUMBA_NUM_THREADS)))
            if not 1 <= requested <= config.NUMBA_NUM_THREADS:
                raise ValueError(f"cpu_threads must be between 1 and {config.NUMBA_NUM_THREADS}")
            self.cpu_threads = requested
            # Partition postsynaptic rows; each worker has exclusive destinations.
            # Stable outgoing order keeps every row's additions deterministic.
            self._cpu_partition_size = max(1, (n + 7) // 8)
            partitions = (n + self._cpu_partition_size - 1) // self._cpu_partition_size
            posts = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
            keys = (posts // self._cpu_partition_size).astype(np.int64) * n + indices
            order = np.argsort(keys, kind="stable")
            self._cpu_posts = posts[order]
            self._cpu_weights = np.asarray(weights, dtype=np.float32)[order]
            sizes = np.bincount(keys, minlength=partitions * n).reshape(partitions, n)
            offsets = np.r_[0, np.cumsum(sizes.ravel())]
            self._cpu_indptr = np.empty((partitions, n + 1), dtype=np.int64)
            for part in range(partitions):
                self._cpu_indptr[part] = offsets[part * n:(part + 1) * n + 1]
            del order, posts, keys, sizes, offsets
        else:
            crow = torch.as_tensor(indptr.astype(np.int32)).to(dev)
            col = torch.as_tensor(indices.astype(np.int32)).to(dev)
            val = torch.as_tensor(weights.astype(np.float32)).to(dev)
            if self.use_triton:
                self.crow, self.col, self.val = crow, col, val
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.W = torch.sparse_csr_tensor(crow, col, val, size=(n, n), check_invariants=False)
        self.v = torch.full((n,), params.v0, device=dev)
        self.g = torch.zeros(n, device=dev)
        self.ref = torch.zeros(n, dtype=torch.int32, device=dev)
        # Spikes of the current block and the synaptic input they produce for the next
        # one. Triton path: spikes as a bitmask per neuron, input laid out (D, n).
        # CPU uses compact words and contiguous neuron-major synaptic inputs.
        # Only the non-Triton accelerator fallback needs a dense spike matrix.
        if dev.type == "cpu":
            self._cpu_spike_bits = np.zeros((n, (self.D + 31) // 32), dtype=np.uint32)
            self.bits = torch.from_numpy(self._cpu_spike_bits.view(np.int32)[:, 0])
        else:
            self.bits = torch.zeros(n, dtype=torch.int32, device=dev)
        self._spk = (torch.zeros(n, self.D, device=dev)
                     if dev.type != "cpu" and not self.use_triton else None)
        self.inp = torch.zeros(self.D, n, device=dev) if self.use_triton else torch.zeros(n, self.D, device=dev)
        # Per-neuron Poisson stimulation: probability per step, and kick size (mV).
        self.rate_hz = torch.zeros(n, device=dev)
        self.rate_p = torch.zeros(n, device=dev)
        self.kick = torch.full((n,), params.poisson_kick, device=dev)
        self.counts = torch.zeros(n, dtype=torch.int32, device=dev)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)
        self._seed = seed
        self.blocks = 0
        if dev.type == "cpu":
            from .cpu_random import CPUStimulusRNG
            self._cpu_rng = CPUStimulusRNG(n, mode=cpu_rng)
            self._cpu_rng.set_rates(self.rate_p)

    # ---- stimulation -------------------------------------------------------
    def set_rates(self, rate_hz: torch.Tensor) -> None:
        """Poisson rate (Hz) for every neuron; 0 = unstimulated."""
        self.rate_hz = rate_hz.to(self.device, torch.float32)
        self.rate_p = (self.rate_hz * (self.p.dt * 1e-3)).clamp_(0.0, 1.0)
        if self.device.type == "cpu":
            self._cpu_rng.set_rates(self.rate_p)

    def reset_state(self) -> None:
        self.v.fill_(self.p.v0)
        self.g.zero_()
        self.ref.zero_()
        self.bits.zero_()
        if self.device.type == "cpu":
            self._cpu_spike_bits.fill(0)
        if self._spk is not None:
            self._spk.zero_()
        self.inp.zero_()
        self.counts.zero_()
        self.rate_hz.zero_()
        self.rate_p.zero_()
        self.blocks = 0
        self.gen.manual_seed(self._seed)
        if self.device.type == "cpu":
            self._cpu_rng.set_rates(self.rate_p)

    @property
    def spk(self) -> torch.Tensor:
        """Spikes of the last block; CPU access reconstructs a debug snapshot.

        Normal stepping keeps only compact masks. Mutating this snapshot does
        not change CPU state. The dense accelerator fallback retains its tensor.
        """
        if self._spk is not None:
            return self._spk
        if self.device.type == "cpu":
            steps = np.arange(self.D)
            spikes = ((self._cpu_spike_bits[:, steps // 32] >>
                       (steps % 32).astype(np.uint32)) & 1).astype(np.float32)
            return torch.from_numpy(spikes)
        steps = torch.arange(self.D, device=self.device)
        return ((self.bits[:, None] >> steps) & 1).to(torch.float32)

    @spk.setter
    def spk(self, value: torch.Tensor) -> None:
        self._spk = value

    @property
    def t_ms(self) -> float:
        return self.blocks * self.D * self.p.dt

    # ---- stepping ----------------------------------------------------------
    def _block_cpu(self) -> None:
        """Advance independent neurons with strict float32 compiled arithmetic."""
        from numba import set_num_threads

        # Numba's thread mask is local to the calling thread; the simulation can
        # move from startup's thread to the server's worker thread.
        set_num_threads(self.cpu_threads)
        p = self.p
        rnd, rows = self._cpu_rng.draw(self.D, self.gen)
        self._cpu_kernels.integrate(self.v.numpy(), self.g.numpy(), self.ref.numpy(),
                  self.rate_p.numpy(), self.kick.numpy(), self._cpu_spike_bits,
                  self.counts.numpy(), self.inp.numpy(), rnd, rows,
                  *map(np.float32, (p.v0, p.v_reset, p.v_th, self.dm, self.ds, self.pgv)),
                  np.int32(p.ref_steps))

    def _block_torch(self) -> None:
        p = self.p
        if self._spk is None:
            self._spk = torch.zeros(self.n, self.D, device=self.device)
        v, g, ref = self.v, self.g, self.ref
        rnd = torch.rand(self.D, self.n, device=self.device, generator=self.gen)
        for k in range(self.D):
            active = ref <= 0
            v = torch.where(active, p.v0 + (v - p.v0) * self.dm + g * self.pgv, v)
            g = torch.where(active, g * self.ds, g)
            v = torch.where(rnd[k] < self.rate_p, v + self.kick, v)
            spk = active & (v > p.v_th)
            g = g + self.inp[:, k]
            v = torch.where(spk, p.v_reset, v)
            g = torch.where(spk, 0.0, g)
            ref = torch.where(spk, p.ref_steps, (ref - 1).clamp_(min=0))
            self.spk[:, k] = spk
            self.counts += spk
        self.v, self.g, self.ref = v, g, ref.to(torch.int32)

    def _block_triton(self) -> None:
        BLOCK = 256
        grid = (triton.cdiv(self.n, BLOCK),)
        seed = (self._seed * 1_000_003 + self.blocks) & 0x7FFFFFFF
        _lif_block_kernel[grid](
            self.v, self.g, self.ref, self.rate_p, self.kick, self.inp, self.bits, self.counts,
            self.n, seed,
            self.dm, self.ds, self.pgv, self.p.v0, self.p.v_reset, self.p.v_th, self.p.ref_steps,
            D=self.D, BLOCK=BLOCK,
        )
        _propagate_kernel[(self.n,)](
            self.crow, self.col, self.val, self.bits, self.inp, self.n,
            D=self.D, DP=triton.next_power_of_2(self.D), CHUNK=64, num_warps=2,
        )

    def _propagate_cpu(self) -> None:
        """Propagate compact spike masks with disjoint, deterministic row writes."""
        self._cpu_kernels.propagate(
            self._cpu_spike_bits, self.inp.numpy(), self._cpu_indptr,
            self._cpu_posts, self._cpu_weights, self._cpu_partition_size)

    def step_block(self) -> None:
        if self.use_triton:
            self._block_triton()
        elif self.device.type == "cpu":
            self._block_cpu()
            self._propagate_cpu()
        else:
            self._block_torch()
            self.inp = torch.sparse.mm(self.W, self.spk)
        self.blocks += 1

    def run(self, n_blocks: int) -> torch.Tensor:
        """Advance n_blocks * D steps. Returns spike counts per neuron (on device)."""
        self.counts.zero_()
        for _ in range(n_blocks):
            self.step_block()
        return self.counts

    def benchmark(self, sim_ms: float = 1000.0) -> dict:
        n_blocks = max(1, round(sim_ms / (self.D * self.p.dt)))
        self.run(2)  # warm up / compile
        self._sync()
        t0 = time.perf_counter()
        counts = self.run(n_blocks)
        self._sync()
        wall = time.perf_counter() - t0
        sim_s = n_blocks * self.D * self.p.dt * 1e-3
        return {
            "sim_ms": sim_s * 1e3,
            "wall_ms": wall * 1e3,
            "realtime_factor": sim_s / wall,
            "spikes": int(counts.sum()),
            "active_neurons": int((counts > 0).sum()),
        }

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
