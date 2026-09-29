"""Deterministic native CPU kernels, compiled by Numba without fast math.

Neurons advance independently. Synaptic propagation partitions *destinations*,
so workers never add to the same output and retain presynaptic summation order.
Spikes use one 32-bit word per 32 time steps, including multi-spike blocks.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=False, nogil=True, parallel=True)
def integrate(v, g, ref, rate_p, kick, bits, counts, inp, rnd, random_rows,
               v0, v_reset, v_th, dm, ds, pgv, ref_steps):
    n, steps = inp.shape
    for i in prange(n):
        vv, gg, rr = v[i], g[i], ref[i]
        count = counts[i]
        probability, impulse = rate_p[i], kick[i]
        random_row = random_rows[i]
        for word in range(bits.shape[1]):
            bits[i, word] = 0
        for k in range(steps):
            active = rr <= 0
            if active:
                vv = v0 + (vv - v0) * dm + gg * pgv
                gg = gg * ds
            if random_row >= 0 and rnd[random_row, k] < probability:
                vv = vv + impulse
            fired = active and vv > v_th
            gg = gg + inp[i, k]
            if fired:
                vv = v_reset
                gg = np.float32(0.0)
                rr = ref_steps
                bits[i, k // 32] |= np.uint32(1) << np.uint32(k % 32)
                count += 1
            elif rr > 0:
                rr -= 1
        v[i], g[i], ref[i], counts[i] = vv, gg, rr, count


@njit(cache=True, fastmath=False, nogil=True, parallel=True)
def propagate(bits, out, ptr, posts, weights, partition_size):
    n, steps = out.shape
    for partition in prange(ptr.shape[0]):
        # Each partition owns complete rows, avoiding races and false sharing
        # between time-step columns. Empty partitions also clear stale inputs.
        start = partition * partition_size
        end = min(n, start + partition_size)
        for post in range(start, end):
            for k in range(steps):
                out[post, k] = np.float32(0.0)
        for pre in range(n):
            for word in range(bits.shape[1]):
                mask = bits[pre, word]
                if mask == 0:
                    continue
                edge_start, edge_end = ptr[partition, pre], ptr[partition, pre + 1]
                if edge_start == edge_end:
                    continue
                k = word * 32
                while mask:
                    if mask & np.uint32(1):
                        for edge in range(edge_start, edge_end):
                            out[posts[edge], k] += weights[edge]
                    mask >>= np.uint32(1)
                    k += 1
