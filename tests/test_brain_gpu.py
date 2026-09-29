"""CUDA regression checks; run on a GPU host and skip CPU-only installations."""

import unittest

import numpy as np
import torch

from flykart.brain import Brain, HAVE_TRITON


def small_network():
    # A signed ring gives each destination one input, avoiding differences in
    # floating-point reduction order between Triton and torch.sparse.mm.
    n = 8
    return (
        np.arange(n + 1, dtype=np.int64),
        np.roll(np.arange(n, dtype=np.int32), 1),
        np.array([64, 128, 256, -64, 128, 64, -128, 256], dtype=np.float32),
        n,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware is unavailable")
class BrainGPUTest(unittest.TestCase):
    def check_reset_replay(self, use_triton):
        brain = Brain(*small_network(), device="cuda", use_triton=use_triton, seed=29)
        self.assertFalse(hasattr(brain, "_cpu_rng"))
        self.assertFalse(hasattr(brain, "_cpu_kernels"))
        self.assertEqual(brain.use_triton, use_triton)
        rates = torch.tensor([300, 0, 150, 0, 200, 0, 0, 75], dtype=torch.float32)
        brain.set_rates(rates)
        first = []
        states = ("v", "g", "ref", "inp", "bits", "counts", "spk")
        for _ in range(3):
            brain.run(8)
            first.append({name: getattr(brain, name).clone() for name in states})
        self.assertGreater(sum(int(frame["counts"].sum()) for frame in first), 0)
        for name in ("v", "g", "inp"):
            self.assertTrue(bool(torch.isfinite(getattr(brain, name)).all()), name)

        brain.reset_state()
        self.assertEqual(brain.blocks, 0)
        self.assertEqual(brain.t_ms, 0.0)
        self.assertTrue(bool(torch.all(brain.v == brain.p.v0)))
        for name in ("g", "ref", "inp", "bits", "counts", "spk", "rate_hz", "rate_p"):
            self.assertEqual(int(torch.count_nonzero(getattr(brain, name))), 0, name)
        brain.set_rates(rates)
        for expected in first:
            brain.run(8)
            for name in states:
                self.assertTrue(torch.equal(getattr(brain, name), expected[name]), name)

    def test_torch_cuda_reset_replays_original_generator(self):
        self.check_reset_replay(use_triton=False)

    @unittest.skipUnless(HAVE_TRITON, "Triton is unavailable")
    def test_triton_reset_replays_block_seed_and_state(self):
        self.check_reset_replay(use_triton=True)

    @unittest.skipUnless(HAVE_TRITON, "Triton is unavailable")
    def test_triton_matches_torch_for_deterministic_stimulation(self):
        args = small_network()
        triton = Brain(*args, device="cuda", use_triton=True)
        reference = Brain(*args, device="cuda", use_triton=False)
        # Probability 0 or 1 removes the intentionally different CUDA RNG
        # sequences, so this checks the neuron equations and delayed propagation.
        rates = torch.tensor([10000, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        for brain in (triton, reference):
            brain.set_rates(rates)
        propagated_spikes = 0
        for _ in range(8):
            triton.run(1)
            reference.run(1)
            for name in ("counts", "ref", "spk"):
                self.assertTrue(torch.equal(getattr(triton, name), getattr(reference, name)), name)
            for name in ("v", "g"):
                torch.testing.assert_close(getattr(triton, name), getattr(reference, name),
                                           rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(triton.inp.T, reference.inp, rtol=2e-5, atol=2e-5)
            propagated_spikes += int(triton.counts[1:].sum())
        self.assertGreater(propagated_spikes, 0)


if __name__ == "__main__":
    unittest.main()
