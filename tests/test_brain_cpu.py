import unittest
import warnings

import numpy as np
import torch

from flykart.brain import Brain, LIFParams


class DenseReferenceBrain(Brain):
    """Keep the original PyTorch integration and propagation as a reference."""

    def __init__(self, indptr, indices, weights, n, **kwargs):
        super().__init__(indptr, indices, weights, n, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.W = torch.sparse_csr_tensor(
                torch.tensor(indptr, dtype=torch.int32),
                torch.tensor(indices, dtype=torch.int32),
                torch.tensor(weights, dtype=torch.float32),
                size=(n, n),
            )

    def _block_cpu(self):
        self._block_torch()

    def _propagate_cpu(self):
        self.inp = torch.sparse.mm(self.W, self.spk)


class BrainCPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_matches_dense_reference_and_reset_replay(self):
        rng = np.random.default_rng(17)
        n = 48
        post, pre = np.nonzero(rng.random((n, n)) < 0.25)
        indptr = np.r_[0, np.cumsum(np.bincount(post, minlength=n))]
        weights = rng.uniform(-6.0, 14.0, len(pre)).astype(np.float32)
        args = (indptr, pre.astype(np.int32), weights, n)
        brain = Brain(*args, device="cpu", seed=29, cpu_rng="full")
        reference = DenseReferenceBrain(*args, device="cpu", seed=29, cpu_rng="full")
        rates = rng.uniform(0.0, 220.0, (80, n)).astype(np.float32)
        states = ("counts", "spk", "v", "g", "ref", "inp")
        first_run = []

        for replay in range(2):
            for block, rate in enumerate(rates):
                for model in (brain, reference):
                    model.set_rates(torch.from_numpy(rate.copy()))
                    model.run(1)
                for name in states:
                    self.assertTrue(
                        torch.equal(getattr(brain, name), getattr(reference, name)),
                        f"{name} differs at block {block}, replay {replay}",
                    )
                self.assertEqual(brain.t_ms, reference.t_ms)
                if replay:
                    for name, expected in first_run[block].items():
                        self.assertTrue(torch.equal(getattr(brain, name), expected), name)
                else:
                    first_run.append({name: getattr(brain, name).clone() for name in states})
            for model in (brain, reference):
                model.reset_state()
                self.assertEqual(model.t_ms, 0.0)
                self.assertEqual(torch.count_nonzero(model.rate_hz), 0)
                self.assertEqual(torch.count_nonzero(model.rate_p), 0)

    def test_silent_and_disconnected_spikes_clear_previous_input(self):
        brain = Brain(
            np.zeros(5, dtype=np.int64), np.array([], dtype=np.int32),
            np.array([], dtype=np.float32), 4, device="cpu",
        )
        for firing in (False, True):
            brain.inp.fill_(123.0)
            brain._cpu_spike_bits.fill((1 << brain.D) - 1 if firing else 0)
            brain._propagate_cpu()
            self.assertEqual(torch.count_nonzero(brain.inp), 0)

    def test_multiple_spikes_and_long_delay_masks_match_reference(self):
        # Refractory intervals shorter than a block allow multiple spikes; D=65
        # also checks that spikes beyond one or two 32-bit words propagate.
        args = (np.arange(5, dtype=np.int64), np.array([1, 2, 3, 0], dtype=np.int32),
                np.array([1.5, -2.0, 4.0, -0.5], dtype=np.float32), 4)
        for steps, refractory in ((1, 0), (18, 0), (33, 1), (65, 2), (18, 22)):
            with self.subTest(steps=steps, refractory=refractory):
                params = LIFParams(t_delay=steps * 0.1, t_ref=refractory * 0.1)
                brain = Brain(*args, params=params, device="cpu", cpu_rng="full")
                reference = DenseReferenceBrain(*args, params=params, device="cpu", cpu_rng="full")
                for model in (brain, reference):
                    model.set_rates(torch.full((4,), 10000.0))
                for _ in range(4):
                    brain.run(1)
                    reference.run(1)
                    for name in ("v", "g", "ref", "counts", "spk", "inp"):
                        self.assertTrue(torch.equal(getattr(brain, name), getattr(reference, name)), name)
                self.assertIsNone(brain._spk)
                self.assertEqual(brain._cpu_spike_bits.nbytes, 4 * 4 * ((steps + 31) // 32))
                brain.reset_state()
                self.assertEqual(np.count_nonzero(brain._cpu_spike_bits), 0)

    def test_parallel_sparse_replay_is_independent_of_thread_count(self):
        from numba import config

        n = 4096  # Enough independent neuron and destination work for all workers.
        rng = np.random.default_rng(11)
        pre = np.sort(rng.integers(0, n, size=(n, 3)), axis=1).astype(np.int32).ravel()
        args = (np.arange(n + 1, dtype=np.int64) * 3, pre,
                rng.uniform(-5, 8, size=len(pre)).astype(np.float32), n)
        a = Brain(*args, device="cpu", seed=16, cpu_threads=1)
        b = Brain(*args, device="cpu", seed=16, cpu_threads=min(4, config.NUMBA_NUM_THREADS))
        rates = torch.zeros(n)
        rates[::9] = 250.0
        states = ("v", "g", "ref", "spk", "inp", "counts")
        saved = []
        for replay in range(2):
            for model in (a, b):
                model.set_rates(rates.clone())
            for step in range(8):
                for model in (a, b):
                    model.run(2)
                for name in states:
                    self.assertTrue(torch.equal(getattr(a, name), getattr(b, name)), name)
                    if replay:
                        self.assertTrue(torch.equal(getattr(a, name), saved[step][name]), name)
                if not replay:
                    saved.append({name: getattr(a, name).clone() for name in states})
            a.reset_state()
            b.reset_state()

    def test_parameter_validation_and_equal_time_constants(self):
        args = (np.zeros(2, dtype=np.int64), np.array([], dtype=np.int32),
                np.array([], dtype=np.float32), 1)
        for params in (LIFParams(dt=0), LIFParams(t_delay=0), LIFParams(t_ref=-1),
                       LIFParams(tau_m=0), LIFParams(tau_s=-1)):
            with self.subTest(params=params), self.assertRaises(ValueError):
                Brain(*args, params=params, device="cpu")
        with self.assertRaises(ValueError):
            Brain(*args, device="cpu", cpu_threads=0)
        params = LIFParams(tau_m=5.0, tau_s=5.0)
        dm, ds, pgv = params.propagators()
        self.assertEqual(dm, ds)
        self.assertAlmostEqual(pgv, dm * params.dt / params.tau_m)


if __name__ == "__main__":
    unittest.main()
