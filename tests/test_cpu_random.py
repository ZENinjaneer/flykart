import unittest

import numpy as np
import torch

from flykart.cpu_random import CPUStimulusRNG


class CPUStimulusRNGTest(unittest.TestCase):
    def test_full_mode_preserves_original_draw_order(self):
        source = CPUStimulusRNG(64, "full")
        generator = torch.Generator().manual_seed(7)
        reference = torch.Generator().manual_seed(7)
        for _ in range(3):
            values, rows = source.draw(18, generator)
            expected = torch.rand(18, 64, generator=reference).numpy()
            np.testing.assert_array_equal(values[rows].T, expected)
        self.assertTrue(torch.equal(generator.get_state(), reference.get_state()))

    def test_sparse_draws_only_for_current_stimulated_neurons(self):
        source = CPUStimulusRNG(64)
        generator = torch.Generator().manual_seed(9)
        rates = torch.zeros(64)
        rates[[3, 21, 60]] = torch.tensor([0.001, 0.05, 1.0])
        source.set_rates(rates)
        values, rows = source.draw(18, generator)
        self.assertEqual(values.shape, (3, 18))
        np.testing.assert_array_equal(np.flatnonzero(rows >= 0), [3, 21, 60])
        np.testing.assert_array_equal(rows[[3, 21, 60]], [0, 1, 2])
        self.assertTrue(values.flags.c_contiguous)
        rates.zero_()
        rates[12] = 0.02
        source.set_rates(rates)
        values, rows = source.draw(18, generator)
        self.assertEqual(values.shape, (1, 18))
        np.testing.assert_array_equal(np.flatnonzero(rows >= 0), [12])

    def test_silent_input_does_not_advance_generator(self):
        source = CPUStimulusRNG(64)
        generator = torch.Generator().manual_seed(10)
        before = generator.get_state().clone()
        values, rows = source.draw(18, generator)
        self.assertEqual(values.shape, (0, 18))
        self.assertTrue((rows == -1).all())
        self.assertTrue(torch.equal(before, generator.get_state()))

    def test_sparse_replay_and_stimulation_probabilities(self):
        source = CPUStimulusRNG(6)
        rates = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 1.0])
        source.set_rates(rates)
        generator = torch.Generator().manual_seed(13)
        values, rows = source.draw(50_000, generator)
        original = values.copy()
        generator.manual_seed(13)
        replay, _ = source.draw(50_000, generator)
        np.testing.assert_array_equal(original, replay)
        for neuron in range(1, 6):
            probability = float(rates[neuron])
            observed = float((values[rows[neuron]] < probability).mean())
            tolerance = max(0.0001, 6 * np.sqrt(probability * (1 - probability) / 50_000))
            self.assertLessEqual(abs(observed - probability), tolerance)

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            CPUStimulusRNG(4, "unknown")


if __name__ == "__main__":
    unittest.main()
