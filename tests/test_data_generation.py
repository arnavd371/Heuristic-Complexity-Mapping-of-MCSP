"""Tests for dataset generation utilities."""
import unittest

from mcsp.ml.data_generation import DatasetGenerator


class TestDatasetGenerator(unittest.TestCase):
    def test_generate_dataset_qmc_is_reproducible(self):
        """Ensure random sample generation works (regression for missing numpy import)."""
        gen = DatasetGenerator(n=3, solver_type='qmc', max_size=8)

        samples_a = gen.generate_dataset(num_samples=12, seed=7)
        samples_b = gen.generate_dataset(num_samples=12, seed=7)

        self.assertEqual(len(samples_a), len(samples_b))
        self.assertEqual(samples_a, samples_b)
        for sample in samples_a:
            self.assertEqual(len(sample['truth_table']), 2 ** 3)
            self.assertIsInstance(sample['complexity'], int)


if __name__ == '__main__':
    unittest.main()
