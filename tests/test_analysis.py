"""Tests for analysis and statistics modules."""
import unittest
from mcsp.analysis.statistics import ComplexityStats, analyze_complexity_landscape, compute_hardness_index


class TestComplexityStats(unittest.TestCase):

    def setUp(self):
        self.complexities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.stats = ComplexityStats(self.complexities)

    def test_summary_keys(self):
        summary = self.stats.summary()
        self.assertIn('min', summary)
        self.assertIn('max', summary)
        self.assertIn('mean', summary)
        self.assertIn('std', summary)
        self.assertIn('median', summary)
        self.assertIn('count', summary)

    def test_summary_values(self):
        summary = self.stats.summary()
        self.assertEqual(summary['min'], 1)
        self.assertEqual(summary['max'], 10)
        self.assertAlmostEqual(summary['mean'], 5.5)
        self.assertEqual(summary['count'], 10)

    def test_percentile(self):
        p50 = self.stats.percentile(50)
        self.assertAlmostEqual(p50, 5.5, places=1)

    def test_entropy(self):
        e = self.stats.entropy()
        self.assertGreater(e, 0)
        # Uniform distribution -> high entropy
        uniform_stats = ComplexityStats([1] * 100)
        e_uniform = uniform_stats.entropy()
        self.assertEqual(e_uniform, 0.0)

    def test_compare_functions(self):
        result = self.stats.compare_functions({'test_func': 5})
        self.assertIn('test_func', result)
        self.assertIn('complexity', result['test_func'])
        self.assertIn('z_score', result['test_func'])

    def test_empty_stats(self):
        s = ComplexityStats([])
        self.assertEqual(s.summary(), {})


class TestAnalyzeLandscape(unittest.TestCase):

    def test_n2_landscape(self):
        stats = analyze_complexity_landscape(2, num_random=20)
        summary = stats.summary()
        self.assertEqual(summary['count'], 20)
        self.assertGreaterEqual(summary['min'], 0)

    def test_returns_complexity_stats(self):
        result = analyze_complexity_landscape(2, num_random=10)
        self.assertIsInstance(result, ComplexityStats)


class TestHardnessIndex(unittest.TestCase):

    def test_range(self):
        for n in range(2, 5):
            for c in range(1, 10):
                h = compute_hardness_index(c, n)
                self.assertGreaterEqual(h, 0.0)
                self.assertLessEqual(h, 1.0)

    def test_harder_function_higher_index(self):
        h_easy = compute_hardness_index(1, 3)
        h_hard = compute_hardness_index(5, 3)
        self.assertLessEqual(h_easy, h_hard)


class TestHistogram(unittest.TestCase):

    def test_histogram_shape(self):
        from mcsp.ml.data_generation import ComplexityDistribution
        samples = [{'complexity': i % 5 + 1, 'truth_table': [0]*8, 'n': 3}
                   for i in range(50)]
        dist = ComplexityDistribution(samples)
        counts, edges = dist.histogram()
        self.assertEqual(len(counts) + 1, len(edges))
        self.assertEqual(sum(counts), 50)


if __name__ == '__main__':
    unittest.main()
