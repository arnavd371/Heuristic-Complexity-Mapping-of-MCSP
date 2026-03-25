"""Tests for MCSP solvers."""
import unittest
from mcsp.core.truth_table import TruthTable
from mcsp.solvers.quine_mccluskey import QuineMcCluskey


class TestQMC(unittest.TestCase):

    def test_and2_minimize(self):
        tt = TruthTable(2, [0, 0, 0, 1])
        qmc = QuineMcCluskey(2)
        primes = qmc.minimize(tt)
        self.assertTrue(len(primes) >= 1)

    def test_or2_minimize(self):
        tt = TruthTable(2, [0, 1, 1, 1])
        qmc = QuineMcCluskey(2)
        primes = qmc.minimize(tt)
        self.assertTrue(len(primes) >= 1)

    def test_parity3_minimize(self):
        tt = TruthTable.parity(3)
        qmc = QuineMcCluskey(3)
        primes = qmc.minimize(tt)
        self.assertTrue(len(primes) >= 1)

    def test_zero_function(self):
        tt = TruthTable.zero(2)
        qmc = QuineMcCluskey(2)
        primes = qmc.minimize(tt)
        self.assertEqual(primes, [])

    def test_one_function(self):
        tt = TruthTable.one(2)
        qmc = QuineMcCluskey(2)
        primes = qmc.minimize(tt)
        self.assertEqual(primes, ['--'])

    def test_circuit_correctness_and2(self):
        n = 2
        tt = TruthTable(n, [0, 0, 0, 1])
        qmc = QuineMcCluskey(n)
        circuit = qmc.cover_to_circuit(tt, n)
        self.assertTrue(circuit.is_correct(tt))

    def test_circuit_correctness_or2(self):
        n = 2
        tt = TruthTable(n, [0, 1, 1, 1])
        qmc = QuineMcCluskey(n)
        circuit = qmc.cover_to_circuit(tt, n)
        self.assertTrue(circuit.is_correct(tt))

    def test_circuit_correctness_parity3(self):
        n = 3
        tt = TruthTable.parity(n)
        qmc = QuineMcCluskey(n)
        circuit = qmc.cover_to_circuit(tt, n)
        self.assertTrue(circuit.is_correct(tt))

    def test_estimate_complexity_and2(self):
        tt = TruthTable(2, [0, 0, 0, 1])
        qmc = QuineMcCluskey(2)
        c = qmc.estimate_complexity(tt)
        self.assertGreaterEqual(c, 1)

    def test_n_too_large(self):
        with self.assertRaises(ValueError):
            QuineMcCluskey(7)


class TestSATSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import z3
            cls.z3_available = True
        except ImportError:
            cls.z3_available = False

    def setUp(self):
        if not self.z3_available:
            self.skipTest("z3 not available")

    def test_and2(self):
        from mcsp.solvers.sat_solver import MCSPSatSolver
        n = 2
        tt = TruthTable(n, [0, 0, 0, 1])
        solver = MCSPSatSolver(n, max_gates=3, timeout_ms=10000)
        circuit = solver.find_minimum_circuit(tt)
        self.assertIsNotNone(circuit)
        self.assertTrue(circuit.is_correct(tt))

    def test_xor2(self):
        from mcsp.solvers.sat_solver import MCSPSatSolver
        n = 2
        tt = TruthTable(n, [0, 1, 1, 0])
        solver = MCSPSatSolver(n, max_gates=5, timeout_ms=15000)
        circuit = solver.find_minimum_circuit(tt)
        self.assertIsNotNone(circuit)
        self.assertTrue(circuit.is_correct(tt))

    def test_or2(self):
        from mcsp.solvers.sat_solver import MCSPSatSolver
        n = 2
        tt = TruthTable(n, [0, 1, 1, 1])
        solver = MCSPSatSolver(n, max_gates=3, timeout_ms=10000)
        circuit = solver.find_minimum_circuit(tt)
        self.assertIsNotNone(circuit)
        self.assertTrue(circuit.is_correct(tt))

    def test_size_1_functions(self):
        from mcsp.solvers.sat_solver import MCSPSatSolver
        n = 2
        # AND can be done in 1 gate
        tt = TruthTable(n, [0, 0, 0, 1])
        solver = MCSPSatSolver(n, max_gates=5, timeout_ms=10000)
        sat, circuit = solver.check_size(tt, 1)
        self.assertTrue(sat)


class TestGeneticSolver(unittest.TestCase):

    def test_simple_and(self):
        from mcsp.solvers.genetic_solver import GeneticSolver
        n = 2
        tt = TruthTable(n, [0, 0, 0, 1])
        solver = GeneticSolver(n, population_size=50, max_generations=200, mutation_rate=0.2)
        circuit, stats = solver.solve(tt, max_size=5)
        # Should find a correct circuit
        self.assertTrue(circuit.is_correct(tt))

    def test_simple_or(self):
        from mcsp.solvers.genetic_solver import GeneticSolver
        n = 2
        tt = TruthTable(n, [0, 1, 1, 1])
        solver = GeneticSolver(n, population_size=50, max_generations=200, mutation_rate=0.2)
        circuit, stats = solver.solve(tt, max_size=5)
        self.assertTrue(circuit.is_correct(tt))

    def test_parity3(self):
        from mcsp.solvers.genetic_solver import GeneticSolver
        n = 3
        tt = TruthTable.parity(n)
        solver = GeneticSolver(n, population_size=100, max_generations=500, mutation_rate=0.15)
        circuit, stats = solver.solve(tt, max_size=10)
        # May not always find perfect solution, but should be close
        self.assertGreater(stats['best_fitness'], 0)

    def test_stats_returned(self):
        from mcsp.solvers.genetic_solver import GeneticSolver
        n = 2
        tt = TruthTable(n, [0, 0, 0, 1])
        solver = GeneticSolver(n, population_size=20, max_generations=50)
        circuit, stats = solver.solve(tt)
        self.assertIn('generations', stats)
        self.assertIn('best_fitness', stats)
        self.assertIn('circuit_size', stats)


if __name__ == '__main__':
    unittest.main()
