"""Tests for TruthTable class."""
import unittest
import numpy as np
from mcsp.core.truth_table import TruthTable


class TestTruthTableConstruction(unittest.TestCase):

    def test_basic_construction(self):
        tt = TruthTable(2)
        self.assertEqual(tt.n, 2)
        self.assertEqual(tt.size, 4)
        for i in range(4):
            self.assertEqual(tt.get_bit(i), 0)

    def test_construction_with_table(self):
        tt = TruthTable(2, [0, 1, 1, 0])
        self.assertEqual(tt.get_bit(0), 0)
        self.assertEqual(tt.get_bit(1), 1)
        self.assertEqual(tt.get_bit(2), 1)
        self.assertEqual(tt.get_bit(3), 0)

    def test_invalid_table_length(self):
        with self.assertRaises(ValueError):
            TruthTable(2, [0, 1])

    def test_set_and_get_bit(self):
        tt = TruthTable(3)
        tt.set_bit(5, 1)
        self.assertEqual(tt.get_bit(5), 1)
        tt.set_bit(5, 0)
        self.assertEqual(tt.get_bit(5), 0)


class TestBitwiseOps(unittest.TestCase):

    def test_and(self):
        tt1 = TruthTable(2, [1, 1, 0, 0])
        tt2 = TruthTable(2, [1, 0, 1, 0])
        result = tt1 & tt2
        self.assertEqual(result.to_list(), [1, 0, 0, 0])

    def test_or(self):
        tt1 = TruthTable(2, [1, 1, 0, 0])
        tt2 = TruthTable(2, [1, 0, 1, 0])
        result = tt1 | tt2
        self.assertEqual(result.to_list(), [1, 1, 1, 0])

    def test_xor(self):
        tt1 = TruthTable(2, [1, 1, 0, 0])
        tt2 = TruthTable(2, [1, 0, 1, 0])
        result = tt1 ^ tt2
        self.assertEqual(result.to_list(), [0, 1, 1, 0])

    def test_not(self):
        tt = TruthTable(2, [1, 0, 1, 0])
        result = ~tt
        self.assertEqual(result.to_list(), [0, 1, 0, 1])


class TestPopcount(unittest.TestCase):

    def test_zero(self):
        tt = TruthTable.zero(3)
        self.assertEqual(tt.popcount(), 0)

    def test_one(self):
        tt = TruthTable.one(3)
        self.assertEqual(tt.popcount(), 8)

    def test_parity(self):
        tt = TruthTable.parity(3)
        self.assertEqual(tt.popcount(), 4)

    def test_hamming_weight_alias(self):
        tt = TruthTable(2, [1, 0, 1, 0])
        self.assertEqual(tt.hamming_weight(), 2)


class TestSpecialFunctions(unittest.TestCase):

    def test_zero(self):
        tt = TruthTable.zero(3)
        self.assertEqual(tt.to_list(), [0] * 8)

    def test_one(self):
        tt = TruthTable.one(3)
        self.assertEqual(tt.to_list(), [1] * 8)

    def test_variable(self):
        # x0: bit 0 of row index
        tt = TruthTable.variable(3, 0)
        expected = [(i >> 0) & 1 for i in range(8)]
        self.assertEqual(tt.to_list(), expected)

    def test_parity(self):
        tt = TruthTable.parity(3)
        expected = [bin(i).count('1') % 2 for i in range(8)]
        self.assertEqual(tt.to_list(), expected)

    def test_majority(self):
        tt = TruthTable.majority(3)
        expected = [1 if bin(i).count('1') >= 2 else 0 for i in range(8)]
        self.assertEqual(tt.to_list(), expected)

    def test_threshold(self):
        tt = TruthTable.threshold(3, 2)
        expected = [1 if bin(i).count('1') >= 2 else 0 for i in range(8)]
        self.assertEqual(tt.to_list(), expected)


class TestFromStringAndArray(unittest.TestCase):

    def test_from_string(self):
        tt = TruthTable.from_string(2, "0110")
        self.assertEqual(tt.to_list(), [0, 1, 1, 0])

    def test_to_array(self):
        tt = TruthTable(2, [0, 1, 1, 0])
        arr = tt.to_array()
        self.assertIsInstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [0, 1, 1, 0])

    def test_round_trip(self):
        original = [0, 1, 0, 1, 1, 0, 1, 0]
        tt = TruthTable(3, original)
        self.assertEqual(tt.to_list(), original)


class TestEquality(unittest.TestCase):

    def test_equal(self):
        tt1 = TruthTable.parity(3)
        tt2 = TruthTable.parity(3)
        self.assertEqual(tt1, tt2)

    def test_not_equal(self):
        tt1 = TruthTable.parity(3)
        tt2 = TruthTable.majority(3)
        self.assertNotEqual(tt1, tt2)


class TestRandom(unittest.TestCase):

    def test_random_construction(self):
        tt = TruthTable.random(4, seed=42)
        self.assertEqual(tt.n, 4)
        self.assertEqual(tt.size, 16)

    def test_random_reproducible(self):
        tt1 = TruthTable.random(3, seed=123)
        tt2 = TruthTable.random(3, seed=123)
        self.assertEqual(tt1, tt2)

    def test_random_different_seeds(self):
        tt1 = TruthTable.random(4, seed=1)
        tt2 = TruthTable.random(4, seed=2)
        # Very likely to be different
        # (could fail by extreme coincidence, but extremely unlikely)
        self.assertNotEqual(tt1.to_list(), tt2.to_list())


if __name__ == '__main__':
    unittest.main()
