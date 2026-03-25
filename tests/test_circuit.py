"""Tests for Circuit and AIG classes."""
import unittest
from mcsp.core.circuit import (
    Gate, Circuit, AndInverterGraph, GateType, GATE_TRUTH_TABLES
)
from mcsp.core.truth_table import TruthTable


class TestGateEvaluation(unittest.TestCase):

    def test_and_gate(self):
        op = GATE_TRUTH_TABLES[GateType.AND]
        g = Gate(op, 0, 1)
        self.assertEqual(g.evaluate(0, 0), 0)
        self.assertEqual(g.evaluate(0, 1), 0)
        self.assertEqual(g.evaluate(1, 0), 0)
        self.assertEqual(g.evaluate(1, 1), 1)

    def test_or_gate(self):
        op = GATE_TRUTH_TABLES[GateType.OR]
        g = Gate(op, 0, 1)
        self.assertEqual(g.evaluate(0, 0), 0)
        self.assertEqual(g.evaluate(0, 1), 1)
        self.assertEqual(g.evaluate(1, 0), 1)
        self.assertEqual(g.evaluate(1, 1), 1)

    def test_xor_gate(self):
        op = GATE_TRUTH_TABLES[GateType.XOR]
        g = Gate(op, 0, 1)
        self.assertEqual(g.evaluate(0, 0), 0)
        self.assertEqual(g.evaluate(0, 1), 1)
        self.assertEqual(g.evaluate(1, 0), 1)
        self.assertEqual(g.evaluate(1, 1), 0)

    def test_nand_gate(self):
        op = GATE_TRUTH_TABLES[GateType.NAND]
        g = Gate(op, 0, 1)
        self.assertEqual(g.evaluate(0, 0), 1)
        self.assertEqual(g.evaluate(1, 1), 0)

    def test_nor_gate(self):
        op = GATE_TRUTH_TABLES[GateType.NOR]
        g = Gate(op, 0, 1)
        self.assertEqual(g.evaluate(0, 0), 1)
        self.assertEqual(g.evaluate(0, 1), 0)

    def test_not_gate(self):
        op = GATE_TRUTH_TABLES[GateType.NOT]
        g = Gate(op, 0, 0)
        self.assertEqual(g.evaluate(0, 0), 1)
        self.assertEqual(g.evaluate(1, 0), 0)


class TestCircuitEvaluation(unittest.TestCase):

    def _make_and_circuit(self, n=2):
        c = Circuit(n)
        c.add_gate(GATE_TRUTH_TABLES[GateType.AND], 0, 1)
        return c

    def _make_or_circuit(self, n=2):
        c = Circuit(n)
        c.add_gate(GATE_TRUTH_TABLES[GateType.OR], 0, 1)
        return c

    def _make_xor_circuit(self, n=2):
        c = Circuit(n)
        c.add_gate(GATE_TRUTH_TABLES[GateType.XOR], 0, 1)
        return c

    def test_and_evaluate(self):
        c = self._make_and_circuit()
        self.assertEqual(c.evaluate([0, 0]), 0)
        self.assertEqual(c.evaluate([0, 1]), 0)
        self.assertEqual(c.evaluate([1, 0]), 0)
        self.assertEqual(c.evaluate([1, 1]), 1)

    def test_or_evaluate(self):
        c = self._make_or_circuit()
        self.assertEqual(c.evaluate([0, 0]), 0)
        self.assertEqual(c.evaluate([0, 1]), 1)

    def test_xor_evaluate(self):
        c = self._make_xor_circuit()
        self.assertEqual(c.evaluate([0, 1]), 1)
        self.assertEqual(c.evaluate([1, 1]), 0)

    def test_circuit_size(self):
        c = self._make_and_circuit()
        self.assertEqual(c.size, 1)

    def test_add_gate_returns_wire_idx(self):
        c = Circuit(2)
        wire = c.add_gate(GATE_TRUTH_TABLES[GateType.AND], 0, 1)
        self.assertEqual(wire, 2)

    def test_wrong_input_count(self):
        c = Circuit(2)
        with self.assertRaises(ValueError):
            c.evaluate([1])


class TestTruthTableComputation(unittest.TestCase):

    def test_and_truth_table(self):
        c = Circuit(2)
        c.add_gate(GATE_TRUTH_TABLES[GateType.AND], 0, 1)
        tt = c.compute_truth_table()
        self.assertEqual(tt.to_list(), [0, 0, 0, 1])

    def test_or_truth_table(self):
        c = Circuit(2)
        c.add_gate(GATE_TRUTH_TABLES[GateType.OR], 0, 1)
        tt = c.compute_truth_table()
        self.assertEqual(tt.to_list(), [0, 1, 1, 1])

    def test_xor_truth_table(self):
        c = Circuit(2)
        c.add_gate(GATE_TRUTH_TABLES[GateType.XOR], 0, 1)
        tt = c.compute_truth_table()
        self.assertEqual(tt.to_list(), [0, 1, 1, 0])

    def test_is_correct(self):
        c = Circuit(2)
        c.add_gate(GATE_TRUTH_TABLES[GateType.AND], 0, 1)
        target = TruthTable(2, [0, 0, 0, 1])
        self.assertTrue(c.is_correct(target))

    def test_is_incorrect(self):
        c = Circuit(2)
        c.add_gate(GATE_TRUTH_TABLES[GateType.AND], 0, 1)
        target = TruthTable(2, [0, 1, 1, 0])
        self.assertFalse(c.is_correct(target))


class TestAIG(unittest.TestCase):

    def test_and_aig(self):
        aig = AndInverterGraph(2)
        node = aig.add_and_node(0, False, 1, False)
        aig.set_output(node, False)
        self.assertEqual(aig.evaluate([0, 0]), 0)
        self.assertEqual(aig.evaluate([0, 1]), 0)
        self.assertEqual(aig.evaluate([1, 0]), 0)
        self.assertEqual(aig.evaluate([1, 1]), 1)

    def test_nand_aig(self):
        aig = AndInverterGraph(2)
        node = aig.add_and_node(0, False, 1, False)
        aig.set_output(node, True)  # negate output
        self.assertEqual(aig.evaluate([1, 1]), 0)
        self.assertEqual(aig.evaluate([0, 1]), 1)

    def test_aig_truth_table(self):
        aig = AndInverterGraph(2)
        node = aig.add_and_node(0, False, 1, False)
        aig.set_output(node, False)
        tt = aig.compute_truth_table()
        self.assertEqual(tt.to_list(), [0, 0, 0, 1])

    def test_from_circuit(self):
        c = Circuit(2)
        c.add_gate(GATE_TRUTH_TABLES[GateType.AND], 0, 1)
        aig = AndInverterGraph.from_circuit(c)
        tt_circuit = c.compute_truth_table()
        tt_aig = aig.compute_truth_table()
        self.assertEqual(tt_circuit, tt_aig)

    def test_from_circuit_or(self):
        c = Circuit(2)
        c.add_gate(GATE_TRUTH_TABLES[GateType.OR], 0, 1)
        aig = AndInverterGraph.from_circuit(c)
        tt_circuit = c.compute_truth_table()
        tt_aig = aig.compute_truth_table()
        self.assertEqual(tt_circuit, tt_aig)


if __name__ == '__main__':
    unittest.main()
