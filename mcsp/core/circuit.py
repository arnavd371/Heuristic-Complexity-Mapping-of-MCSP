"""
Circuit representations: DAG-based Circuit and And-Inverter Graph (AIG).
"""
from enum import Enum, auto
from typing import List, Optional, Tuple
import itertools


class GateType(Enum):
    AND = auto()
    OR = auto()
    XOR = auto()
    NAND = auto()
    NOR = auto()
    XNOR = auto()
    NOT = auto()
    IDENTITY = auto()


# 4-bit truth table encoding for 2-input gates (indexed by (left, right) in binary)
# Bit b corresponds to input pair (b>>1, b&1): i.e. b=0 -> (0,0), b=1 -> (0,1), b=2 -> (1,0), b=3 -> (1,1)
GATE_TRUTH_TABLES = {
    GateType.AND:      0b1000,  # 1 only when both 1
    GateType.OR:       0b1110,  # 0 only when both 0
    GateType.XOR:      0b0110,  # 1 when inputs differ
    GateType.NAND:     0b0111,  # 0 only when both 1
    GateType.NOR:      0b0001,  # 1 only when both 0
    GateType.XNOR:     0b1001,  # 1 when inputs equal
    GateType.NOT:      0b0011,  # ignores right, inverts left
    GateType.IDENTITY: 0b1100,  # ignores right, passes left
}


class Gate:
    """A single gate in the circuit."""

    def __init__(self, op: int, left: int, right: int):
        """
        op: 4-bit integer encoding the truth table of the gate.
        left, right: wire indices (0..n-1 are inputs, n.. are gate outputs).
        """
        self.op = op  # 4-bit operation
        self.left = left
        self.right = right

    def evaluate(self, l_val: int, r_val: int) -> int:
        """Evaluate gate given left and right input values (0 or 1)."""
        idx = (l_val << 1) | r_val
        return (self.op >> idx) & 1

    def __repr__(self):
        return f"Gate(op={self.op:04b}, left={self.left}, right={self.right})"


class Circuit:
    """DAG-based Boolean circuit."""

    def __init__(self, n: int, gates: Optional[List[Gate]] = None):
        """n: number of primary inputs."""
        self.n = n
        self.gates: List[Gate] = gates if gates is not None else []

    @property
    def size(self) -> int:
        return len(self.gates)

    def add_gate(self, op: int, left: int, right: int) -> int:
        """Add a gate and return its wire index."""
        wire_idx = self.n + len(self.gates)
        self.gates.append(Gate(op, left, right))
        return wire_idx

    def evaluate(self, inputs: List[int]) -> int:
        """Evaluate circuit on given input list; returns output of last gate."""
        if len(inputs) != self.n:
            raise ValueError(f"Expected {self.n} inputs, got {len(inputs)}")
        wire_values = list(inputs)
        for gate in self.gates:
            l = wire_values[gate.left]
            r = wire_values[gate.right]
            wire_values.append(gate.evaluate(l, r))
        return wire_values[-1] if wire_values else 0

    def compute_truth_table(self):
        """Compute and return the TruthTable computed by this circuit."""
        from mcsp.core.truth_table import TruthTable
        tt = TruthTable(self.n)
        for i in range(1 << self.n):
            inputs = [(i >> j) & 1 for j in range(self.n)]
            tt.set_bit(i, self.evaluate(inputs))
        return tt

    def is_correct(self, target) -> bool:
        """Check if circuit computes the target TruthTable."""
        return self.compute_truth_table() == target

    def __repr__(self):
        return f"Circuit(n={self.n}, size={self.size}, gates={self.gates})"


class AndInverterGraph:
    """And-Inverter Graph (AIG) representation."""

    def __init__(self, n: int):
        """n: number of primary inputs."""
        self.n = n
        # Each node is (left_node, left_neg, right_node, right_neg)
        # Inputs are nodes 0..n-1 (virtual)
        self.nodes: List[Tuple[int, bool, int, bool]] = []
        self.output_node: Optional[int] = None
        self.output_neg: bool = False

    def add_and_node(self, left: int, left_neg: bool, right: int, right_neg: bool) -> int:
        """Add AND node, return its index (offset by n for inputs)."""
        idx = len(self.nodes)
        self.nodes.append((left, left_neg, right, right_neg))
        return idx + self.n

    def set_output(self, node: int, neg: bool):
        self.output_node = node
        self.output_neg = neg

    def _eval_node(self, node_idx: int, input_vals: List[int]) -> int:
        if node_idx < self.n:
            return input_vals[node_idx]
        aig_idx = node_idx - self.n
        left, left_neg, right, right_neg = self.nodes[aig_idx]
        lv = self._eval_node(left, input_vals)
        rv = self._eval_node(right, input_vals)
        if left_neg:
            lv = 1 - lv
        if right_neg:
            rv = 1 - rv
        return lv & rv

    def evaluate(self, inputs: List[int]) -> int:
        if self.output_node is None:
            return 0
        val = self._eval_node(self.output_node, inputs)
        return (1 - val) if self.output_neg else val

    def compute_truth_table(self):
        from mcsp.core.truth_table import TruthTable
        tt = TruthTable(self.n)
        for i in range(1 << self.n):
            inputs = [(i >> j) & 1 for j in range(self.n)]
            tt.set_bit(i, self.evaluate(inputs))
        return tt

    @classmethod
    def from_circuit(cls, circuit: Circuit) -> 'AndInverterGraph':
        """Convert a DAG Circuit to an AIG (approximate conversion)."""
        aig = cls(circuit.n)
        # Map wire indices to AIG node indices
        wire_to_aig = {i: i for i in range(circuit.n)}
        wire_to_neg = {i: False for i in range(circuit.n)}

        for gate_idx, gate in enumerate(circuit.gates):
            wire_idx = circuit.n + gate_idx
            op = gate.op
            l = wire_to_aig[gate.left]
            r = wire_to_aig[gate.right]
            l_neg = wire_to_neg[gate.left]
            r_neg = wire_to_neg[gate.right]

            # Convert operation to AIG using De Morgan's laws
            if op == GATE_TRUTH_TABLES[GateType.AND]:
                # AND: a & b
                node = aig.add_and_node(l, l_neg, r, r_neg)
                wire_to_aig[wire_idx] = node
                wire_to_neg[wire_idx] = False
            elif op == GATE_TRUTH_TABLES[GateType.NAND]:
                # NAND: ~(a & b)
                node = aig.add_and_node(l, l_neg, r, r_neg)
                wire_to_aig[wire_idx] = node
                wire_to_neg[wire_idx] = True
            elif op == GATE_TRUTH_TABLES[GateType.OR]:
                # OR: ~(~a & ~b)
                node = aig.add_and_node(l, not l_neg, r, not r_neg)
                wire_to_aig[wire_idx] = node
                wire_to_neg[wire_idx] = True
            elif op == GATE_TRUTH_TABLES[GateType.NOR]:
                # NOR: ~a & ~b
                node = aig.add_and_node(l, not l_neg, r, not r_neg)
                wire_to_aig[wire_idx] = node
                wire_to_neg[wire_idx] = False
            elif op == GATE_TRUTH_TABLES[GateType.NOT]:
                wire_to_aig[wire_idx] = l
                wire_to_neg[wire_idx] = not l_neg
            elif op == GATE_TRUTH_TABLES[GateType.IDENTITY]:
                wire_to_aig[wire_idx] = l
                wire_to_neg[wire_idx] = l_neg
            elif op == GATE_TRUTH_TABLES[GateType.XOR]:
                # XOR: (a | b) & ~(a & b) = ~(~(a|b)) & ~(a&b)
                # = ~(~a & ~b) & ~(a & b)
                and_node = aig.add_and_node(l, l_neg, r, r_neg)      # a & b
                or_node = aig.add_and_node(l, not l_neg, r, not r_neg)  # ~(~a & ~b) = a | b; store as neg
                xor_node = aig.add_and_node(and_node, True, or_node, True)  # ~(a&b) & ~(a|b) = NOR -> negate
                wire_to_aig[wire_idx] = xor_node
                wire_to_neg[wire_idx] = True
            elif op == GATE_TRUTH_TABLES[GateType.XNOR]:
                and_node = aig.add_and_node(l, l_neg, r, r_neg)
                or_node = aig.add_and_node(l, not l_neg, r, not r_neg)
                xor_node = aig.add_and_node(and_node, True, or_node, True)
                wire_to_aig[wire_idx] = xor_node
                wire_to_neg[wire_idx] = False
            else:
                # Generic: just pass through with identity
                wire_to_aig[wire_idx] = l
                wire_to_neg[wire_idx] = l_neg

        # Output is the last gate
        if circuit.gates:
            last_wire = circuit.n + circuit.size - 1
            aig.set_output(wire_to_aig[last_wire], wire_to_neg[last_wire])
        return aig
