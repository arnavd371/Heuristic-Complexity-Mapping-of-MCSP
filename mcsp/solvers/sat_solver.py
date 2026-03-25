"""
SAT-based exact MCSP solver using z3.
"""
from typing import Optional, Tuple, List
import time

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


class MCSPSatSolver:
    """Exact MCSP solver using SAT encoding via z3."""

    def __init__(self, n: int, max_gates: int = 20, timeout_ms: int = 30000):
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver is required for MCSPSatSolver")
        self.n = n
        self.max_gates = max_gates
        self.timeout_ms = timeout_ms

    def _encode_circuit(self, s: int, target_table: List[int]):
        """
        Encode a circuit of size s that computes target_table.
        Returns a z3.Solver with all constraints added.
        """
        n = self.n
        num_rows = 1 << n
        num_wires = n + s  # inputs + gates

        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)

        # val[w][r] = value of wire w on row r
        val = [[z3.Bool(f"val_{w}_{r}") for r in range(num_rows)] for w in range(num_wires)]

        # Input wire constraints: wire i has value = bit i of row r
        for i in range(n):
            for r in range(num_rows):
                bit = (r >> i) & 1
                solver.add(val[i][r] == (bit == 1))

        # Gate variables
        # sel_l[g][i] = gate g selects wire i as left input
        # sel_r[g][i] = gate g selects wire i as right input
        # op[g][b] = bit b of the 4-bit operation of gate g
        sel_l = [[z3.Bool(f"sel_l_{g}_{i}") for i in range(n + g)] for g in range(s)]
        sel_r = [[z3.Bool(f"sel_r_{g}_{i}") for i in range(n + g)] for g in range(s)]
        op_bits = [[z3.Bool(f"op_{g}_{b}") for b in range(4)] for g in range(s)]

        for g in range(s):
            gate_wire = n + g
            num_inputs = n + g  # wires available as inputs

            # Exactly one left/right input selected
            if num_inputs > 0:
                solver.add(z3.PbEq([(sel_l[g][i], 1) for i in range(num_inputs)], 1))
                solver.add(z3.PbEq([(sel_r[g][i], 1) for i in range(num_inputs)], 1))

            # Gate value consistency
            for r in range(num_rows):
                # left_val and right_val are the selected input values
                # We use auxiliary variables
                left_val = z3.Bool(f"lv_{g}_{r}")
                right_val = z3.Bool(f"rv_{g}_{r}")

                # left_val = OR over i: (sel_l[g][i] AND val[i][r])
                left_clauses = [z3.And(sel_l[g][i], val[i][r]) for i in range(num_inputs)]
                if left_clauses:
                    solver.add(left_val == z3.Or(*left_clauses))
                else:
                    solver.add(left_val == False)

                right_clauses = [z3.And(sel_r[g][i], val[i][r]) for i in range(num_inputs)]
                if right_clauses:
                    solver.add(right_val == z3.Or(*right_clauses))
                else:
                    solver.add(right_val == False)

                # Gate output: op_bit = op[g][(left<<1)|right]
                # val[gate_wire][r] = op_bits[g][(left_val<<1)|right_val]
                # 4 cases: (F,F)->op[0], (F,T)->op[1], (T,F)->op[2], (T,T)->op[3]
                out = val[gate_wire][r]
                solver.add(out == z3.Or(
                    z3.And(z3.Not(left_val), z3.Not(right_val), op_bits[g][0]),
                    z3.And(z3.Not(left_val), right_val, op_bits[g][1]),
                    z3.And(left_val, z3.Not(right_val), op_bits[g][2]),
                    z3.And(left_val, right_val, op_bits[g][3]),
                ))

        # Output constraint: last gate matches target
        if s > 0:
            output_wire = n + s - 1
            for r in range(num_rows):
                expected = target_table[r] == 1
                solver.add(val[output_wire][r] == expected)

        return solver, val, sel_l, sel_r, op_bits

    def check_size(self, truth_table, s: int) -> Tuple[bool, Optional[object]]:
        """Check if there exists a circuit of size s computing truth_table."""
        from mcsp.core.circuit import Circuit, Gate
        if hasattr(truth_table, 'to_list'):
            table = truth_table.to_list()
        else:
            table = list(truth_table)

        if s == 0:
            # Check if truth table is constant 0 or 1 -- no gates needed (not really valid circuit)
            return False, None

        solver, val, sel_l, sel_r, op_bits = self._encode_circuit(s, table)
        result = solver.check()

        if result == z3.sat:
            model = solver.model()
            circuit = self.extract_circuit(model, s, self.n, sel_l, sel_r, op_bits)
            return True, circuit
        return False, None

    def extract_circuit(self, model, s: int, n: int, sel_l, sel_r, op_bits) -> object:
        """Extract Circuit from a satisfying z3 model."""
        from mcsp.core.circuit import Circuit, Gate

        circuit = Circuit(n)
        for g in range(s):
            # Find selected left input
            num_inputs_l = n + g
            left_wire = 0
            for i in range(num_inputs_l):
                if z3.is_true(model.evaluate(sel_l[g][i])):
                    left_wire = i
                    break

            # Find selected right input
            right_wire = 0
            for i in range(num_inputs_l):
                if z3.is_true(model.evaluate(sel_r[g][i])):
                    right_wire = i
                    break

            # Extract operation bits
            op = 0
            for b in range(4):
                if z3.is_true(model.evaluate(op_bits[g][b])):
                    op |= (1 << b)

            circuit.add_gate(op, left_wire, right_wire)

        return circuit

    def find_minimum_circuit(self, truth_table) -> Optional[object]:
        """Find minimum-size circuit via binary search on size."""
        # Try sizes from 1 to max_gates
        lo, hi = 1, self.max_gates
        best_circuit = None

        # First find an upper bound
        for s in range(lo, hi + 1):
            sat, circuit = self.check_size(truth_table, s)
            if sat:
                best_circuit = circuit
                hi = s - 1
                break

        if best_circuit is None:
            return None

        # Binary search for minimum
        while lo <= hi:
            mid = (lo + hi) // 2
            sat, circuit = self.check_size(truth_table, mid)
            if sat:
                best_circuit = circuit
                hi = mid - 1
            else:
                lo = mid + 1

        return best_circuit
