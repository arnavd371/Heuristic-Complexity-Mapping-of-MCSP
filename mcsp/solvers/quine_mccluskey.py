"""
Quine-McCluskey minimization algorithm for Boolean function minimization.
"""
from typing import List, Optional, Tuple, Set, Dict
from itertools import combinations


class Minterm:
    """A minterm or group of minterms in QMC, with don't-care bits."""

    def __init__(self, bits: str, mask: str, indices: Set[int]):
        """
        bits: binary string, e.g. '0110'
        mask: string with '-' for don't-care positions, e.g. '01-0'
        indices: set of minterm indices covered
        """
        self.bits = bits
        self.mask = mask
        self.indices = set(indices)
        self.used = False

    def can_combine(self, other: 'Minterm') -> bool:
        """Check if two minterms can be combined (differ in exactly one non-masked bit)."""
        if self.mask != other.mask:
            return False
        diff_count = 0
        for a, b, m in zip(self.bits, other.bits, self.mask):
            if m == '-':
                continue
            if a != b:
                diff_count += 1
        return diff_count == 1

    def combine(self, other: 'Minterm') -> 'Minterm':
        """Combine two minterms into one with a don't-care bit."""
        new_bits = []
        new_mask = []
        for a, b, m in zip(self.bits, other.bits, self.mask):
            if m == '-':
                new_bits.append(a)
                new_mask.append('-')
            elif a != b:
                new_bits.append('-')
                new_mask.append('-')
            else:
                new_bits.append(a)
                new_mask.append(m)
        return Minterm(''.join(new_bits), ''.join(new_mask), self.indices | other.indices)

    def covers(self, idx: int) -> bool:
        return idx in self.indices

    def __repr__(self):
        return f"Minterm(bits={self.bits}, mask={self.mask}, indices={self.indices})"

    def __eq__(self, other):
        return self.bits == other.bits and self.mask == other.mask

    def __hash__(self):
        return hash((self.bits, self.mask))


class QuineMcCluskey:
    """Quine-McCluskey Boolean minimization, supporting n up to 6 variables."""

    def __init__(self, n: int):
        if n > 6:
            raise ValueError("QMC supports n up to 6")
        self.n = n

    def minimize(self, truth_table) -> List[str]:
        """
        Return list of prime implicants as binary strings with '-' for don't-cares.
        truth_table: TruthTable object or list/array of 0/1 values.
        """
        if hasattr(truth_table, 'to_list'):
            table = truth_table.to_list()
        else:
            table = list(truth_table)

        ones = [i for i, v in enumerate(table) if v == 1]
        if not ones:
            return []
        if len(ones) == (1 << self.n):
            return ['-' * self.n]

        # Initialize minterms
        minterms = [
            Minterm(
                format(i, f'0{self.n}b'),
                '0' * self.n,
                {i}
            )
            for i in ones
        ]

        prime_implicants: Set[Minterm] = set()
        current_group = minterms

        while current_group:
            next_group = []
            used = set()
            for i in range(len(current_group)):
                for j in range(i + 1, len(current_group)):
                    a, b = current_group[i], current_group[j]
                    if a.can_combine(b):
                        combined = a.combine(b)
                        if combined not in next_group:
                            next_group.append(combined)
                        used.add(i)
                        used.add(j)
            for i, m in enumerate(current_group):
                if i not in used:
                    prime_implicants.add(m)
            current_group = next_group

        return [m.bits for m in prime_implicants]

    def get_essential_primes(self, minterms: List[int], prime_implicants: List[str]) -> List[str]:
        """Return essential prime implicants using Petrick's method."""
        n = self.n
        if not prime_implicants:
            return []

        def pi_covers(pi: str, minterm: int) -> bool:
            bits = format(minterm, f'0{n}b')
            for pb, mb in zip(pi, bits):
                if pb != '-' and pb != mb:
                    return False
            return True

        # Build coverage table
        coverage: Dict[int, List[int]] = {}
        for m in minterms:
            coverage[m] = [i for i, pi in enumerate(prime_implicants) if pi_covers(pi, m)]

        essential = []
        covered = set()

        # Find essential PIs (only one PI covers a minterm)
        for m, covering_pis in coverage.items():
            if len(covering_pis) == 1:
                pi_idx = covering_pis[0]
                if prime_implicants[pi_idx] not in essential:
                    essential.append(prime_implicants[pi_idx])
                    for m2, pis in coverage.items():
                        if pi_idx in pis:
                            covered.add(m2)

        # Petrick's method for remaining minterms
        remaining = [m for m in minterms if m not in covered]
        if remaining:
            remaining_pis = [pi for pi in prime_implicants if pi not in essential]
            # Greedy cover of remaining minterms
            while remaining:
                best_pi = None
                best_count = -1
                for pi in remaining_pis:
                    count = sum(1 for m in remaining if pi_covers(pi, m))
                    if count > best_count:
                        best_count = count
                        best_pi = pi
                if best_pi is None:
                    break
                essential.append(best_pi)
                remaining = [m for m in remaining if not pi_covers(best_pi, m)]
                remaining_pis.remove(best_pi)

        return essential

    def cover_to_circuit(self, tt, n: int):
        """Convert truth table to Circuit implementing SOP (sum of products) form."""
        from mcsp.core.circuit import Circuit, GATE_TRUTH_TABLES, GateType

        if hasattr(tt, 'to_list'):
            table = tt.to_list()
        else:
            table = list(tt)

        ones = [i for i, v in enumerate(table) if v == 1]

        circuit = Circuit(n)
        and_op = GATE_TRUTH_TABLES[GateType.AND]
        or_op = GATE_TRUTH_TABLES[GateType.OR]
        not_op = GATE_TRUTH_TABLES[GateType.NOT]

        if not ones:
            # Constant 0: x AND NOT(x)
            not_wire = circuit.add_gate(not_op, 0, 0)
            circuit.add_gate(and_op, 0, not_wire)
            return circuit

        if len(ones) == (1 << n):
            # Constant 1: x OR NOT(x)
            not_wire = circuit.add_gate(not_op, 0, 0)
            circuit.add_gate(or_op, 0, not_wire)
            return circuit

        prime_implicants = self.minimize(tt)
        if not prime_implicants:
            not_wire = circuit.add_gate(not_op, 0, 0)
            circuit.add_gate(and_op, 0, not_wire)
            return circuit

        essential = self.get_essential_primes(ones, prime_implicants)
        if not essential:
            essential = prime_implicants[:1]

        # Build NOT gates for each input (for negated literals)
        not_wires = {}
        for i in range(n):
            not_wires[i] = circuit.add_gate(not_op, i, i)

        # Build AND gate for each product term
        product_wires = []
        for pi in essential:
            literals = []
            for bit_pos, c in enumerate(pi):
                if c == '1':
                    literals.append(bit_pos)  # positive literal
                elif c == '0':
                    literals.append(-(bit_pos + 1))  # negative literal

            if not literals:
                # All don't-cares: constant 1
                not_w = circuit.add_gate(not_op, 0, 0) if 0 not in not_wires else not_wires[0]
                w = circuit.add_gate(or_op, 0, not_w)
                product_wires.append(w)
            elif len(literals) == 1:
                lit = literals[0]
                if lit >= 0:
                    product_wires.append(lit)
                else:
                    product_wires.append(not_wires[(-lit) - 1])
            else:
                # Build AND tree
                def get_wire(lit):
                    if lit >= 0:
                        return lit
                    else:
                        return not_wires[(-lit) - 1]

                w = circuit.add_gate(and_op, get_wire(literals[0]), get_wire(literals[1]))
                for lit in literals[2:]:
                    w = circuit.add_gate(and_op, w, get_wire(lit))
                product_wires.append(w)

        # Build OR tree
        if len(product_wires) == 1:
            # Add identity-like gate to have at least one "output" gate
            circuit.add_gate(GATE_TRUTH_TABLES[GateType.OR], product_wires[0], product_wires[0])
        else:
            w = circuit.add_gate(or_op, product_wires[0], product_wires[1])
            for pw in product_wires[2:]:
                w = circuit.add_gate(or_op, w, pw)

        return circuit

    def estimate_complexity(self, tt) -> int:
        """Estimate circuit complexity from SOP form."""
        if hasattr(tt, 'to_list'):
            table = tt.to_list()
        else:
            table = list(tt)

        ones = [i for i, v in enumerate(table) if v == 1]
        if not ones or len(ones) == (1 << self.n):
            return 1

        prime_implicants = self.minimize(tt)
        essential = self.get_essential_primes(ones, prime_implicants)

        if not essential:
            return 1

        total = 0
        for pi in essential:
            num_literals = sum(1 for c in pi if c != '-')
            if num_literals > 1:
                total += num_literals - 1  # AND gates for product term
        if len(essential) > 1:
            total += len(essential) - 1  # OR gates

        return max(1, total)
