"""
Dataset generation for MCSP complexity mapping.
"""
import json
import random
from typing import List, Dict, Any, Optional


class DatasetGenerator:
    """Generates labeled datasets of (truth_table, complexity) pairs."""

    def __init__(self, n: int, solver_type: str = 'genetic', max_size: int = 15):
        self.n = n
        self.solver_type = solver_type
        self.max_size = max_size

    def _compute_complexity(self, tt, seed=None) -> int:
        """Estimate circuit complexity using the configured solver."""
        if self.solver_type == 'qmc' and self.n <= 6:
            from mcsp.solvers.quine_mccluskey import QuineMcCluskey
            qmc = QuineMcCluskey(self.n)
            return qmc.estimate_complexity(tt)
        else:
            from mcsp.solvers.genetic_solver import GeneticSolver
            solver = GeneticSolver(
                self.n,
                population_size=50,
                max_generations=100,
                mutation_rate=0.15,
            )
            circuit, stats = solver.solve(tt, max_size=self.max_size)
            return circuit.size

    def generate_sample(self, seed=None) -> Dict[str, Any]:
        """Generate a single (truth_table, complexity) sample."""
        from mcsp.core.truth_table import TruthTable
        tt = TruthTable.random(self.n, seed=seed)
        complexity = self._compute_complexity(tt, seed=seed)
        return {
            'truth_table': tt.to_list(),
            'complexity': complexity,
            'n': self.n,
        }

    def generate_dataset(self, num_samples: int, output_path: Optional[str] = None,
                         seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate a dataset of samples, including special functions."""
        from mcsp.core.truth_table import TruthTable
        rng = random.Random(seed)
        samples = []

        # Special functions
        specials = [
            ('zero', TruthTable.zero(self.n)),
            ('one', TruthTable.one(self.n)),
            ('parity', TruthTable.parity(self.n)),
        ]
        if self.n >= 3:
            specials.append(('majority', TruthTable.majority(self.n)))
        for i in range(self.n):
            specials.append((f'var_{i}', TruthTable.variable(self.n, i)))

        # AND of all inputs
        and_tt = TruthTable.variable(self.n, 0)
        for i in range(1, self.n):
            and_tt = and_tt & TruthTable.variable(self.n, i)
        specials.append(('and_all', and_tt))

        # OR of all inputs
        or_tt = TruthTable.variable(self.n, 0)
        for i in range(1, self.n):
            or_tt = or_tt | TruthTable.variable(self.n, i)
        specials.append(('or_all', or_tt))

        for name, tt in specials:
            complexity = self._compute_complexity(tt)
            samples.append({
                'truth_table': tt.to_list(),
                'complexity': complexity,
                'n': self.n,
                'name': name,
            })

        # Random samples
        for i in range(num_samples - len(specials)):
            s = int(rng.randint(0, np.iinfo(np.int32).max))
            sample = self.generate_sample(seed=s)
            samples.append(sample)

        if output_path:
            self.save_dataset(samples, output_path)

        return samples

    def save_dataset(self, samples: List[Dict[str, Any]], path: str):
        with open(path, 'w') as f:
            json.dump(samples, f, indent=2)

    def load_dataset(self, path: str) -> List[Dict[str, Any]]:
        with open(path, 'r') as f:
            return json.load(f)


class ComplexityDistribution:
    """Statistical analysis of complexity distributions."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
        self.complexities = [s['complexity'] for s in samples]

    def summary(self) -> Dict[str, Any]:
        import statistics
        if not self.complexities:
            return {}
        c = self.complexities
        return {
            'min': min(c),
            'max': max(c),
            'mean': statistics.mean(c),
            'std': statistics.stdev(c) if len(c) > 1 else 0.0,
            'median': statistics.median(c),
            'p25': sorted(c)[len(c) // 4],
            'p75': sorted(c)[3 * len(c) // 4],
            'count': len(c),
        }

    def histogram(self):
        import numpy as np
        c = np.array(self.complexities)
        counts, bin_edges = np.histogram(c, bins='auto')
        return counts, bin_edges
