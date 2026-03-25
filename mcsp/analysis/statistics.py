"""
Statistical analysis of Boolean function complexity.
"""
import math
from typing import List, Dict, Optional, Any
import numpy as np


class ComplexityStats:
    """Statistical analysis of circuit complexity data."""

    def __init__(self, complexities: List[int], function_names: Optional[List[str]] = None):
        self.complexities = list(complexities)
        self.function_names = function_names

    def summary(self) -> Dict[str, Any]:
        if not self.complexities:
            return {}
        c = np.array(self.complexities, dtype=float)
        return {
            'min': float(np.min(c)),
            'max': float(np.max(c)),
            'mean': float(np.mean(c)),
            'std': float(np.std(c)),
            'median': float(np.median(c)),
            'p10': float(np.percentile(c, 10)),
            'p25': float(np.percentile(c, 25)),
            'p75': float(np.percentile(c, 75)),
            'p90': float(np.percentile(c, 90)),
            'count': len(self.complexities),
        }

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.complexities, p))

    def entropy(self) -> float:
        """Shannon entropy of the complexity distribution."""
        if not self.complexities:
            return 0.0
        c = np.array(self.complexities)
        unique, counts = np.unique(c, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs)))

    def compare_functions(self, func_dict: Dict[str, int]) -> Dict[str, Any]:
        """Compare named functions against the random baseline."""
        summary = self.summary()
        result = {}
        for name, complexity in func_dict.items():
            mean = summary.get('mean', 0)
            std = summary.get('std', 1)
            z_score = (complexity - mean) / (std + 1e-10)
            hardness = compute_hardness_index(complexity, int(math.log2(max(1, len(self.complexities)))))
            result[name] = {
                'complexity': complexity,
                'z_score': z_score,
                'hardness_index': hardness,
                'relative_to_mean': complexity - mean,
            }
        return result


def analyze_complexity_landscape(n: int, num_random: int = 1000) -> ComplexityStats:
    """Analyze complexity landscape for n-variable functions via random sampling."""
    from mcsp.core.truth_table import TruthTable
    from mcsp.solvers.quine_mccluskey import QuineMcCluskey

    complexities = []
    if n <= 6:
        qmc = QuineMcCluskey(n)
        for seed in range(num_random):
            tt = TruthTable.random(n, seed=seed)
            c = qmc.estimate_complexity(tt)
            complexities.append(c)
    else:
        from mcsp.solvers.genetic_solver import GeneticSolver
        for seed in range(num_random):
            tt = TruthTable.random(n, seed=seed)
            solver = GeneticSolver(n, population_size=30, max_generations=50)
            circuit, _ = solver.solve(tt)
            complexities.append(circuit.size)

    return ComplexityStats(complexities)


def compute_hardness_index(complexity: int, n: int) -> float:
    """
    Compute relative hardness index in [0, 1].
    Normalized relative to theoretical bounds for n-variable functions.
    """
    # Upper bound: roughly 2^n / n gates for worst-case functions
    upper_bound = max(1, (1 << n) // max(1, n))
    return min(1.0, complexity / upper_bound)
