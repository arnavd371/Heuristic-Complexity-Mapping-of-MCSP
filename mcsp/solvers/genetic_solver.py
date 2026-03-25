"""
Genetic Algorithm for circuit synthesis.
"""
import random
import copy
from typing import List, Tuple, Optional, Dict, Any


class Individual:
    """A candidate circuit represented as a list of gate genes."""

    def __init__(self, n: int, genes: List[Tuple[int, int, int]]):
        """
        n: number of inputs
        genes: list of (op, left, right) tuples
        """
        self.n = n
        self.genes = list(genes)
        self.fitness: float = -float('inf')

    def evaluate(self, truth_table) -> float:
        """Evaluate fitness against truth table."""
        from mcsp.core.circuit import Circuit
        circuit = self._to_circuit()
        ct = circuit.compute_truth_table()
        if hasattr(truth_table, 'to_array'):
            target = truth_table.to_array()
            computed = ct.to_array()
        else:
            target = list(truth_table)
            computed = ct.to_list()

        correct = sum(1 for t, c in zip(target, computed) if t == c)
        total = len(target)
        fraction_correct = correct / total
        self.fitness = fraction_correct * 1000.0 - len(self.genes)
        return self.fitness

    def _to_circuit(self):
        from mcsp.core.circuit import Circuit
        circuit = Circuit(self.n)
        for op, left, right in self.genes:
            circuit.add_gate(op, left, right)
        return circuit

    def mutate(self, mutation_rate: float) -> 'Individual':
        """Return a mutated copy."""
        new_genes = []
        for i, (op, left, right) in enumerate(self.genes):
            if random.random() < mutation_rate:
                # Mutate operation
                op = random.randint(0, 15)
            if random.random() < mutation_rate:
                # Mutate left input
                max_wire = self.n + i  # wires available at this gate
                left = random.randint(0, max(0, max_wire - 1))
            if random.random() < mutation_rate:
                # Mutate right input
                max_wire = self.n + i
                right = random.randint(0, max(0, max_wire - 1))
            new_genes.append((op, left, right))

        # Occasionally add or remove a gate
        if random.random() < mutation_rate * 0.3 and len(new_genes) > 1:
            idx = random.randint(0, len(new_genes) - 1)
            new_genes.pop(idx)
            # Fix downstream references
            for j in range(idx, len(new_genes)):
                op2, l2, r2 = new_genes[j]
                l2 = min(l2, self.n + j - 1)
                r2 = min(r2, self.n + j - 1)
                new_genes[j] = (op2, max(0, l2), max(0, r2))

        child = Individual(self.n, new_genes)
        return child

    def crossover(self, other: 'Individual') -> 'Individual':
        """One-point crossover."""
        if not self.genes or not other.genes:
            return Individual(self.n, self.genes or other.genes)

        min_len = min(len(self.genes), len(other.genes))
        point = random.randint(1, min_len)
        child_genes = self.genes[:point] + other.genes[point:]

        # Fix references in child genes
        fixed = []
        for i, (op, l, r) in enumerate(child_genes):
            max_wire = self.n + i - 1
            l = min(l, max(0, max_wire))
            r = min(r, max(0, max_wire))
            fixed.append((op, l, r))

        return Individual(self.n, fixed)

    def __repr__(self):
        return f"Individual(n={self.n}, size={len(self.genes)}, fitness={self.fitness:.2f})"


class GeneticSolver:
    """Genetic Algorithm solver for circuit synthesis."""

    def __init__(self, n: int, population_size: int = 100, max_generations: int = 500,
                 mutation_rate: float = 0.1, target_size: Optional[int] = None):
        self.n = n
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.target_size = target_size

    def _random_individual(self, size: int) -> Individual:
        genes = []
        for i in range(size):
            op = random.randint(0, 15)
            max_wire = self.n + i
            left = random.randint(0, max(0, max_wire - 1))
            right = random.randint(0, max(0, max_wire - 1))
            genes.append((op, left, right))
        return Individual(self.n, genes)

    def _initialize_population(self, size: int) -> List[Individual]:
        pop = []
        for _ in range(self.population_size):
            s = random.randint(1, max(1, size))
            pop.append(self._random_individual(s))
        return pop

    def _tournament_select(self, population: List[Individual], tournament_size: int = 5) -> Individual:
        candidates = random.sample(population, min(tournament_size, len(population)))
        return max(candidates, key=lambda ind: ind.fitness)

    def _fitness(self, individual: Individual, truth_table) -> float:
        return individual.evaluate(truth_table)

    def _evolve_generation(self, population: List[Individual], truth_table) -> List[Individual]:
        new_pop = []
        # Keep elite
        elite_count = max(1, self.population_size // 10)
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        new_pop.extend(sorted_pop[:elite_count])

        while len(new_pop) < self.population_size:
            parent1 = self._tournament_select(population)
            if random.random() < 0.7:
                parent2 = self._tournament_select(population)
                child = parent1.crossover(parent2)
            else:
                child = copy.deepcopy(parent1)

            child = child.mutate(self.mutation_rate)
            if not child.genes:
                child = self._random_individual(random.randint(1, 5))
            self._fitness(child, truth_table)
            new_pop.append(child)

        return new_pop

    def solve(self, truth_table, max_size: int = 15) -> Tuple[object, Dict[str, Any]]:
        """
        Solve using genetic algorithm.
        Returns (best_circuit, stats_dict).
        """
        from mcsp.core.circuit import Circuit

        population = self._initialize_population(max_size)

        # Evaluate initial population
        for ind in population:
            self._fitness(ind, truth_table)

        best_individual = max(population, key=lambda ind: ind.fitness)
        history = {'best_fitness': [], 'avg_fitness': [], 'generation': []}

        for gen in range(self.max_generations):
            population = self._evolve_generation(population, truth_table)
            current_best = max(population, key=lambda ind: ind.fitness)

            if current_best.fitness > best_individual.fitness:
                best_individual = copy.deepcopy(current_best)

            avg_fit = sum(ind.fitness for ind in population) / len(population)
            history['best_fitness'].append(best_individual.fitness)
            history['avg_fitness'].append(avg_fit)
            history['generation'].append(gen)

            # Early termination if perfect solution found
            if best_individual.fitness >= 1000.0 - max_size:
                ct = best_individual._to_circuit()
                if ct.is_correct(truth_table):
                    break

        best_circuit = best_individual._to_circuit()
        stats = {
            'generations': len(history['generation']),
            'best_fitness': best_individual.fitness,
            'history': history,
            'circuit_size': best_individual._to_circuit().size,
        }
        return best_circuit, stats
