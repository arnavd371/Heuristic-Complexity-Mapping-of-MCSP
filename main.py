#!/usr/bin/env python3
"""
Main demonstration of MCSP Heuristic Complexity Mapping.
Shows: truth tables, circuit synthesis, exact/heuristic solvers, ML prediction, and analysis.
"""
import sys


def demo_truth_tables():
    print("\n=== Truth Table Demo ===")
    from mcsp.core.truth_table import TruthTable

    n = 3
    parity = TruthTable.parity(n)
    print(f"Parity(n={n}): {parity}")

    majority = TruthTable.majority(n)
    print(f"Majority(n={n}): {majority}")

    rnd = TruthTable.random(n, seed=42)
    print(f"Random(n={n}): {rnd}")

    # Test bitwise ops
    x0 = TruthTable.variable(n, 0)
    x1 = TruthTable.variable(n, 1)
    x0_and_x1 = x0 & x1
    print(f"x0 AND x1: {x0_and_x1}")
    print(f"Popcount of parity: {parity.popcount()}")


def demo_qmc_solver():
    print("\n=== Quine-McCluskey Solver Demo ===")
    from mcsp.core.truth_table import TruthTable
    from mcsp.solvers.quine_mccluskey import QuineMcCluskey

    n = 3
    # AND of 3 inputs (only 1 is set)
    tt = TruthTable(n, [0, 0, 0, 0, 0, 0, 0, 1])
    qmc = QuineMcCluskey(n)
    primes = qmc.minimize(tt)
    print(f"AND3 prime implicants: {primes}")

    parity = TruthTable.parity(n)
    primes_parity = qmc.minimize(parity)
    print(f"Parity(3) prime implicants: {primes_parity}")

    circuit = qmc.cover_to_circuit(tt, n)
    print(f"AND3 circuit: size={circuit.size}, correct={circuit.is_correct(tt)}")

    complexity = qmc.estimate_complexity(parity)
    print(f"Parity(3) estimated complexity: {complexity}")


def demo_sat_solver():
    print("\n=== SAT-Based Solver Demo ===")
    try:
        import z3
    except ImportError:
        print("z3 not available, skipping SAT demo")
        return

    from mcsp.core.truth_table import TruthTable
    from mcsp.solvers.sat_solver import MCSPSatSolver

    n = 2
    # XOR of 2 inputs
    xor_tt = TruthTable(n, [0, 1, 1, 0])
    solver = MCSPSatSolver(n, max_gates=5, timeout_ms=10000)

    print(f"Searching for minimum circuit for XOR(n=2)...")
    circuit = solver.find_minimum_circuit(xor_tt)
    if circuit:
        print(f"Found circuit: size={circuit.size}, correct={circuit.is_correct(xor_tt)}")
    else:
        print("No circuit found within size limit")

    # AND of 2 inputs
    and_tt = TruthTable(n, [0, 0, 0, 1])
    circuit2 = solver.find_minimum_circuit(and_tt)
    if circuit2:
        print(f"AND(n=2) circuit: size={circuit2.size}, correct={circuit2.is_correct(and_tt)}")


def demo_genetic_solver():
    print("\n=== Genetic Algorithm Solver Demo ===")
    from mcsp.core.truth_table import TruthTable
    from mcsp.solvers.genetic_solver import GeneticSolver

    n = 3
    parity = TruthTable.parity(n)

    solver = GeneticSolver(n, population_size=80, max_generations=200, mutation_rate=0.15)
    print(f"Solving parity(n={n}) with GA...")
    circuit, stats = solver.solve(parity, max_size=8)

    print(f"Found circuit: size={circuit.size}")
    print(f"Correct: {circuit.is_correct(parity)}")
    print(f"Generations: {stats['generations']}, Best fitness: {stats['best_fitness']:.2f}")


def demo_complexity_analysis():
    print("\n=== Complexity Analysis Demo ===")
    from mcsp.analysis.statistics import analyze_complexity_landscape, compute_hardness_index

    n = 3
    print(f"Analyzing complexity landscape for n={n} (100 random functions)...")
    stats = analyze_complexity_landscape(n, num_random=100)
    summary = stats.summary()

    print(f"Min: {summary['min']}, Max: {summary['max']}, "
          f"Mean: {summary['mean']:.2f}, Std: {summary['std']:.2f}")
    print(f"Entropy: {stats.entropy():.3f}")

    # Compare special functions
    from mcsp.core.truth_table import TruthTable
    from mcsp.solvers.quine_mccluskey import QuineMcCluskey
    qmc = QuineMcCluskey(n)

    func_dict = {
        'parity': qmc.estimate_complexity(TruthTable.parity(n)),
        'majority': qmc.estimate_complexity(TruthTable.majority(n)),
    }
    comparison = stats.compare_functions(func_dict)
    for name, info in comparison.items():
        print(f"  {name}: complexity={info['complexity']}, z_score={info['z_score']:.2f}")


def demo_ml():
    print("\n=== ML Complexity Prediction Demo ===")
    try:
        import torch
    except ImportError:
        print("torch not available, skipping ML demo")
        return

    from mcsp.ml.gnn_model import TruthTableMLP
    from mcsp.ml.data_generation import DatasetGenerator, ComplexityDistribution
    from mcsp.ml.train import ComplexityTrainer

    n = 3
    print(f"Generating 50 samples for n={n}...")
    gen = DatasetGenerator(n, solver_type='qmc')
    samples = gen.generate_dataset(50, seed=42)

    dist = ComplexityDistribution(samples)
    print(f"Complexity distribution: {dist.summary()}")

    model = TruthTableMLP(n)
    trainer = ComplexityTrainer(model)

    print("Training MLP for 20 epochs...")
    history = trainer.train(samples, epochs=20, batch_size=16)

    metrics = trainer.evaluate(samples)
    print(f"Evaluation: MAE={metrics['mae']:.3f}, MSE={metrics['mse']:.3f}")


def main():
    print("=" * 60)
    print("MCSP Heuristic Complexity Mapping - Demonstration")
    print("=" * 60)

    demo_truth_tables()
    demo_qmc_solver()
    demo_sat_solver()
    demo_genetic_solver()
    demo_complexity_analysis()
    demo_ml()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
