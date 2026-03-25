# Heuristic Complexity Mapping of MCSP

An end-to-end playground for the **Minimum Circuit Size Problem (MCSP)**. Given a Boolean truth table and a size bound `s`, we explore whether a circuit of size ≤ `s` exists using exact SAT encodings, Quine–McCluskey minimization, genetic heuristics, and lightweight ML models.

**GitHub Pages demo:** https://arnavd371.github.io/Heuristic-Complexity-Mapping-of-MCSP/ (enable Pages with the `docs/` folder as the source).

## What's inside
- **Core primitives:** Bitset-backed truth tables, circuits, and And-Inverter Graphs.
- **Exact solver:** Z3-based SAT encoding with binary search over circuit size.
- **Symbolic minimization:** Quine–McCluskey with Petrick's method for prime implicants.
- **Heuristic search:** Genetic solver targeting circuit correctness and size.
- **Analysis & visualization:** Complexity statistics, hardness index, and plotting helpers.
- **ML hooks:** MLP/Transformer baselines over truth tables plus dataset generation.

## Quickstart
```bash
pip install -r requirements.txt  # z3/torch are optional; tests run without torch
python main.py                   # runs the end-to-end demo suite
```

To exercise individual components:
```bash
python -m pytest                 # run the test suite
python -c "from mcsp.core.truth_table import TruthTable; print(TruthTable.parity(3))"
python -c "from mcsp.solvers.sat_solver import MCSPSatSolver; print(MCSPSatSolver(2,5).find_minimum_circuit(TruthTable(2,[0,1,1,0])))"
```

## GitHub Pages
1. Open **Settings → Pages** in GitHub and choose **Source: `main` / Folder: `/docs`**.
2. The site will be served at `https://<username>.github.io/Heuristic-Complexity-Mapping-of-MCSP/`.
3. Update `docs/index.html` to tweak copy or add visuals; it is self-contained.
