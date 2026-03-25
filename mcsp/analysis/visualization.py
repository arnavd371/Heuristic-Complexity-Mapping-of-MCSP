"""
Visualization utilities for MCSP complexity analysis.
"""
from typing import Dict, List, Optional
import numpy as np


def plot_complexity_distribution(complexities: List[int], title: Optional[str] = None,
                                  save_path: Optional[str] = None):
    """Plot histogram of complexity distribution."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(complexities, bins='auto', color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Circuit Size (# gates)')
    ax.set_ylabel('Count')
    ax.set_title(title or 'Complexity Distribution')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_complexity_vs_n(results_dict: Dict[int, List[int]], save_path: Optional[str] = None):
    """Plot complexity statistics vs number of variables n."""
    import matplotlib.pyplot as plt

    ns = sorted(results_dict.keys())
    means = [np.mean(results_dict[n]) for n in ns]
    stds = [np.std(results_dict[n]) for n in ns]
    maxs = [np.max(results_dict[n]) for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ns, means, yerr=stds, marker='o', label='Mean ± Std', capsize=4)
    ax.plot(ns, maxs, marker='s', linestyle='--', label='Max', color='red')
    ax.set_xlabel('Number of Variables (n)')
    ax.set_ylabel('Circuit Size')
    ax.set_title('Circuit Complexity vs n')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_truth_table(tt, title: Optional[str] = None, save_path: Optional[str] = None):
    """Visualize a truth table as a grid."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if hasattr(tt, 'to_array'):
        arr = tt.to_array()
        n = tt.n
    else:
        arr = np.array(tt)
        n = int(np.log2(len(arr)))

    size = 1 << n
    # Build input columns + output column
    cols = []
    for i in range(n):
        cols.append(np.array([(r >> i) & 1 for r in range(size)]))
    cols.append(arr[:size])

    data = np.column_stack(cols)
    col_labels = [f'x{i}' for i in range(n)] + ['f']

    fig, ax = plt.subplots(figsize=(max(4, n + 2), min(16, size // 2 + 2)))
    cmap = mcolors.ListedColormap(['#FFFFFF', '#4682B4'])
    ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(size))
    ax.set_yticklabels(range(size))
    ax.set_title(title or f'Truth Table (n={n})')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Row')

    # Add text
    for i in range(size):
        for j in range(n + 1):
            val = int(data[i, j])
            ax.text(j, i, str(val), ha='center', va='center',
                    color='white' if val else 'black', fontsize=8)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_circuit(circuit, save_path: Optional[str] = None):
    """Visualize a circuit as a DAG using matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = circuit.n
    num_gates = circuit.size

    fig, ax = plt.subplots(figsize=(max(6, num_gates + 2), 6))

    positions = {}
    for i in range(n):
        positions[i] = (0, i * 2)

    for g_idx, gate in enumerate(circuit.gates):
        wire = n + g_idx
        x = g_idx + 1
        y = (gate.left + gate.right) / 2
        positions[wire] = (x, y)

    # Draw edges
    for g_idx, gate in enumerate(circuit.gates):
        wire = n + g_idx
        x2, y2 = positions[wire]
        for src in [gate.left, gate.right]:
            x1, y1 = positions[src]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='gray'))

    # Draw nodes
    for node_id, (x, y) in positions.items():
        if node_id < n:
            color = '#90EE90'
            label = f'x{node_id}'
        else:
            g_idx = node_id - n
            gate = circuit.gates[g_idx]
            color = '#87CEEB' if node_id < n + num_gates - 1 else '#FFD700'
            label = f'g{g_idx}\n({gate.op:04b})'
        circle = plt.Circle((x, y), 0.4, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=7, zorder=4)

    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
        ax.set_ylim(min(ys) - 1, max(ys) + 1)

    ax.set_title(f'Circuit (n={n}, size={num_gates})')
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training loss curves."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history.get('train_loss', [])) + 1)

    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history and history['val_loss']:
        ax.plot(range(1, len(history['val_loss']) + 1),
                history['val_loss'], label='Val Loss', color='orange')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_complexity_heatmap(n: int, samples: List[Dict], save_path: Optional[str] = None):
    """Plot heatmap of complexity for n=2 functions."""
    import matplotlib.pyplot as plt

    if n != 2:
        # For n != 2, fall back to scatter plot
        complexities = [s['complexity'] for s in samples]
        plot_complexity_distribution(complexities, title=f'Complexity Distribution (n={n})',
                                     save_path=save_path)
        return

    # For n=2: 16 possible functions, index 0..15
    complexities_map = {}
    for s in samples:
        key = tuple(s['truth_table'])
        complexities_map[key] = s['complexity']

    # All 16 functions
    data = np.zeros((4, 4))
    for idx in range(16):
        tt = tuple((idx >> i) & 1 for i in range(4))
        data[idx // 4, idx % 4] = complexities_map.get(tt, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='Complexity')
    ax.set_title(f'Complexity Heatmap (n={n})')
    ax.set_xlabel('Function Index (mod 4)')
    ax.set_ylabel('Function Index (div 4)')

    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{data[i,j]:.0f}', ha='center', va='center', fontsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
