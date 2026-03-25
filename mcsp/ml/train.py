"""
Training and evaluation for complexity prediction models.
"""
from typing import List, Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


class ComplexityTrainer:
    """Trainer for complexity prediction models."""

    def __init__(self, model, lr: float = 1e-3, device: str = 'cpu'):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for ComplexityTrainer")
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.model.to(device)

    def prepare_data(self, samples: List[Dict[str, Any]]):
        """Convert samples to (X, y) tensors."""
        X = np.array([s['truth_table'] for s in samples], dtype=np.float32)
        y = np.array([s['complexity'] for s in samples], dtype=np.float32).reshape(-1, 1)
        X_tensor = torch.tensor(X, device=self.device)
        y_tensor = torch.tensor(y, device=self.device)
        return X_tensor, y_tensor

    def train(self, samples: List[Dict[str, Any]], epochs: int = 100,
              batch_size: int = 32, val_split: float = 0.2) -> Dict[str, List[float]]:
        """Train the model. Returns training history."""
        X, y = self.prepare_data(samples)
        n = len(samples)
        val_size = int(n * val_split)
        train_size = n - val_size

        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / max(1, len(train_loader))
            history['train_loss'].append(train_loss)

            if val_size > 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val)
                    val_loss = self.criterion(val_pred, y_val).item()
                history['val_loss'].append(val_loss)

        return history

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the model. Returns MAE, MSE, and accuracy."""
        self.model.eval()
        X, y = self.prepare_data(samples)
        with torch.no_grad():
            pred = self.model(X)

        pred_np = pred.cpu().numpy().flatten()
        y_np = y.cpu().numpy().flatten()

        mae = np.mean(np.abs(pred_np - y_np))
        mse = np.mean((pred_np - y_np) ** 2)
        # "Accuracy" within 1 gate
        accuracy = np.mean(np.abs(pred_np - y_np) <= 1.0)

        return {'mae': float(mae), 'mse': float(mse), 'accuracy': float(accuracy)}

    def predict(self, truth_table_list: List[List[int]]) -> List[float]:
        """Predict complexities for a list of truth tables."""
        self.model.eval()
        X = torch.tensor(np.array(truth_table_list, dtype=np.float32), device=self.device)
        with torch.no_grad():
            pred = self.model(X)
        return pred.cpu().numpy().flatten().tolist()

    def save_model(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])


def train_and_evaluate(n: int, num_samples: int = 200, epochs: int = 50):
    """Convenience function to train and evaluate a complexity prediction model."""
    if not TORCH_AVAILABLE:
        print("torch not available, skipping ML training")
        return None

    from mcsp.ml.gnn_model import TruthTableMLP
    from mcsp.ml.data_generation import DatasetGenerator

    print(f"Generating {num_samples} samples for n={n}...")
    gen = DatasetGenerator(n, solver_type='qmc' if n <= 4 else 'genetic')
    samples = gen.generate_dataset(num_samples, seed=42)

    model = TruthTableMLP(n)
    trainer = ComplexityTrainer(model)

    print(f"Training for {epochs} epochs...")
    history = trainer.train(samples, epochs=epochs)

    metrics = trainer.evaluate(samples)
    print(f"Train metrics: MAE={metrics['mae']:.3f}, MSE={metrics['mse']:.3f}, "
          f"Acc±1={metrics['accuracy']:.3f}")

    return trainer, history, metrics
