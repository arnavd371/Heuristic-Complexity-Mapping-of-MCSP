"""
Neural network models for circuit complexity prediction.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TruthTableMLP(nn.Module):
        """
        Multi-layer perceptron for predicting circuit complexity from truth table.
        Input: 2^n binary features (truth table flattened).
        Output: scalar complexity prediction.
        """

        def __init__(self, n: int):
            super().__init__()
            input_size = 1 << n
            self.network = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.network(x)

    class TruthTableTransformer(nn.Module):
        """
        Transformer-based model that treats the truth table as a sequence.
        Each row of the truth table is an input-output pair token.
        """

        def __init__(self, n: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
            super().__init__()
            self.n = n
            self.d_model = d_model
            # Each token is (n input bits + 1 output bit) = n+1 dimensional
            self.token_embed = nn.Linear(n + 1, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            """
            x: (batch, 2^n) truth table binary features.
            """
            n = self.n
            batch_size = x.shape[0]
            num_rows = 1 << n

            # Build token features: each row r gets [input_bits..., output_bit]
            tokens = []
            for r in range(num_rows):
                input_bits = torch.tensor(
                    [(r >> i) & 1 for i in range(n)], dtype=torch.float32, device=x.device
                )
                input_bits = input_bits.unsqueeze(0).expand(batch_size, -1)
                output_bit = x[:, r:r+1]
                token = torch.cat([input_bits, output_bit], dim=1)
                tokens.append(token)

            # Stack to (batch, seq_len, n+1)
            tokens = torch.stack(tokens, dim=1)

            # Embed tokens
            embedded = self.token_embed(tokens)  # (batch, seq_len, d_model)

            # Apply transformer
            encoded = self.transformer(embedded)  # (batch, seq_len, d_model)

            # Pool over sequence dimension
            pooled = encoded.mean(dim=1)  # (batch, d_model)

            # Predict complexity
            return self.head(pooled)

else:
    class TruthTableMLP:
        def __init__(self, n):
            raise ImportError("torch is required for TruthTableMLP")

    class TruthTableTransformer:
        def __init__(self, n, d_model=64, nhead=4, num_layers=2):
            raise ImportError("torch is required for TruthTableTransformer")
