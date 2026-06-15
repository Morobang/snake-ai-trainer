"""
model.py — The Neural Network (DQN)
=====================================
This is the brain. It replaces the Q-table from the browser version.

Instead of a lookup table with 2048 rows, we have a neural network
that takes the 11-value state and outputs 4 Q-values — one per action.

Why a neural net instead of a table?
  - A table can only memorise exact situations it has seen before.
  - A neural net can *generalise* — it learns patterns and applies
    them to situations it has never seen.
  - This is the core idea behind Deep Q-Networks (DQN).

Architecture:
  Input  (11)  →  Hidden (256)  →  Hidden (256)  →  Output (4)
  state signals     ReLU              ReLU           Q-values per action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class DQN(nn.Module):
    """
    Deep Q-Network.

    Takes an 11-dimensional state vector, outputs 4 Q-values.
    The action with the highest Q-value is what the agent should do.
    """

    def __init__(self, input_size: int = 11, hidden_size: int = 256, output_size: int = 4):
        super().__init__()

        # Three fully-connected layers
        # Think of each layer as learning increasingly abstract features:
        #   Layer 1: raw signals → "is this dangerous?"
        #   Layer 2: danger patterns → "what does this mean for strategy?"
        #   Layer 3: strategy → Q-value per action
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — runs the input through all three layers.

        This is called automatically when you do: model(state)
        You never call forward() directly.

        Args:
            x: tensor of shape (batch_size, 11) or (11,) for single state

        Returns:
            tensor of shape (batch_size, 4) — Q-value for each action
        """
        x = F.relu(self.fc1(x))   # layer 1 + activation
        x = F.relu(self.fc2(x))   # layer 2 + activation
        x = self.fc3(x)            # layer 3 — no activation on output
        return x

    # ── CONVENIENCE METHODS ─────────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> int:
        """
        Given a state (numpy array), return the best action index.
        Used during inference — no gradient needed.

        Args:
            state: np.ndarray of shape (11,)

        Returns:
            int — action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.forward(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def save(self, path: str = 'model/weights.pth'):
        """Save model weights to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"✓ Model saved → {path}")

    def load(self, path: str = 'model/weights.pth'):
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()
        print(f"✓ Model loaded ← {path}")

    def export_json(self, path: str = 'model/weights.json'):
        """
        Export weights as JSON so the browser can load and run inference.
        This is the bridge between Python training and the browser demo.
        """
        import json
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.detach().cpu().numpy().tolist()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(weights, f)
        print(f"✓ Weights exported → {path}")

    def count_parameters(self) -> int:
        """How many trainable parameters does this network have?"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── QUICK TEST ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing DQN...\n")

    model = DQN(input_size=11, hidden_size=256, output_size=4)

    print(f"Architecture:")
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")

    # simulate a batch of 32 states (as if from a replay buffer)
    dummy_batch = torch.randn(32, 11)
    q_values    = model(dummy_batch)

    print(f"\nInput shape:  {dummy_batch.shape}   → 32 states, 11 signals each")
    print(f"Output shape: {q_values.shape}  → 32 states, 4 Q-values each")
    print(f"\nSample Q-values for first state:")
    print(f"  UP={q_values[0,0]:.4f}  DOWN={q_values[0,1]:.4f}  "
          f"LEFT={q_values[0,2]:.4f}  RIGHT={q_values[0,3]:.4f}")
    print(f"  Best action: {['UP','DOWN','LEFT','RIGHT'][q_values[0].argmax().item()]}")

    # test single state predict
    single_state = np.zeros(11, dtype=np.float32)
    single_state[10] = 1  # food is to the right
    action = model.predict(single_state)
    print(f"\npredict() on single state → action {action} ({['UP','DOWN','LEFT','RIGHT'][action]})")

    # test save/load
    model.save('/tmp/test_weights.pth')
    model2 = DQN()
    model2.load('/tmp/test_weights.pth')

    # test JSON export
    model.export_json('/tmp/test_weights.json')
    import json
    with open('/tmp/test_weights.json') as f:
        exported = json.load(f)
    print(f"\nExported layers: {list(exported.keys())}")

    print("\n✓ DQN works correctly.")
