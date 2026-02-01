import time
import torch
import torch.nn as nn
import torch.optim as optim
from config.config import (
    SEQUENTIAL_BATCH_SIZE,
    SEQUENTIAL_HIDDEN_DIM,
    SEQUENTIAL_TOTAL_LAYERS,
    SEQUENTIAL_STEPS,
    RANDOM_SEED
)

# Multi Layer Perceptron
class MultiLayerPerceptron(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, 2)) # Binary Classification
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return self.loss_fn(logits, targets)
    
# Setup
torch.manual_seed(RANDOM_SEED)

model = MultiLayerPerceptron(SEQUENTIAL_HIDDEN_DIM, SEQUENTIAL_TOTAL_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate one fixed batch to overfit
fixed_input = torch.randn(SEQUENTIAL_BATCH_SIZE, SEQUENTIAL_HIDDEN_DIM)
fixed_target = torch.randint(0, 2, (SEQUENTIAL_BATCH_SIZE,)) # low -> inclusive, high -> exclusive

loss = None

print('--- Training MLP (Ground Truth) ---')
start_time = time.time()
model.train()
for step in range(SEQUENTIAL_STEPS):
    optimizer.zero_grad()
    # Simple forward and backward
    loss = model(fixed_input, fixed_target)
    loss.backward()
    optimizer.step()
    if step % 5 == 0:
        print(f"Step {step} | Loss: {loss:.4f}")

duration = time.time() - start_time

if loss is not None:
    print(f"Final Loss: {loss.item():.6f} Time: {duration:.3f}s")

# First Training Output (in case you dont have a GPU to compare)
# --- Training MLP (Ground Truth) ---
# Step 0 | Loss: 0.6988
# Step 5 | Loss: 0.6839
# Step 10 | Loss: 0.6753
# Step 15 | Loss: 0.6696
# Step 20 | Loss: 0.5181
# Step 25 | Loss: 0.3544
# Step 30 | Loss: 0.2974
# Step 35 | Loss: 0.2742
# Step 40 | Loss: 0.2586
# Step 45 | Loss: 0.2436
# Final Loss: 0.231414 Time: 1.887s