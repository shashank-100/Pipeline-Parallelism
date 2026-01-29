import time
import torch
import torch.nn as nn
import torch.optim as optim
from config.config import (
    BATCH_SIZE,
    HIDDEN_DIM,
    TOTAL_LAYERS,
    STEPS
)

class FirstPipelineStage(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        layers = []
        for i in range(depth//2):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SecondPipelineStage(nn.Module):
    def __init__(self, dim: int, depth: int):
        super().__init__()
        layers = []
        for i in range(depth//2):
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
torch.manual_seed(42)

cuda = False
if torch.cuda.is_available() and torch.device_count() >= 2:
    cuda = True

if cuda:
    model_part1 = FirstPipelineStage(HIDDEN_DIM, TOTAL_LAYERS).to(torch.cuda.device(0))
    model_part2 = SecondPipelineStage(HIDDEN_DIM, TOTAL_LAYERS).to(torch.cuda.device(1))
else:
    model_part1 = FirstPipelineStage(HIDDEN_DIM, TOTAL_LAYERS)
    model_part2 = SecondPipelineStage(HIDDEN_DIM, TOTAL_LAYERS)

# IMPORTANT: The optimizer must track parameters from BOTH parts
optimizer = optim.Adam(
    list(model_part1.parameters()) + list(model_part2.parameters()), 
    lr=0.001
)

# Generate one fixed batch to overfit
if cuda:
    fixed_input = torch.randn(BATCH_SIZE, HIDDEN_DIM).to(torch.cuda.device(0))
    fixed_target = torch.randint(0, 2, (BATCH_SIZE,)).to(torch.cuda.device(1))
else:
    fixed_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    fixed_target = torch.randint(0, 2, (BATCH_SIZE,))

print('--- Training MLP (Simulated Pipeline) ---')
start_time = time.time()
model_part1.train()
model_part2.train()
for step in range(STEPS):
    optimizer.zero_grad()
    # Simple forward and backward
    hidden = model_part1(fixed_input)
    # In PP, we'd send 'hidden' to the next GPU here.
    # We must ensure 'hidden' is ready to receive gradients later.
    if cuda:
        hidden = hidden.to(torch.cuda.device(1))
    # retain_grad() for inspection: in PP,
    # this is what we'd send backward.
    # Without it, hidden.grad would be None.
    hidden.retain_grad()
    loss = model_part2(hidden, fixed_target)
    loss.backward()
    optimizer.step()
    if step == 0:  # Check once
        print(hidden.requires_grad, hidden.grad is not None, hidden.grad)
    optimizer.step()
    if step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")

duration = time.time() - start_time
print(f"Final Loss: {loss.item():.6f} Time: {duration:.3f}s")

# First Training Output (in case you dont have a GPU to compare)
# --- Training MLP (Simulated Pipeline) ---
# True True tensor([[ 8.2752e-07,  1.3318e-06, -1.0661e-06,  ..., -4.0409e-07,
#           1.2868e-06,  1.1869e-07],
#         [-7.8245e-07, -1.2593e-06,  1.0080e-06,  ...,  3.8209e-07,
#          -1.2167e-06, -1.1223e-07],
#         [ 8.2752e-07,  1.3318e-06, -1.0661e-06,  ..., -4.0409e-07,
#           1.2868e-06,  1.1869e-07],
#         ...,
#         [-7.8245e-07, -1.2593e-06,  1.0080e-06,  ...,  3.8209e-07,
#          -1.2167e-06, -1.1223e-07],
#         [-7.8245e-07, -1.2593e-06,  1.0080e-06,  ...,  3.8209e-07,
#          -1.2167e-06, -1.1222e-07],
#         [-7.8245e-07, -1.2593e-06,  1.0080e-06,  ...,  3.8209e-07,
#          -1.2167e-06, -1.1222e-07]])
# Step 00 | Loss: 0.698788
# Step 05 | Loss: 0.675527
# Step 10 | Loss: 0.615663
# Step 15 | Loss: 0.322652
# Step 20 | Loss: 0.268519
# Step 25 | Loss: 0.240126
# Step 30 | Loss: 0.215348
# Step 35 | Loss: 0.218474
# Step 40 | Loss: 0.202449
# Step 45 | Loss: 0.166487
# Final Loss: 0.091202 Time: 0.363s