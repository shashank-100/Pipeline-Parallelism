import torch
import torch.nn as nn


class ShardedMLP(nn.Module):
    def __init__(
            self, 
            dim: int,
            total_layers: int,
            rank: int,
            world_size: int
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        # 1. Calculate how many layers THIS GPU is responsible for
        num_layers = total_layers//world_size

        # 2. Build the local stack of layers
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        if rank == world_size - 1:
            layers.append(nn.Linear(dim, 2))
            self.loss_fn = nn.CrossEntropyLoss()
        self.net = nn.Sequential(*layers)


    def forward(
            self, 
            x: torch.Tensor,
            targets: torch.Tensor = torch.Tensor(0,0)
    ):
        # Run the local chunk of the network
        x = self.net(x)

        # Only the last GPU calculates loss
        if self.rank == self.world_size - 1 and targets is not None:
            x = self.loss_fn(x, targets)

        # Everyone else just returns the hidden states (activations)
        return x