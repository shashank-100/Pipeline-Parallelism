import os
import torch
import torch.distributed as dist
from typing import Tuple


def init_distributed():
    """
    Initializes the distributed process group.
    Reads state directly from environment variables set by torchrun.
    """
    # 1. Read Environment Variables (set by torchrun)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    
    # 2. Initialize Group
    # NCCL (NVIDIA Collective Communications Library) is optimized for NVIDIA GPUs
    # Gloo (Meta Communication Library) is a collective 
    # communications library that supports both CPU and GPU
    dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo',
                            rank=rank,
                            world_size=world_size)

    return rank, world_size, device


class PipelineCommunication:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.prev_rank = self.rank - 1 if self.rank != 0 else None 
        self.next_rank = self.rank + 1 if self.rank != self.world_size else None

    def send_forward(self, tensor: torch.Tensor) -> None:
        """Send activation to the next GPU."""
        dist.send(tensor, self.next_rank)

    def recv_forward(self, 
                     shape: Tuple[int, ...],
                     device: torch.device,
                     dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Receive activation from the previous GPU."""
        tensor = torch.zeros(shape, device=device, dtype=dtype)
        dist.recv(tensor, self.prev_rank)
        return tensor

    def send_backward(self, tensor: torch.Tensor) -> None:
        """Send gradients back to the previous GPU."""
        dist.send(tensor, self.prev_rank)

    def recv_backward(self,
                      shape: Tuple[int, ...],
                      device: torch.device,
                      dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Receive gradients from the next GPU."""
        tensor = torch.zeros(shape, device=device, dtype=dtype)
        dist.recv(tensor, self.next_rank)
        return tensor
    
# NOTE
# Here recv and send both are done synchronously but in prod these are done
# Asynchronously but too complex and my skill issue exists .. so deal with it.