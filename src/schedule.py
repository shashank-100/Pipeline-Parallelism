import torch
import torch.distributed as dist
from communication import PipelineCommunication
from model import ShardedMLP
from profiler import PipelineProfiler

from typing import Union, Optional


def naive_pipeline_step(
    model: ShardedMLP,
    communication: PipelineCommunication,
    batch: Union[torch.Tensor, int], 
    targets: torch.Tensor,
    hidden_dim: int, 
    device: torch.device,
    profiler: Optional[PipelineProfiler] = None
):
    """
    A single training step using the Naive (Stop-and-Wait) schedule.

    TODOs:
    - Receive input from previous stage if not first stage (requires_grad)
    - Forward batch through model
    - Send output to next stage if not last stage (detach)
    - Perform backward pass:
        - If last stage, compute loss and call backward on it
        - Else, receive grad from next stage and call backward
    - Send grad to previous stage if not first stage
    - Return loss if last stage, else None
    """

    # In sequential (ground-truth) training code,
    # loss = torch.tensor(0.0)
    # is a scalar placeholder. 
    # It’s harmless because it gets overwritten before any use. 
    # I never backprop through it, and it participates in no graph.
    # It’s just an initialization.

    # In pipeline code,
    # loss = torch.Tensor(0,0)
    # constructs a 0×0 empty tensor, which is not a scalar.
    # It has no value, no shape compatible with loss logic, 
    # no connection to the graph, and cannot be used as a placeholder. 
    # It is a shape error, not a placeholder.

    # The pipeline step requires loss = None, not a fake tensor.
    loss = None

    if communication.rank == 0:
        input_data = batch
    else:
        if isinstance(batch, int):
            batch_size = batch
        else:
            batch_size = batch.shape[0]     
        shape = (batch_size, hidden_dim)
        if profiler:
            with profiler.time_block("recv_forward"):
                input_data = communication.recv_forward(shape, device)
        else:
            input_data = communication.recv_forward(shape, device)
        input_data.requires_grad = True

    if profiler:
        with profiler.time_block("compute_forward"):
            output = model(input_data, targets)
    else:
        output = model(input_data, targets)

    if model.rank < model.world_size - 1:
        if profiler:
            with profiler.time_block("send_forward"):
                communication.send_forward(output.detach())
        else:
            communication.send_forward(output.detach())

    if model.rank == model.world_size - 1:
        loss = output
        if profiler:
            with profiler.time_block("compute_backward"):
                loss.backward()
        else:
            loss.backward()
    else:
        if profiler:
            with profiler.time_block("recv_backward"):
                gradients = communication.recv_backward(output.shape, device)
            with profiler.time_block("compute_backward"):
                output.backward(gradients)
        else:
            gradients = communication.recv_backward(output.shape, device)
            output.backward(gradients)

    if model.rank > 0:
        if isinstance(input_data, torch.Tensor) and input_data.grad is not None:
            if profiler:
                with profiler.time_block("send_backward"):
                    communication.send_backward(input_data.grad)
            else:
                communication.send_backward(input_data.grad)
    
    if model.rank == model.world_size - 1:
        return loss


def gpipe_pipeline_step(
        model: ShardedMLP,
        communication: PipelineCommunication,
        batch: Union[torch.Tensor, int],
        targets: torch.Tensor,
        hidden_dim: int,
        chunks: int, 
        device: torch.device,
        profiler: Optional[PipelineProfiler] = None
    ):
    """
    GPipe Schedule: FWD all chunks -> BWD all chunks.
    """

    micro_batches = None
    micro_targets = None

    if model.rank == 0 and isinstance(batch, torch.Tensor):
        micro_batches = torch.chunk(batch, chunks)
        micro_targets = torch.chunk(targets, chunks)
    
    if model.rank == model.world_size - 1 and isinstance(targets, torch.Tensor):
        micro_targets = torch.chunk(targets, chunks)

    input_buffer = []
    output_buffer = []

    for i in range(chunks):
        if model.rank == 0 and micro_batches is not None and micro_targets is not None:
            input_data = micro_batches[i]
            current_target = micro_targets[i]
        else:
            if isinstance(batch, int):
                batch_size = batch // chunks
            else:
                batch_size = batch.shape[0] // chunks
            shape = (batch_size, hidden_dim)
            if profiler:
                with profiler.time_block("recv_forward"):
                    input_data = communication.recv_forward(shape, device)
            else:
                input_data = communication.recv_forward(shape, device)
            input_data.requires_grad = True
            current_target = micro_targets[i] if micro_targets is not None else None

        if profiler:
            with profiler.time_block("compute_forward"):
                output = model(input_data, current_target)
        else:
            output = model(input_data, current_target)

        if model.rank < model.world_size - 1:
            if profiler:
                with profiler.time_block("send_forward"):
                    communication.send_forward(output.detach())
            else:
                communication.send_forward(output.detach())

        input_buffer.append(input_data)
        output_buffer.append(output)

    loss = None

    for i in range(chunks - 1, -1, -1):
        output = output_buffer[i]
        input_data = input_buffer[i]
        
        if model.rank == model.world_size - 1:
            # Accumulate loss from all chunks
            if loss is None:
                loss = output
            else:
                loss = loss + output
            
            # Use retain_graph for all but the last (first in reverse) iteration
            retain = (i > 0)
            if profiler:
                with profiler.time_block("compute_backward"):
                    output.backward(retain_graph=retain)
            else:
                output.backward(retain_graph=retain)
        else:
            if profiler:
                with profiler.time_block("recv_backward"):
                    gradients = communication.recv_backward(output.shape, device)
                with profiler.time_block("compute_backward"):
                    output.backward(gradients)
            else:
                gradients = communication.recv_backward(output.shape, device)
                output.backward(gradients)

        if model.rank > 0:
            if isinstance(input_data, torch.Tensor) and input_data.grad is not None:
                if profiler:
                    with profiler.time_block("send_backward"):
                        communication.send_backward(input_data.grad)
                else:
                    communication.send_backward(input_data.grad)
        
    if model.rank == model.world_size - 1:
        return loss


def onef_oneb_pipeline_step(
        model: ShardedMLP,
        communication: PipelineCommunication,
        batch: Union[torch.Tensor, int],
        targets: torch.Tensor,
        hidden_dim: int,
        chunks: int, 
        device: torch.device,
        profiler: Optional[PipelineProfiler] = None
    ):
    """
    1F1B Schedule: Interleaves Forward and Backward passes in a pipelined manner.
    """
    
    micro_batches = None
    micro_targets = None
    
    if model.rank == 0 and isinstance(batch, torch.Tensor):
        micro_batches = torch.chunk(batch, chunks)
    
    if model.rank == model.world_size - 1 and isinstance(targets, torch.Tensor):
        micro_targets = torch.chunk(targets, chunks)

    input_buffer: list[Optional[torch.Tensor]] = [None] * chunks
    output_buffer: list[Optional[torch.Tensor]] = [None] * chunks
    async_requests = []

    num_warmup = model.world_size - model.rank - 1
    num_1f1b = chunks - num_warmup

    def run_forward(micro_batch_idx):
        if model.rank == 0 and micro_batches is not None:
            input_data = micro_batches[micro_batch_idx]
        else:
            if isinstance(batch, int):
                batch_size = batch // chunks
            else:
                batch_size = batch.shape[0] // chunks
            shape = (batch_size, hidden_dim)
            if profiler:
                with profiler.time_block("recv_forward"):
                    input_data = communication.recv_forward(shape, device)
            else:
                input_data = communication.recv_forward(shape, device)
            input_data.requires_grad = True

        if profiler:
            with profiler.time_block("compute_forward"):
                if model.rank == model.world_size - 1 and micro_targets is not None:
                    output = model(input_data, micro_targets[micro_batch_idx])
                else:
                    output = model(input_data, None)
        else:
            if model.rank == model.world_size - 1 and micro_targets is not None:
                output = model(input_data, micro_targets[micro_batch_idx])
            else:
                output = model(input_data, None)

        if model.rank < model.world_size - 1:
            if profiler:
                with profiler.time_block("send_forward"):
                    req = dist.isend(output.detach(), communication.next_rank)
                    async_requests.append(req)
            else:
                req = dist.isend(output.detach(), communication.next_rank)
                async_requests.append(req)

        input_buffer[micro_batch_idx] = input_data
        output_buffer[micro_batch_idx] = output

    def run_backward(micro_batch_idx):
        input_data = input_buffer[micro_batch_idx]
        output = output_buffer[micro_batch_idx]

        if model.rank == model.world_size - 1:
            loss = output
            if profiler:
                with profiler.time_block("compute_backward"):
                    loss.backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)
        else:
            if profiler:
                with profiler.time_block("recv_backward"):
                    gradients = communication.recv_backward(output.shape, device)
                with profiler.time_block("compute_backward"):
                    output.backward(gradients)
            else:
                gradients = communication.recv_backward(output.shape, device)
                output.backward(gradients)

        if model.rank > 0:
            if input_data.grad is not None:
                if profiler:
                    with profiler.time_block("send_backward"):
                        communication.send_backward(input_data.grad)
                else:
                    communication.send_backward(input_data.grad)

        if model.rank == model.world_size - 1:
            return loss

    loss = None
    if model.rank == model.world_size - 1:
        loss = torch.tensor(0.0, device=device)

    # Warmup: forward only
    for i in range(num_warmup):
        run_forward(i)

    # Steady state: 1 forward + 1 backward
    for i in range(num_1f1b):
        run_forward(i + num_warmup)
        res = run_backward(i)
        if model.rank == model.world_size - 1 and res is not None:
            loss = loss + res

    # Drain: backward only
    for i in range(num_warmup):
        res = run_backward(i + num_1f1b)
        if model.rank == model.world_size - 1 and res is not None:
            loss = loss + res

    # Wait for all async sends
    for req in async_requests:
        req.wait()

    if model.rank == model.world_size - 1:
        return loss