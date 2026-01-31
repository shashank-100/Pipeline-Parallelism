import torch
from communication import init_distributed, PipelineCommunication
from config.config import RANDOM_SIZE

# Cant run this code normally as its a distributed code (requires torchrun)
# This is the error i got hit with :-
# uv run src/ping_pong.py 
# Traceback (most recent call last):
#   File "/home/abhinav/Projects/Pipeline-Parallelism/src/ping_pong.py", line 16, in <module>
#     ping_pong()
#   File "/home/abhinav/Projects/Pipeline-Parallelism/src/ping_pong.py", line 11, in ping_pong
#     rank, world_size, device = init_distributed()
#                                ^^^^^^^^^^^^^^^^^^
#   File "/home/abhinav/Projects/Pipeline-Parallelism/src/communication.py", line 13, in init_distributed
#     rank = int(os.environ["RANK"])
#                ~~~~~~~~~~^^^^^^^^
#   File "<frozen os>", line 714, in __getitem__
# KeyError: 'RANK'

def ping_pong():
    """
    Send a tensor from device rank 0 to device rank 1 
    and print to verify.

    we are running 2 copies of this on cuda:0 and cuda:1
    """
    rank, world_size, device = init_distributed()
    print(rank, world_size, device)

    communication = PipelineCommunication(rank, world_size)

    # if having asynchronous code then to synchronise them use
    # torch.distributed.barrier()

    if rank == 0:
        tensor = torch.rand(RANDOM_SIZE)
        communication.send_forward(tensor)
        print(f'Tensor sent: {tensor}')

    if rank == 1:
        tensor = communication.recv_forward((RANDOM_SIZE,), device)
        print(f'Tensor recieved: {tensor}')


if __name__ == "__main__":
    ping_pong()

# My results (same tensors sent and recieved)
# Tensor sent: tensor([0.4860, 0.9617, 0.0503])
# Tensor recieved: tensor([0.4860, 0.9617, 0.0503])