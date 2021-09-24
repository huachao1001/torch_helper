import functools
import torch.distributed as dist

def get_rank():
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank

def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = get_rank()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
