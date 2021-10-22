import functools
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

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

def get_bare_model(net):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        net = net.module
    return net