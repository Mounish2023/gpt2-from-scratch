

import os
import torch
from torch.distributed import init_process_group

# setup data distributed parallel
# torchrun setups the environment variables for distributed training automatically
# ddp_rank, ddp_local_rank, ddp_world_size
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to the rank
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ.get("RANK"))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK"))
    ddp_world_size = int(os.environ.get("WORLD_SIZE"))
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    
    master_process = True
    # attempt to auto detect the device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    print(f"the device available is {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"
