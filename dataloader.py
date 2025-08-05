####### Data Loader ##########

import numpy as np
import torch
import os
from ddp import master_process

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        data_root = "gpt2-datasets"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(self.shards) > 0, "No shards found for the split"

        if master_process:
            print(f"Found {len(self.shards)} shards for the split {split}")
        
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_position : self.current_position + (T * B)+1]
        x= (buf[:-1].view(B, T))
        y= (buf[1:].view(B, T))
        self.current_position += T * B * self.num_processes

        if self.current_position + (T * B * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.process_rank * B * T
        return x,y

