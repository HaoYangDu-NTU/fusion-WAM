"""Epoch-addressable order for faithful mid-epoch resume with skipped batches."""

import torch
from torch.utils.data import Sampler


class EpochRandomSampler(Sampler):
    """All ranks share one permutation; Accelerate shards its batches.

    Order depends only on seed/epoch, never on the DataLoader generator state
    captured after a permutation has already been consumed.
    """

    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self), generator=generator).tolist())
