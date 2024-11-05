import torch
from torch.utils.data import DataLoader,Sampler
import math
import os
from sklearn.utils.class_weight import compute_sample_weight
from .csv_dataset import CSVDFDataset

class DistributedWeightedSampler(Sampler):
    """
    Weighted Sampler adapted to be used as Distributed Sampler
    Adapted from https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/8
    """
    def __init__(self, weights, replacement=True, shuffle=False, num_replicas=None, rank=None):
        self.weights = weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(self.weights.size()[0] * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle

    def __iter__(self):
        # g is always created with the same seed
        g = torch.Generator()
        # deterministically shuffle based on epoch (to make torch.multinomial yield different indices)
        if self.shuffle:
              g.manual_seed(self.epoch)
        # do the weighted sampling
        subsample_balanced_indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g)
        # subsample the indices according to rank
        subsample_balanced_indices = subsample_balanced_indices[self.rank:self.total_size:self.num_replicas]
        return iter(subsample_balanced_indices.tolist())

    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch

def get_distributed_weighted_dataloader_from_csv(csv_file, sampling_col, filter_func, transform=None, **dataloader_kwargs): #
    global_rank = int(gr) if (gr:=os.environ.get('SLURM_PROCID', None)) is not None else 0
    world_size = int(nt) if (nt:=os.environ.get('SLURM_NTASKS', None)) is not None else 1
    dataset = CSVDFDataset(csv_file, filter_func, transform)
    targets = dataset.df[sampling_col].values
    weights = compute_sample_weight('balanced', targets)
    sampler = DistributedWeightedSampler(torch.tensor(weights), shuffle=True, num_replicas=world_size, rank=global_rank)
    dataloader = DataLoader(dataset, sampler=sampler, **dataloader_kwargs)
    return dataloader