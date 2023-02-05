from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from typing import Optional

class MyDistributedBatchSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        super().__init__( dataset, num_replicas, rank, shuffle,
                         seed, drop_last)
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        chunk_size = self.total_size // self.num_replicas
        indices = indices[self.rank*chunk_size : (self.rank+1)*chunk_size]
        return iter(indices)
