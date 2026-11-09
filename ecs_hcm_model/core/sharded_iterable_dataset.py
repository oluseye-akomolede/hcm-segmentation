
from torch.utils.data import IterableDataset

class ShardedIterableDataset(IterableDataset):
    def __init__(self, base_dataset, rank=0, world_size=1):
        super().__init__()
        self.base_dataset: IterableDataset = base_dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        iterator = iter(self.base_dataset)
        for i, item in enumerate(iterator):
            if i % self.world_size == self.rank:
                yield item
