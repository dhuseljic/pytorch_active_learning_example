import random
import torch
import numpy as np


class ALDataset:
    def __init__(self, dataset, random_state=None):
        self.dataset = dataset
        self.random_state = random_state

        if random_state:
            random.seed(random_state)
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.unlabeled_idx = range(len(self.dataset))
        self.labeled_idx = []

    @property
    def unlabeled_ds(self):
        return torch.utils.data.Subset(self.dataset, indices=self.unlabeled_idx)

    @property
    def labeled_ds(self):
        return torch.utils.data.Subset(self.dataset, indices=self.labeled_idx)


    def random_init(self, n_samples: int):
        """Randomly buys samples from  the unlabeled pool and adds them to the labeled one."""
        assert len(self.labeled_idx) == 0, 'Pools already initialized.'
        buy_idx = random.sample(self.unlabeled_idx, k=n_samples)
        self.labeled_idx = self.union(self.labeled_idx, buy_idx)
        self.unlabeled_idx = self.diff(self.unlabeled_idx, buy_idx)

    def update_annotations(self, indices: list):
        """
        Args:
            indices (list): List of indices which identify samples of the unlabeled pool.
        """
        buy_idx = [self.unlabeled_idx[idx] for idx in indices]
        self.labeled_idx = self.union(self.labeled_idx, buy_idx)
        self.unlabeled_idx = self.diff(self.unlabeled_idx, buy_idx)

    def union(self, a: list, b: list):
        return list(set(a).union(set(b)))

    def diff(self, a: list, b: list):
        return list(set(a).difference(set(b)))

    def __len__(self):
        return len(self.labeled_idx)
