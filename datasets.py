from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch
import numpy as np


def make_loaders(batch_size: int = 128,
                 few_shot: int | None = None,
                 noise_std: float = 0.0):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    tf_test = transforms.ToTensor()

    train_set = datasets.CIFAR10(root="data", train=True, download=True,
                                 transform=tf_train)
    test_set = datasets.CIFAR10(root="data", train=False, download=True,
                                transform=tf_test)

    # Few-shot: keep N random examples per class
    if few_shot:
        idx = []
        for c in range(10):
            cls_idx = np.where(np.asarray(train_set.targets) == c)[0]
            idx.extend(np.random.choice(cls_idx, few_shot, replace=False))
        train_set = Subset(train_set, idx)

    # On-the-fly Gaussian noise wrapper
    if noise_std > 0:
        class _Noise(torch.nn.Module):
            def __init__(self, std): super().__init__(); self.std = std
            def forward(self, x): return x + torch.randn_like(x) * self.std
        tf_train.transforms.insert(0, _Noise(noise_std))

    loader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    loader_test = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader_train, loader_test
