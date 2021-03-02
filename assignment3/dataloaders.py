from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import typing
import numpy as np
np.random.seed(0)

mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)


def load_cifar10(batch_size: int, validation_fraction: float = 0.1, trans_opt: str = '') -> typing.List[torch.utils.data.DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_train = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform_train)

    # Conditional image transformation
    if trans_opt == 'x':
        print("Model X transform")
        # Define transforms
        p = 0.5
        x_transforms = [transforms.RandomCrop((32, 32)),
                        transforms.RandomHorizontalFlip(p),
                        transforms.ColorJitter(),
                        transforms.RandomGrayscale(p)]
        transform_train_x = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomApply(x_transforms, p)
        ])

        # Apply transforms
        data_train_x = datasets.CIFAR10('data/cifar10',
                                        train=True,
                                        download=True,
                                        transform=transform_train_x)

        # Concatenate with base data
        data_train = torch.utils.data.ConcatDataset(
            (data_train, data_train_x))

    elif trans_opt == 'y':
        print("Model Y transform")
        transform_train_y = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    else:
        print("Default transform")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    data_test = datasets.CIFAR10('data/cifar10',
                                 train=False,
                                 download=True,
                                 transform=transform_test)

    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_val, dataloader_test
