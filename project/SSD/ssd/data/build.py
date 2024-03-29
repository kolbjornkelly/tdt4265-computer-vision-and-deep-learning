import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from ssd.data import samplers
from ssd.data.datasets import build_dataset
from ssd.data.transforms import build_transforms, build_target_transform
from ssd.container import Container


class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            targets = Container(
                {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
            )
        else:
            targets = None
        return images, targets, img_ids


def make_data_loader(cfg, is_train=True, augment=False, max_iter=None, start_iter=0):
    target_transform = build_target_transform(cfg) if is_train else None
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    
    if augment:
        # TODO: remove this print
        print("Generating augmented dataset")
        augmentation_transforms = build_transforms(
            cfg, is_train=is_train, augment=augment)
        datasets = build_dataset(
            cfg.DATASET_DIR,
            dataset_list, transform=augmentation_transforms,
            target_transform=target_transform, is_train=is_train)

    else:
        train_transform = build_transforms(cfg, is_train=is_train)
        datasets = build_dataset(
        cfg.DATASET_DIR,
        dataset_list, transform=train_transform,
        target_transform=target_transform, is_train=is_train)

    shuffle = is_train

    data_loaders = []

    for dataset in datasets:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=is_train)
        if max_iter is not None:
            batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter, start_iter=start_iter)

        data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train))
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
