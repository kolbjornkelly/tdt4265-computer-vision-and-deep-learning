from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *
from torchvision import transforms


def build_transforms(cfg, is_train=True, augment=False):

    if augment:
        print("Using data augmentation")
        transform = [
            ConvertFromInts(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            # RandomSampleCrop(),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            # RandomMirror(),
            transforms.RandomCrop(cfg.INPUT.IMAGE_SIZE),
            ToTensor(),
        ]
    elif is_train:
        transform = [
            ConvertFromInts(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            ToTensor(),
        ]

    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
