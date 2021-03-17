import torch
from torch import nn


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """

    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.conv_kernel = 3
        self.pool_kernel = 2
        self.padding = 1
        self.pool_stride = 2

        self.backbone_1 = nn.sequential(
            nn.Conv2d(
                in_channels=image_channels,
                output_channels=32,
                kernel_size=self.conv_kernel,
                stride=1,
                padding=self.padding
            ),
            nn.MaxPool2d(
                kernel_size=self.pool_kernel,
                stride=self.pool_stride
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                output_channels=64,
                kernel_size=self.conv_kernel,
                stride=1,
                padding=self.padding
            ),
            nn.MaxPool2d(
                kernel_size=self.pool_kernel,
                stride=self.pool_stride
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                output_channels=64,
                kernel_size=self.conv_kernel,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                output_channels=self.output_channels[0],
                kernel_size=self.conv_kernel,
                stride=2,
                padding=self.padding
            )
        )

        self.backbone_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[0],
                output_channels=128,
                kernel_size=self.conv_kernel,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                output_channels=self.output_channels[1],
                kernel_size=self.conv_kernel,
                stride=2,
                padding=self.padding
            )
        )

        self.backbone_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[1],
                output_channels=256,
                kernel_size=self.conv_kernel,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                output_channels=self.output_channels[2],
                kernel_size=self.conv_kernel,
                stride=2,
                padding=self.padding
            )
        )

        self.backbone_4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[2],
                output_channels=128,
                kernel_size=self.conv_kernel,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                output_channels=self.output_channels[3],
                kernel_size=self.conv_kernel,
                stride=2,
                padding=self.padding
            )
        )

        self.backbone_5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[3],
                output_channels=128,
                kernel_size=self.conv_kernel,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                output_channels=self.output_channels[4],
                kernel_size=self.conv_kernel,
                stride=2,
                padding=self.padding
            )
        )

        self.backbone_6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[4],
                output_channels=128,
                kernel_size=self.conv_kernel,
                stride=1,
                padding=self.padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                output_channels=self.output_channels[5],
                kernel_size=self.conv_kernel,
                stride=1,
                padding=0
            )
        )

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        # TODO: fjern denne kommentaren
        out_features = []
        # Feed through network
        out_features.append(self.backbone_1(x))
        out_features.append(self.backbone_2(out_features[-1]))
        out_features.append(self.backbone_3(out_features[-1]))
        out_features.append(self.backbone_4(out_features[-1]))
        out_features.append(self.backbone_5(out_features[-1]))
        out_features.append(self.backbone_6(out_features[-1]))

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
