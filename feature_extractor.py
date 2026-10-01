from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    Wide_ResNet50_2_Weights,
    wide_resnet50_2,
)
from torchvision.models.feature_extraction import (
    create_feature_extractor,
)


class PatchFeatureExtractor(
    nn.Module,
):

    def __init__(
        self,
        *,
        pretrained: bool = True,
        pool_kernel: int = 3,
    ) -> None:

        super().__init__()

        weights = (
            Wide_ResNet50_2_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = wide_resnet50_2(
            weights=weights,
        )

        self.extractor = (
            create_feature_extractor(
                backbone,
                return_nodes={
                    "layer2": "layer2",
                    "layer3": "layer3",
                },
            )
        )

        self.pool_kernel = (
            pool_kernel
        )

        for parameter in (
            self.extractor.parameters()
        ):
            parameter.requires_grad = False

        self.extractor.eval()

    def train(
        self,
        mode: bool = True,
    ):
        # Backbone은 항상 inference mode.
        super().train(False)
        self.extractor.eval()

        return self

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        features = self.extractor(
            x
        )

        layer2 = features[
            "layer2"
        ]

        layer3 = features[
            "layer3"
        ]

        padding = (
            self.pool_kernel // 2
        )

        layer2 = F.avg_pool2d(
            layer2,
            kernel_size=self.pool_kernel,
            stride=1,
            padding=padding,
        )

        layer3 = F.avg_pool2d(
            layer3,
            kernel_size=self.pool_kernel,
            stride=1,
            padding=padding,
        )

        layer3 = F.interpolate(
            layer3,
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        embedding = torch.cat(
            [
                layer2,
                layer3,
            ],
            dim=1,
        )

        return embedding


def flatten_patch_embeddings(
    embedding: torch.Tensor,
) -> torch.Tensor:

    # B, C, H, W
    # ↓
    # B*H*W, C

    return (
        embedding
        .permute(
            0,
            2,
            3,
            1,
        )
        .reshape(
            -1,
            embedding.shape[1],
        )
    )
