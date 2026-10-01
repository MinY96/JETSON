from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path

import torch
import torch.nn.functional as F

from tqdm import tqdm

from .feature_extractor import (
    PatchFeatureExtractor,
    flatten_patch_embeddings,
)
from .memory_bank import (
    ApproximateGreedyCoreset,
    PriorityReservoir,
)


@dataclass
class PatchCoreConfig:

    input_size: int = 256

    pool_kernel: int = 3

    candidate_pool_size: int = 12000

    coreset_ratio: float = 0.10

    max_memory_patches: int = 1200

    projection_dim: int = 64

    query_chunk_size: int = 2048

    seed: int = 42


class PatchCoreModel:

    def __init__(
        self,
        config: PatchCoreConfig,
        device: torch.device,
        *,
        pretrained: bool = True,
    ) -> None:

        self.config = config
        self.device = device

        self.feature_extractor = (
            PatchFeatureExtractor(
                pretrained=pretrained,
                pool_kernel=(
                    config.pool_kernel
                ),
            )
            .to(
                device
            )
            .eval()
        )

        self.memory_bank: (
            torch.Tensor | None
        ) = None

    @torch.no_grad()
    def fit(
        self,
        data_loader,
    ) -> dict:

        reservoir = (
            PriorityReservoir(
                capacity=(
                    self.config
                    .candidate_pool_size
                ),
                seed=self.config.seed,
            )
        )

        feature_grid = None
        embedding_dim = None

        for images, _ in tqdm(
            data_loader,
            desc="Extract train patches",
        ):

            images = images.to(
                self.device,
                non_blocking=True,
            )

            embedding = (
                self.feature_extractor(
                    images
                )
            )

            feature_grid = (
                int(
                    embedding.shape[-2]
                ),
                int(
                    embedding.shape[-1]
                ),
            )

            embedding_dim = int(
                embedding.shape[1]
            )

            patches = (
                flatten_patch_embeddings(
                    embedding
                )
            )

            reservoir.update(
                patches
            )

        candidates = (
            reservoir.get()
        )

        sampler = (
            ApproximateGreedyCoreset(
                ratio=(
                    self.config
                    .coreset_ratio
                ),
                max_patches=(
                    self.config
                    .max_memory_patches
                ),
                projection_dim=(
                    self.config
                    .projection_dim
                ),
                seed=self.config.seed,
            )
        )

        memory_bank = sampler.select(
            candidates,
            device=self.device,
        )

        self.memory_bank = (
            memory_bank
            .cpu()
            .contiguous()
        )

        return {
            "total_patches_seen":
                reservoir.total_seen,

            "candidate_patches":
                int(
                    candidates.shape[0]
                ),

            "memory_patches":
                int(
                    self.memory_bank
                    .shape[0]
                ),

            "embedding_dim":
                embedding_dim,

            "feature_grid":
                feature_grid,
        }

    def _memory_on_device(
        self,
    ) -> torch.Tensor:

        if self.memory_bank is None:
            raise RuntimeError(
                "Memory bank is not initialized."
            )

        return self.memory_bank.to(
            self.device,
            non_blocking=True,
        )

    @torch.no_grad()
    def _nearest_distances(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:

        chunks = []

        chunk_size = (
            self.config
            .query_chunk_size
        )

        for start in range(
            0,
            queries.shape[0],
            chunk_size,
        ):

            query = queries[
                start:start
                + chunk_size
            ]

            distances = torch.cdist(
                query.float(),
                memory.float(),
                p=2.0,
            )

            min_distance = (
                distances
                .min(
                    dim=1
                )
                .values
            )

            chunks.append(
                min_distance
            )

        return torch.cat(
            chunks,
            dim=0,
        )

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        if self.memory_bank is None:
            raise RuntimeError(
                "Model is not fitted."
            )

        images = images.to(
            self.device,
            non_blocking=True,
        )

        embedding = (
            self.feature_extractor(
                images
            )
        )

        batch_size = (
            embedding.shape[0]
        )

        grid_height = (
            embedding.shape[-2]
        )

        grid_width = (
            embedding.shape[-1]
        )

        patches = (
            flatten_patch_embeddings(
                embedding
            )
        )

        memory = (
            self._memory_on_device()
        )

        distances = (
            self._nearest_distances(
                patches,
                memory,
            )
        )

        patch_scores = (
            distances.reshape(
                batch_size,
                grid_height,
                grid_width,
            )
        )

        # 이미지 anomaly score:
        # 가장 비정상적인 patch의 NN distance
        image_scores = (
            patch_scores
            .flatten(
                1
            )
            .amax(
                dim=1
            )
        )

        anomaly_maps = (
            F.interpolate(
                patch_scores.unsqueeze(
                    1
                ),
                size=(
                    images.shape[-2],
                    images.shape[-1],
                ),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(
                1
            )
        )

        return (
            image_scores,
            anomaly_maps,
        )

    def save(
        self,
        path: Path,
        extra: dict | None = None,
    ) -> None:

        if self.memory_bank is None:
            raise RuntimeError(
                "Memory bank is empty."
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        extractor_state = {
            key: value.detach().cpu()
            for key, value
            in self.feature_extractor
            .state_dict()
            .items()
        }

        torch.save(
            {
                "config":
                    asdict(
                        self.config
                    ),

                "memory_bank":
                    self.memory_bank
                    .cpu(),

                "feature_extractor_state":
                    extractor_state,

                "extra":
                    extra or {},
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        device: torch.device,
    ) -> "PatchCoreModel":

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        config = PatchCoreConfig(
            **checkpoint[
                "config"
            ]
        )

        model = cls(
            config,
            device,
            pretrained=False,
        )

        model.feature_extractor.load_state_dict(
            checkpoint[
                "feature_extractor_state"
            ]
        )

        model.memory_bank = (
            checkpoint[
                "memory_bank"
            ]
            .float()
            .contiguous()
        )

        model.feature_extractor.eval()

        return model
