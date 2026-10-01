from __future__ import annotations

import math

import torch


class PriorityReservoir:

    def __init__(
        self,
        capacity: int,
        seed: int = 42,
    ) -> None:

        self.capacity = capacity

        self.generator = (
            torch.Generator(
                device="cpu"
            )
        )

        self.generator.manual_seed(
            seed
        )

        self.keys: (
            torch.Tensor | None
        ) = None

        self.features: (
            torch.Tensor | None
        ) = None

        self.total_seen = 0

    def update(
        self,
        features: torch.Tensor,
    ) -> None:

        features = (
            features
            .detach()
            .cpu()
            .to(
                dtype=torch.float16
            )
        )

        count = features.shape[0]

        keys = torch.rand(
            count,
            generator=self.generator,
        )

        self.total_seen += count

        if self.features is None:

            self.features = features
            self.keys = keys

        else:

            self.features = torch.cat(
                [
                    self.features,
                    features,
                ],
                dim=0,
            )

            self.keys = torch.cat(
                [
                    self.keys,
                    keys,
                ],
                dim=0,
            )

        if (
            self.features.shape[0]
            > self.capacity
        ):

            indices = torch.topk(
                self.keys,
                k=self.capacity,
                largest=True,
            ).indices

            self.features = (
                self.features[
                    indices
                ]
            )

            self.keys = (
                self.keys[
                    indices
                ]
            )

    def get(
        self,
    ) -> torch.Tensor:

        if self.features is None:
            raise RuntimeError(
                "Reservoir is empty."
            )

        return (
            self.features
            .float()
            .contiguous()
        )


class ApproximateGreedyCoreset:

    def __init__(
        self,
        *,
        ratio: float = 0.10,
        max_patches: int = 1200,
        projection_dim: int = 64,
        seed: int = 42,
    ) -> None:

        self.ratio = ratio
        self.max_patches = max_patches
        self.projection_dim = (
            projection_dim
        )
        self.seed = seed

    @torch.no_grad()
    def select(
        self,
        candidates: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:

        total = candidates.shape[0]

        target = max(
            1,
            int(
                round(
                    total
                    * self.ratio
                )
            ),
        )

        target = min(
            target,
            self.max_patches,
            total,
        )

        if target >= total:
            return candidates.float()

        generator = (
            torch.Generator(
                device="cpu"
            )
        )

        generator.manual_seed(
            self.seed
        )

        feature_dim = (
            candidates.shape[1]
        )

        # Gaussian random projection
        projection = torch.randn(
            (
                feature_dim,
                self.projection_dim,
            ),
            generator=generator,
            dtype=torch.float32,
        )

        projection /= math.sqrt(
            self.projection_dim
        )

        projected = (
            candidates.float()
            @ projection
        )

        projected = projected.to(
            device
        )

        start_index = int(
            torch.randint(
                low=0,
                high=total,
                size=(1,),
                generator=generator,
            ).item()
        )

        selected: list[int] = []

        min_distances = torch.full(
            (
                total,
            ),
            float("inf"),
            device=device,
        )

        current_index = (
            start_index
        )

        for _ in range(
            target
        ):

            selected.append(
                current_index
            )

            center = projected[
                current_index
            ]

            distances = (
                (
                    projected
                    - center
                )
                .square()
                .sum(
                    dim=1
                )
            )

            min_distances = (
                torch.minimum(
                    min_distances,
                    distances,
                )
            )

            current_index = int(
                torch.argmax(
                    min_distances
                ).item()
            )

        indices = torch.tensor(
            selected,
            dtype=torch.long,
        )

        return (
            candidates[
                indices
            ]
            .float()
            .contiguous()
        )
