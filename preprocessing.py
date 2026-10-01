from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def imread_rgb(
    path: Path,
) -> np.ndarray:

    data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


class AspectPadPreprocessor:

    def __init__(
        self,
        size: int = 256,
    ) -> None:

        self.size = size

    def __call__(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:

        height, width = image.shape[:2]

        scale = min(
            self.size / width,
            self.size / height,
        )

        resized_width = max(
            1,
            int(round(width * scale)),
        )

        resized_height = max(
            1,
            int(round(height * scale)),
        )

        resized = cv2.resize(
            image,
            (
                resized_width,
                resized_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        # ImageNet mean으로 padding
        # normalize 후 padding 영역이 거의 0이 되도록 함.
        fill = np.round(
            IMAGENET_MEAN * 255.0
        ).astype(
            np.uint8
        )

        canvas = np.empty(
            (
                self.size,
                self.size,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[:] = fill

        x = (
            self.size
            - resized_width
        ) // 2

        y = (
            self.size
            - resized_height
        ) // 2

        canvas[
            y:y + resized_height,
            x:x + resized_width,
        ] = resized

        tensor = (
            torch.from_numpy(
                canvas.copy()
            )
            .permute(
                2,
                0,
                1,
            )
            .float()
            / 255.0
        )

        mean = torch.tensor(
            IMAGENET_MEAN,
            dtype=torch.float32,
        ).view(
            3,
            1,
            1,
        )

        std = torch.tensor(
            IMAGENET_STD,
            dtype=torch.float32,
        ).view(
            3,
            1,
            1,
        )

        tensor = (
            tensor - mean
        ) / std

        return tensor


class NormalImageDataset(
    Dataset,
):

    def __init__(
        self,
        image_dir: Path,
        input_size: int = 256,
    ) -> None:

        self.image_dir = image_dir

        self.paths = sorted(
            [
                path
                for path
                in image_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]
        )

        if not self.paths:
            raise RuntimeError(
                f"No images found: {image_dir}"
            )

        self.preprocessor = (
            AspectPadPreprocessor(
                input_size
            )
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.paths
        )

    def __getitem__(
        self,
        index: int,
    ):

        path = self.paths[
            index
        ]

        image = imread_rgb(
            path
        )

        tensor = self.preprocessor(
            image
        )

        return (
            tensor,
            str(path),
        )
