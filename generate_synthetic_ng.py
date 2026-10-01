from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.synthetic.procedural import (
    generate_defect,
)


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


# ============================================================
# Utility
# ============================================================

def resolve_path(
    path: str | Path,
) -> Path:

    path = Path(
        path
    )

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def imread_safe(
    path: Path,
) -> np.ndarray | None:

    try:

        data = np.fromfile(
            str(path),
            dtype=np.uint8,
        )

        if data.size == 0:
            return None

        return cv2.imdecode(
            data,
            cv2.IMREAD_COLOR,
        )

    except Exception:
        return None


def imwrite_safe(
    path: Path,
    image: np.ndarray,
) -> bool:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        path.suffix.lower()
        or ".png"
    )

    success, encoded = (
        cv2.imencode(
            suffix,
            image,
        )
    )

    if not success:
        return False

    encoded.tofile(
        str(path)
    )

    return True


# ============================================================
# Source parsing
# ============================================================

def parse_video_name(
    path: Path,
) -> str:

    stem = path.stem

    token = "__frame_"

    if token in stem:
        return stem.split(
            token,
            1,
        )[0]

    return stem


def collect_images(
    directory: Path,
) -> list[Path]:

    return sorted(
        path
        for path
        in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


def balanced_sample(
    paths: list[Path],
    count: int,
    seed: int,
) -> list[Path]:

    rng = random.Random(
        seed
    )

    groups: dict[
        str,
        list[Path],
    ] = defaultdict(
        list
    )

    for path in paths:
        groups[
            parse_video_name(
                path
            )
        ].append(
            path
        )

    for values in (
        groups.values()
    ):
        rng.shuffle(
            values
        )

    video_names = list(
        groups.keys()
    )

    rng.shuffle(
        video_names
    )

    result = []

    indices = {
        name: 0
        for name
        in video_names
    }

    while (
        len(result) < count
        and video_names
    ):

        remaining = []

        for video_name in (
            video_names
        ):

            index = indices[
                video_name
            ]

            values = groups[
                video_name
            ]

            if index >= len(
                values
            ):
                continue

            result.append(
                values[
                    index
                ]
            )

            indices[
                video_name
            ] += 1

            if (
                indices[
                    video_name
                ]
                < len(values)
            ):
                remaining.append(
                    video_name
                )

            if len(result) >= count:
                break

        video_names = remaining

    return result


# ============================================================
# Config
# ============================================================

def load_anomaly_roi(
    normal_roi_config: Path,
    roi_name: str,
) -> tuple[
    int,
    int,
    int,
    int,
]:

    with normal_roi_config.open(
        "r",
        encoding="utf-8",
    ) as stream:

        config = yaml.safe_load(
            stream
        )

    roi = (
        config[
            "rois"
        ][
            roi_name
        ][
            "anomaly_roi"
        ]
    )

    if (
        roi is None
        or len(roi) != 4
    ):
        raise ValueError(
            f"Invalid anomaly_roi "
            f"for {roi_name}"
        )

    return tuple(
        int(v)
        for v in roi
    )


# ============================================================
# Crop
# ============================================================

def crop_roi(
    image: np.ndarray,
    roi: tuple[
        int,
        int,
        int,
        int,
    ],
) -> np.ndarray:

    x, y, width, height = roi

    return image[
        y:y + height,
        x:x + width,
    ]


# ============================================================
# Generate
# ============================================================

def generate_roi(
    *,
    roi_name: str,
    settings: dict,
    normal_roi_config: Path,
    normal_root: Path,
    output_root: Path,
    split: str,
    samples_per_type: int,
    defect_types: list[str],
    seed: int,
) -> dict:

    aligned_dir = (
        normal_root
        / split
        / roi_name
        / "aligned"
    )

    if not aligned_dir.exists():

        raise FileNotFoundError(
            f"Aligned images not found: "
            f"{aligned_dir}\n"
            "Run 06_extract_normal_roi.py "
            "with --save-aligned first."
        )

    source_paths = (
        collect_images(
            aligned_dir
        )
    )

    if not source_paths:

        raise RuntimeError(
            f"No aligned images: "
            f"{aligned_dir}"
        )

    anomaly_roi = (
        load_anomaly_roi(
            normal_roi_config,
            roi_name,
        )
    )

    effects_config = (
        settings[
            "effects"
        ]
    )

    save_aligned = bool(
        settings[
            "output"
        ].get(
            "save_aligned",
            True,
        )
    )

    save_mask = bool(
        settings[
            "output"
        ].get(
            "save_mask",
            True,
        )
    )

    roi_output = (
        output_root
        / split
        / roi_name
    )

    records = []

    summary = {}

    for type_index, defect_type in enumerate(
        defect_types
    ):

        type_seed = (
            seed
            + type_index * 10000
            + (
                100000
                if roi_name == "right"
                else 0
            )
        )

        selected = balanced_sample(
            source_paths,
            samples_per_type,
            type_seed,
        )

        generated = 0

        for index, source_path in enumerate(
            selected
        ):

            image = imread_safe(
                source_path
            )

            if image is None:
                continue

            x, y, width, height = (
                anomaly_roi
            )

            image_height, image_width = (
                image.shape[:2]
            )

            if (
                x < 0
                or y < 0
                or x + width > image_width
                or y + height > image_height
            ):
                raise ValueError(
                    f"anomaly_roi out of bounds: "
                    f"{anomaly_roi}, "
                    f"image={image_width}x{image_height}"
                )

            sample_seed = (
                type_seed
                + index
            )

            rng = np.random.default_rng(
                sample_seed
            )

            (
                synthetic,
                mask,
                params,
            ) = generate_defect(
                image,
                defect_type,
                anomaly_roi,
                rng,
                effects_config,
            )

            crop = crop_roi(
                synthetic,
                anomaly_roi,
            )

            mask_crop = crop_roi(
                mask,
                anomaly_roi,
            )

            filename = (
                f"{source_path.stem}"
                f"__{defect_type}"
                f"__{index:03d}.png"
            )

            crop_path = (
                roi_output
                / "images"
                / defect_type
                / filename
            )

            aligned_path = (
                roi_output
                / "aligned"
                / defect_type
                / filename
            )

            mask_path = (
                roi_output
                / "masks"
                / defect_type
                / filename
            )

            if not imwrite_safe(
                crop_path,
                crop,
            ):
                continue

            if save_aligned:

                imwrite_safe(
                    aligned_path,
                    synthetic,
                )

            if save_mask:

                mask_u8 = (
                    np.clip(
                        mask_crop,
                        0,
                        1,
                    )
                    * 255
                ).astype(
                    np.uint8
                )

                imwrite_safe(
                    mask_path,
                    mask_u8,
                )

            records.append(
                {
                    "roi":
                        roi_name,

                    "split":
                        split,

                    "defect_type":
                        defect_type,

                    "source_image":
                        str(
                            source_path
                        ),

                    "synthetic_image":
                        str(
                            crop_path
                        ),

                    "synthetic_aligned":
                        (
                            str(
                                aligned_path
                            )
                            if save_aligned
                            else ""
                        ),

                    "mask_image":
                        (
                            str(
                                mask_path
                            )
                            if save_mask
                            else ""
                        ),

                    "seed":
                        sample_seed,

                    "parameters":
                        json.dumps(
                            params,
                            ensure_ascii=False,
                        ),
                }
            )

            generated += 1

        summary[
            defect_type
        ] = generated

    metadata = pd.DataFrame(
        records
    )

    roi_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        roi_output
        / "metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        roi_output
        / "summary.json"
    ).write_text(
        json.dumps(
            {
                "roi":
                    roi_name,

                "split":
                    split,

                "source_images":
                    len(
                        source_paths
                    ),

                "anomaly_roi":
                    list(
                        anomaly_roi
                    ),

                "generated":
                    summary,

                "total_generated":
                    int(
                        sum(
                            summary.values()
                        )
                    ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 60
    )

    print(
        f"SYNTHETIC NG: "
        f"{roi_name.upper()}"
    )

    print(
        "=" * 60
    )

    print(
        f"Source images : "
        f"{len(source_paths)}"
    )

    for key, value in (
        summary.items()
    ):
        print(
            f"{key:15s}: "
            f"{value}"
        )

    print(
        f"Total         : "
        f"{sum(summary.values())}"
    )

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/synthetic_ng.yaml"
        ),
    )

    parser.add_argument(
        "--normal-roi-config",
        type=Path,
        default=Path(
            "config/normal_roi.yaml"
        ),
    )

    parser.add_argument(
        "--normal-root",
        type=Path,
        default=Path(
            "data/normal_roi"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/synthetic_ng"
        ),
    )

    parser.add_argument(
        "--roi",
        choices=[
            "left",
            "right",
            "all",
        ],
        default="all",
    )

    args = parser.parse_args()

    config_path = resolve_path(
        args.config
    )

    normal_roi_config = (
        resolve_path(
            args.normal_roi_config
        )
    )

    normal_root = resolve_path(
        args.normal_root
    )

    output_root = resolve_path(
        args.output_root
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        settings = yaml.safe_load(
            stream
        )

    generation = settings[
        "generation"
    ]

    split = str(
        generation.get(
            "source_split",
            "val",
        )
    )

    samples_per_type = int(
        generation.get(
            "samples_per_type",
            30,
        )
    )

    defect_types = list(
        generation[
            "defect_types"
        ]
    )

    seed = int(
        generation.get(
            "seed",
            42,
        )
    )

    if args.roi == "all":

        roi_names = [
            "left",
            "right",
        ]

    else:

        roi_names = [
            args.roi
        ]

    summaries = {}

    for roi_name in roi_names:

        summaries[
            roi_name
        ] = generate_roi(
            roi_name=roi_name,
            settings=settings,
            normal_roi_config=(
                normal_roi_config
            ),
            normal_root=(
                normal_root
            ),
            output_root=(
                output_root
            ),
            split=split,
            samples_per_type=(
                samples_per_type
            ),
            defect_types=(
                defect_types
            ),
            seed=seed,
        )

    summary_path = (
        output_root
        / split
        / "summary.json"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summaries,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
