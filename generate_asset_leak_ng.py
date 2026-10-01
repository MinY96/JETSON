from __future__ import annotations

import argparse
import json
import random
import sys
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


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


# ============================================================
# Path
# ============================================================

def resolve_path(
    path: str | Path,
) -> Path:

    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


# ============================================================
# Image IO
# ============================================================

def imread_color(
    path: Path,
) -> np.ndarray | None:

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


def imread_rgba(
    path: Path,
) -> np.ndarray | None:

    data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    if data.size == 0:
        return None

    image = cv2.imdecode(
        data,
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        return None

    if (
        image.ndim != 3
        or image.shape[2] != 4
    ):
        return None

    return image


def imwrite_safe(
    path: Path,
    image: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success, encoded = cv2.imencode(
        ".png",
        image,
    )

    if not success:
        raise RuntimeError(
            f"Failed to encode: {path}"
        )

    encoded.tofile(
        str(path)
    )


# ============================================================
# Collect files
# ============================================================

def collect_images(
    directory: Path,
) -> list[Path]:

    return sorted(
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


# ============================================================
# Config
# ============================================================

def load_anomaly_roi(
    config_path: Path,
    roi_name: str,
) -> tuple[int, int, int, int]:

    with config_path.open(
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
            f"Invalid anomaly_roi: "
            f"{roi_name}"
        )

    return tuple(
        int(value)
        for value in roi
    )


# ============================================================
# Asset transform
# ============================================================

def resize_asset(
    asset: np.ndarray,
    target_width: int,
) -> np.ndarray:

    h, w = asset.shape[:2]

    scale = (
        target_width
        / w
    )

    target_height = max(
        1,
        int(
            round(
                h * scale
            )
        ),
    )

    return cv2.resize(
        asset,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_CUBIC,
    )


def rotate_asset(
    asset: np.ndarray,
    angle: float,
) -> np.ndarray:

    h, w = asset.shape[:2]

    center = (
        w / 2,
        h / 2,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    cos = abs(
        matrix[
            0,
            0
        ]
    )

    sin = abs(
        matrix[
            0,
            1
        ]
    )

    new_w = int(
        h * sin
        + w * cos
    )

    new_h = int(
        h * cos
        + w * sin
    )

    matrix[
        0,
        2
    ] += (
        new_w / 2
        - center[0]
    )

    matrix[
        1,
        2
    ] += (
        new_h / 2
        - center[1]
    )

    return cv2.warpAffine(
        asset,
        matrix,
        (
            new_w,
            new_h,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            0,
            0,
            0,
            0,
        ),
    )


def transform_asset(
    asset: np.ndarray,
    *,
    roi_width: int,
    config: dict,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    dict,
]:

    width_ratio = float(
        rng.uniform(
            config[
                "width_ratio_min"
            ],
            config[
                "width_ratio_max"
            ],
        )
    )

    target_width = max(
        5,
        int(
            roi_width
            * width_ratio
        ),
    )

    asset = resize_asset(
        asset,
        target_width,
    )

    angle = float(
        rng.uniform(
            config[
                "rotation_min"
            ],
            config[
                "rotation_max"
            ],
        )
    )

    asset = rotate_asset(
        asset,
        angle,
    )

    opacity = float(
        rng.uniform(
            config[
                "opacity_min"
            ],
            config[
                "opacity_max"
            ],
        )
    )

    asset[
        :,
        :,
        3
    ] = (
        asset[
            :,
            :,
            3
        ].astype(
            np.float32
        )
        * opacity
    ).clip(
        0,
        255,
    ).astype(
        np.uint8
    )

    blur_sigma = float(
        rng.uniform(
            config[
                "blur_sigma_min"
            ],
            config[
                "blur_sigma_max"
            ],
        )
    )

    if blur_sigma > 0.05:

        for channel in range(4):

            asset[
                :,
                :,
                channel
            ] = cv2.GaussianBlur(
                asset[
                    :,
                    :,
                    channel
                ],
                (
                    0,
                    0,
                ),
                sigmaX=blur_sigma,
                sigmaY=blur_sigma,
            )

    params = {
        "width_ratio":
            width_ratio,

        "angle":
            angle,

        "opacity":
            opacity,

        "blur_sigma":
            blur_sigma,
    }

    return (
        asset,
        params,
    )


# ============================================================
# Placement
# ============================================================

def choose_anchor(
    anomaly_roi: tuple[
        int,
        int,
        int,
        int,
    ],
    zones: list,
    rng: np.random.Generator,
) -> tuple[int, int]:

    x, y, width, height = (
        anomaly_roi
    )

    zone = zones[
        int(
            rng.integers(
                0,
                len(zones),
            )
        )
    ]

    x1, y1, x2, y2 = [
        float(value)
        for value in zone
    ]

    anchor_x = (
        x
        + int(
            rng.uniform(
                x1,
                x2,
            )
            * width
        )
    )

    anchor_y = (
        y
        + int(
            rng.uniform(
                y1,
                y2,
            )
            * height
        )
    )

    return (
        anchor_x,
        anchor_y,
    )


def get_asset_origin(
    asset: np.ndarray,
    anchor_x: int,
    anchor_y: int,
    defect_type: str,
) -> tuple[int, int]:

    h, w = asset.shape[:2]

    if defect_type == "leak":

        # water leak asset는 위쪽 중앙이
        # 호스 접합부에 붙는다고 가정
        x = (
            anchor_x
            - w // 2
        )

        y = (
            anchor_y
            - int(
                h * 0.08
            )
        )

    else:

        # splash는 asset 중심을 anchor에 배치
        x = (
            anchor_x
            - w // 2
        )

        y = (
            anchor_y
            - h // 2
        )

    return (
        x,
        y,
    )


# ============================================================
# Natural alpha blend
# ============================================================

def blend_asset(
    background: np.ndarray,
    asset: np.ndarray,
    origin_x: int,
    origin_y: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    output = (
        background.copy()
    )

    image_h, image_w = (
        background.shape[:2]
    )

    asset_h, asset_w = (
        asset.shape[:2]
    )

    # --------------------------------------------------------
    # Clip
    # --------------------------------------------------------

    bg_x1 = max(
        0,
        origin_x,
    )

    bg_y1 = max(
        0,
        origin_y,
    )

    bg_x2 = min(
        image_w,
        origin_x + asset_w,
    )

    bg_y2 = min(
        image_h,
        origin_y + asset_h,
    )

    if (
        bg_x1 >= bg_x2
        or bg_y1 >= bg_y2
    ):
        return (
            output,
            np.zeros(
                (
                    image_h,
                    image_w,
                ),
                dtype=np.float32,
            ),
        )

    asset_x1 = (
        bg_x1
        - origin_x
    )

    asset_y1 = (
        bg_y1
        - origin_y
    )

    asset_x2 = (
        asset_x1
        + (
            bg_x2
            - bg_x1
        )
    )

    asset_y2 = (
        asset_y1
        + (
            bg_y2
            - bg_y1
        )
    )

    fg = asset[
        asset_y1:asset_y2,
        asset_x1:asset_x2,
        :3,
    ].astype(
        np.float32
    )

    alpha = (
        asset[
            asset_y1:asset_y2,
            asset_x1:asset_x2,
            3,
        ].astype(
            np.float32
        )
        / 255.0
    )

    bg = background[
        bg_y1:bg_y2,
        bg_x1:bg_x2,
    ].astype(
        np.float32
    )

    # --------------------------------------------------------
    # Edge feather
    # --------------------------------------------------------

    alpha = cv2.GaussianBlur(
        alpha,
        (
            0,
            0,
        ),
        sigmaX=0.5,
        sigmaY=0.5,
    )

    alpha = np.clip(
        alpha,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Mild luminance adaptation
    #
    # PNG asset와 CCTV 밝기가 너무 달라
    # 합성 티가 나는 현상을 줄임.
    # --------------------------------------------------------

    valid = (
        alpha > 0.10
    )

    if np.any(
        valid
    ):

        fg_gray = cv2.cvtColor(
            fg.astype(
                np.uint8
            ),
            cv2.COLOR_BGR2GRAY,
        ).astype(
            np.float32
        )

        bg_gray = cv2.cvtColor(
            bg.astype(
                np.uint8
            ),
            cv2.COLOR_BGR2GRAY,
        ).astype(
            np.float32
        )

        fg_mean = float(
            fg_gray[
                valid
            ].mean()
        )

        bg_mean = float(
            bg_gray[
                valid
            ].mean()
        )

        if fg_mean > 1:

            gain = np.clip(
                bg_mean
                / fg_mean,
                0.75,
                1.25,
            )

            # 완전 match하면 물의 highlight가 사라지므로
            # 35%만 scene brightness에 적응
            gain = (
                1.0
                + (
                    gain
                    - 1.0
                )
                * 0.35
            )

            fg *= gain

    fg = np.clip(
        fg,
        0,
        255,
    )

    alpha_3 = alpha[
        ...,
        None
    ]

    blended = (
        bg
        * (
            1.0
            - alpha_3
        )
        + fg
        * alpha_3
    )

    output[
        bg_y1:bg_y2,
        bg_x1:bg_x2,
    ] = np.clip(
        blended,
        0,
        255,
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Full-size mask
    # --------------------------------------------------------

    full_mask = np.zeros(
        (
            image_h,
            image_w,
        ),
        dtype=np.float32,
    )

    full_mask[
        bg_y1:bg_y2,
        bg_x1:bg_x2,
    ] = alpha

    return (
        output,
        full_mask,
    )


# ============================================================
# Main generation
# ============================================================

def generate_roi(
    *,
    roi_name: str,
    settings: dict,
    normal_roi_config: Path,
    output_root: Path,
) -> None:

    generation = settings[
        "generation"
    ]

    split = generation[
        "source_split"
    ]

    samples_per_type = int(
        generation[
            "samples_per_type"
        ]
    )

    seed = int(
        generation[
            "seed"
        ]
    )

    normal_root = (
        PROJECT_ROOT
        / "data"
        / "normal_roi"
        / split
        / roi_name
        / "aligned"
    )

    sources = collect_images(
        normal_root
    )

    if not sources:
        raise RuntimeError(
            f"No aligned images: "
            f"{normal_root}"
        )

    anomaly_roi = (
        load_anomaly_roi(
            normal_roi_config,
            roi_name,
        )
    )

    (
        roi_x,
        roi_y,
        roi_width,
        roi_height,
    ) = anomaly_roi

    rng = np.random.default_rng(
        seed
        + (
            10000
            if roi_name == "right"
            else 0
        )
    )

    records = []

    summary = {}

    for defect_type in generation[
        "defect_types"
    ]:

        asset_dir = resolve_path(
            settings[
                "assets"
            ][
                defect_type
            ][
                "directory"
            ]
        )

        assets = collect_images(
            asset_dir
        )

        if not assets:
            raise RuntimeError(
                f"No assets: "
                f"{asset_dir}"
            )

        zones = (
            settings[
                "placement"
            ][
                roi_name
            ][
                defect_type
            ][
                "zones"
            ]
        )

        transform_config = (
            settings[
                "transform"
            ][
                defect_type
            ]
        )

        generated = 0

        for index in range(
            samples_per_type
        ):

            source_path = Path(
                sources[
                    int(
                        rng.integers(
                            0,
                            len(sources),
                        )
                    )
                ]
            )

            asset_path = Path(
                assets[
                    int(
                        rng.integers(
                            0,
                            len(assets),
                        )
                    )
                ]
            )

            image = imread_color(
                source_path
            )

            asset = imread_rgba(
                asset_path
            )

            if (
                image is None
                or asset is None
            ):
                continue

            asset, params = (
                transform_asset(
                    asset,
                    roi_width=roi_width,
                    config=(
                        transform_config
                    ),
                    rng=rng,
                )
            )

            (
                anchor_x,
                anchor_y,
            ) = choose_anchor(
                anomaly_roi,
                zones,
                rng,
            )

            (
                origin_x,
                origin_y,
            ) = get_asset_origin(
                asset,
                anchor_x,
                anchor_y,
                defect_type,
            )

            (
                synthetic,
                full_mask,
            ) = blend_asset(
                image,
                asset,
                origin_x,
                origin_y,
            )

            crop = synthetic[
                roi_y:
                roi_y + roi_height,

                roi_x:
                roi_x + roi_width,
            ]

            crop_mask = full_mask[
                roi_y:
                roi_y + roi_height,

                roi_x:
                roi_x + roi_width,
            ]

            filename = (
                f"{source_path.stem}"
                f"__{defect_type}"
                f"__{index:03d}.png"
            )

            base = (
                output_root
                / split
                / roi_name
            )

            image_path = (
                base
                / "images"
                / defect_type
                / filename
            )

            aligned_path = (
                base
                / "aligned"
                / defect_type
                / filename
            )

            mask_path = (
                base
                / "masks"
                / defect_type
                / filename
            )

            imwrite_safe(
                image_path,
                crop,
            )

            if settings[
                "output"
            ][
                "save_aligned"
            ]:

                imwrite_safe(
                    aligned_path,
                    synthetic,
                )

            if settings[
                "output"
            ][
                "save_mask"
            ]:

                imwrite_safe(
                    mask_path,
                    (
                        np.clip(
                            crop_mask,
                            0,
                            1,
                        )
                        * 255
                    ).astype(
                        np.uint8
                    ),
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

                    "asset_image":
                        str(
                            asset_path
                        ),

                    "synthetic_image":
                        str(
                            image_path
                        ),

                    "synthetic_aligned":
                        str(
                            aligned_path
                        ),

                    "mask_image":
                        str(
                            mask_path
                        ),

                    "anchor_x":
                        anchor_x,

                    "anchor_y":
                        anchor_y,

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

    base = (
        output_root
        / split
        / roi_name
    )

    base.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        records
    ).to_csv(
        base
        / "metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        base
        / "summary.json"
    ).write_text(
        json.dumps(
            {
                "roi":
                    roi_name,
                "split":
                    split,
                "anomaly_roi":
                    list(
                        anomaly_roi
                    ),
                "generated":
                    summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"[{roi_name.upper()}]"
    )

    for (
        defect_type,
        count,
    ) in summary.items():

        print(
            f"{defect_type:10s}: "
            f"{count}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/"
            "synthetic_ng_realistic.yaml"
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
        "--output-root",
        type=Path,
        default=Path(
            "data/"
            "synthetic_ng/"
            "realistic"
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

    roi_names = (
        [
            "left",
            "right",
        ]
        if args.roi == "all"
        else [
            args.roi
        ]
    )

    for roi_name in roi_names:

        generate_roi(
            roi_name=roi_name,
            settings=settings,
            normal_roi_config=(
                normal_roi_config
            ),
            output_root=(
                output_root
            ),
        )


if __name__ == "__main__":
    main()
