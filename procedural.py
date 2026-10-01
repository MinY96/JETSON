from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _clip_mask(
    mask: np.ndarray,
) -> np.ndarray:
    return np.clip(
        mask,
        0.0,
        1.0,
    ).astype(np.float32)


def _random_point(
    width: int,
    height: int,
    rng: np.random.Generator,
    margin_ratio: float = 0.08,
) -> tuple[int, int]:

    margin_x = max(
        1,
        int(width * margin_ratio),
    )

    margin_y = max(
        1,
        int(height * margin_ratio),
    )

    x = int(
        rng.integers(
            margin_x,
            max(
                margin_x + 1,
                width - margin_x,
            ),
        )
    )

    y = int(
        rng.integers(
            margin_y,
            max(
                margin_y + 1,
                height - margin_y,
            ),
        )
    )

    return x, y


# ============================================================
# Droplet
# ============================================================

def generate_droplet_mask(
    height: int,
    width: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict]:

    mask = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    count = int(
        rng.integers(
            int(config["count_min"]),
            int(config["count_max"]) + 1,
        )
    )

    base_size = min(
        height,
        width,
    )

    droplets = []

    for _ in range(count):

        x, y = _random_point(
            width,
            height,
            rng,
        )

        radius = int(
            rng.uniform(
                float(
                    config[
                        "radius_min_ratio"
                    ]
                ),
                float(
                    config[
                        "radius_max_ratio"
                    ]
                ),
            )
            * base_size
        )

        radius = max(
            2,
            radius,
        )

        aspect = float(
            rng.uniform(
                0.65,
                1.35,
            )
        )

        angle = float(
            rng.uniform(
                -40,
                40,
            )
        )

        axis_x = radius

        axis_y = max(
            2,
            int(
                radius * aspect
            ),
        )

        temp = np.zeros_like(
            mask
        )

        cv2.ellipse(
            temp,
            (x, y),
            (
                axis_x,
                axis_y,
            ),
            angle,
            0,
            360,
            1.0,
            -1,
            cv2.LINE_AA,
        )

        sigma = max(
            1.0,
            radius * 0.20,
        )

        temp = cv2.GaussianBlur(
            temp,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )

        mask = np.maximum(
            mask,
            temp,
        )

        droplets.append(
            {
                "x": x,
                "y": y,
                "radius": radius,
            }
        )

    return (
        _clip_mask(mask),
        {
            "droplet_count": count,
            "droplets": droplets,
        },
    )


# ============================================================
# Wet spot
# ============================================================

def generate_wet_spot_mask(
    height: int,
    width: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict]:

    mask = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    blob_count = int(
        rng.integers(
            int(config["blob_count_min"]),
            int(config["blob_count_max"]) + 1,
        )
    )

    base_size = min(
        height,
        width,
    )

    center_x, center_y = (
        _random_point(
            width,
            height,
            rng,
            margin_ratio=0.20,
        )
    )

    blobs = []

    for _ in range(blob_count):

        size = int(
            rng.uniform(
                float(
                    config[
                        "size_min_ratio"
                    ]
                ),
                float(
                    config[
                        "size_max_ratio"
                    ]
                ),
            )
            * base_size
        )

        size = max(
            5,
            size,
        )

        jitter_x = int(
            rng.normal(
                0,
                size * 0.45,
            )
        )

        jitter_y = int(
            rng.normal(
                0,
                size * 0.45,
            )
        )

        x = int(
            np.clip(
                center_x + jitter_x,
                0,
                width - 1,
            )
        )

        y = int(
            np.clip(
                center_y + jitter_y,
                0,
                height - 1,
            )
        )

        axis_x = max(
            3,
            int(
                size
                * rng.uniform(
                    0.7,
                    1.4,
                )
            ),
        )

        axis_y = max(
            3,
            int(
                size
                * rng.uniform(
                    0.5,
                    1.2,
                )
            ),
        )

        angle = float(
            rng.uniform(
                0,
                180,
            )
        )

        cv2.ellipse(
            mask,
            (x, y),
            (
                axis_x,
                axis_y,
            ),
            angle,
            0,
            360,
            1.0,
            -1,
            cv2.LINE_AA,
        )

        blobs.append(
            {
                "x": x,
                "y": y,
                "axis_x": axis_x,
                "axis_y": axis_y,
            }
        )

    sigma = max(
        2.0,
        base_size * 0.03,
    )

    mask = cv2.GaussianBlur(
        mask,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    if mask.max() > 0:
        mask /= mask.max()

    return (
        _clip_mask(mask),
        {
            "blob_count": blob_count,
            "center": [
                center_x,
                center_y,
            ],
            "blobs": blobs,
        },
    )


# ============================================================
# Thin stream
# ============================================================

def generate_thin_stream_mask(
    height: int,
    width: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict]:

    mask = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    start_x = int(
        rng.uniform(
            width * 0.20,
            width * 0.80,
        )
    )

    start_y = int(
        rng.uniform(
            height * 0.05,
            height * 0.35,
        )
    )

    end_y = int(
        rng.uniform(
            height * 0.65,
            height * 0.98,
        )
    )

    point_count = 7

    ys = np.linspace(
        start_y,
        end_y,
        point_count,
    )

    xs = []

    current_x = start_x

    for index in range(
        point_count
    ):

        if index > 0:
            current_x += int(
                rng.normal(
                    0,
                    width * 0.025,
                )
            )

        current_x = int(
            np.clip(
                current_x,
                2,
                width - 3,
            )
        )

        xs.append(
            current_x
        )

    points = np.array(
        [
            [
                int(x),
                int(y),
            ]
            for x, y
            in zip(
                xs,
                ys,
            )
        ],
        dtype=np.int32,
    )

    thickness = int(
        rng.integers(
            int(config["width_min"]),
            int(config["width_max"]) + 1,
        )
    )

    cv2.polylines(
        mask,
        [
            points.reshape(
                -1,
                1,
                2,
            )
        ],
        False,
        1.0,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # stream 아래쪽에 작은 droplet 추가
    droplet_count = int(
        rng.integers(
            1,
            4,
        )
    )

    for _ in range(
        droplet_count
    ):

        x = int(
            np.clip(
                points[-1, 0]
                + rng.normal(
                    0,
                    width * 0.04,
                ),
                2,
                width - 3,
            )
        )

        y = int(
            rng.uniform(
                end_y,
                height - 2,
            )
        )

        radius = int(
            rng.integers(
                2,
                max(
                    3,
                    thickness * 2,
                ),
            )
        )

        cv2.circle(
            mask,
            (x, y),
            radius,
            1.0,
            -1,
            cv2.LINE_AA,
        )

    mask = cv2.GaussianBlur(
        mask,
        (0, 0),
        sigmaX=1.2,
        sigmaY=1.2,
    )

    if mask.max() > 0:
        mask /= mask.max()

    return (
        _clip_mask(mask),
        {
            "points":
                points.tolist(),

            "thickness":
                thickness,

            "droplet_count":
                droplet_count,
        },
    )


# ============================================================
# Water appearance
# ============================================================

def apply_water_effect(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    darkness: float,
    rng: np.random.Generator,
) -> np.ndarray:

    image_float = (
        image.astype(
            np.float32
        )
    )

    mask = _clip_mask(
        mask
    )

    alpha = mask[
        ...,
        None,
    ]

    # --------------------------------------------------------
    # Wet surface:
    # slight local smoothing + darkening
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        image_float,
        (0, 0),
        sigmaX=1.3,
        sigmaY=1.3,
    )

    wet = (
        image_float * 0.70
        + blurred * 0.30
    )

    wet *= (
        1.0
        - darkness
    )

    output = (
        image_float
        * (
            1.0
            - alpha
        )
        + wet
        * alpha
    )

    # --------------------------------------------------------
    # Specular edge highlight
    # --------------------------------------------------------

    mask_u8 = (
        mask
        * 255.0
    ).astype(
        np.uint8
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    edge = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_GRADIENT,
        kernel,
    ).astype(
        np.float32
    ) / 255.0

    edge_strength = float(
        rng.uniform(
            10.0,
            30.0,
        )
    )

    output += (
        edge[
            ...,
            None,
        ]
        * edge_strength
    )

    # --------------------------------------------------------
    # Small highlight on wet region
    # --------------------------------------------------------

    highlight = cv2.GaussianBlur(
        mask,
        (0, 0),
        sigmaX=2.0,
        sigmaY=2.0,
    )

    highlight *= float(
        rng.uniform(
            3.0,
            12.0,
        )
    )

    output += highlight[
        ...,
        None,
    ]

    return np.clip(
        output,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# Public generator
# ============================================================

def generate_defect(
    image: np.ndarray,
    defect_type: str,
    roi: tuple[
        int,
        int,
        int,
        int,
    ],
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict,
]:

    x, y, width, height = roi

    local_config = config[
        defect_type
    ]

    if defect_type == "droplet":

        local_mask, params = (
            generate_droplet_mask(
                height,
                width,
                rng,
                local_config,
            )
        )

    elif defect_type == "wet_spot":

        local_mask, params = (
            generate_wet_spot_mask(
                height,
                width,
                rng,
                local_config,
            )
        )

    elif defect_type == "thin_stream":

        local_mask, params = (
            generate_thin_stream_mask(
                height,
                width,
                rng,
                local_config,
            )
        )

    else:
        raise ValueError(
            f"Unsupported defect type: "
            f"{defect_type}"
        )

    darkness = float(
        rng.uniform(
            float(
                local_config[
                    "darkness_min"
                ]
            ),
            float(
                local_config[
                    "darkness_max"
                ]
            ),
        )
    )

    full_mask = np.zeros(
        image.shape[:2],
        dtype=np.float32,
    )

    full_mask[
        y:y + height,
        x:x + width,
    ] = local_mask

    output = apply_water_effect(
        image,
        full_mask,
        darkness=darkness,
        rng=rng,
    )

    params[
        "darkness"
    ] = darkness

    return (
        output,
        full_mask,
        params,
    )
