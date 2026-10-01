from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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


# ============================================================
# Utility
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

    if (
        image is None
        or image.ndim != 3
        or image.shape[2] != 4
    ):
        return None

    return image


def trim_transparent(
    asset: np.ndarray,
    alpha_threshold: int = 5,
) -> np.ndarray:

    alpha = asset[:, :, 3]

    ys, xs = np.where(
        alpha > alpha_threshold
    )

    if (
        len(xs) == 0
        or len(ys) == 0
    ):
        return asset

    return asset[
        int(ys.min()):
        int(ys.max()) + 1,

        int(xs.min()):
        int(xs.max()) + 1,
    ]


def collect_assets(
    directory: Path,
) -> list[Path]:

    return sorted(
        path
        for path in directory.rglob("*.png")
        if path.is_file()
    )


# ============================================================
# Config loader
# ============================================================

def load_anomaly_roi(
    path: Path,
    roi_name: str,
) -> tuple[int, int, int, int]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        config = yaml.safe_load(
            stream
        )

    values = (
        config[
            "rois"
        ][
            roi_name
        ][
            "anomaly_roi"
        ]
    )

    return tuple(
        int(value)
        for value in values
    )


# ============================================================
# Gate metadata
# ============================================================

def parse_homography(
    value,
) -> np.ndarray | None:

    if value is None:
        return None

    try:

        if isinstance(
            value,
            str,
        ):
            value = json.loads(
                value
            )

        matrix = np.asarray(
            value,
            dtype=np.float64,
        )

        if matrix.shape != (
            3,
            3,
        ):
            return None

        if not np.all(
            np.isfinite(
                matrix
            )
        ):
            return None

        return matrix

    except Exception:
        return None


class HomographyLookup:

    def __init__(
        self,
        metadata_path: Path,
        source_video: Path,
    ):

        df = pd.read_csv(
            metadata_path,
            encoding="utf-8-sig",
        )

        source_name = (
            source_video.name
        )

        if "video_path" in df.columns:

            matched = df[
                df[
                    "video_path"
                ]
                .astype(str)
                .map(
                    lambda x:
                    Path(x).name
                    == source_name
                )
            ]

        else:

            matched = df[
                df[
                    "video_name"
                ].astype(str)
                == source_video.stem
            ]

        matched = matched[
            matched[
                "match_status"
            ] == "matched"
        ].copy()

        matched[
            "homography_matrix"
        ] = matched[
            "homography"
        ].apply(
            parse_homography
        )

        matched = matched[
            matched[
                "homography_matrix"
            ].notna()
        ]

        if matched.empty:
            raise RuntimeError(
                f"No matched gate metadata: "
                f"{metadata_path}"
            )

        self.df = (
            matched
            .sort_values(
                "timestamp_sec"
            )
            .reset_index(
                drop=True
            )
        )

        self.timestamps = (
            self.df[
                "timestamp_sec"
            ]
            .to_numpy(
                dtype=float
            )
        )

    def nearest(
        self,
        timestamp_sec: float,
        max_gap: float,
    ):

        index = int(
            np.argmin(
                np.abs(
                    self.timestamps
                    - timestamp_sec
                )
            )
        )

        row = self.df.iloc[
            index
        ]

        gap = abs(
            float(
                row[
                    "timestamp_sec"
                ]
            )
            - timestamp_sec
        )

        if gap > max_gap:
            return None

        return row


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
        interpolation=(
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_CUBIC
        ),
    )


def rotate_rgba(
    asset: np.ndarray,
    angle: float,
) -> np.ndarray:

    h, w = asset.shape[:2]

    center = (
        w / 2.0,
        h / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    cos = abs(
        matrix[0, 0]
    )

    sin = abs(
        matrix[0, 1]
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
        new_w / 2.0
        - center[0]
    )

    matrix[
        1,
        2
    ] += (
        new_h / 2.0
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


# ============================================================
# Event
# ============================================================

@dataclass
class SyntheticEvent:

    event_id: str
    roi: str
    defect_type: str

    start_sec: float
    end_sec: float

    canonical_rgba: np.ndarray

    opacity: float

    def active(
        self,
        timestamp: float,
    ) -> bool:

        return (
            self.start_sec
            <= timestamp
            <= self.end_sec
        )

    def progress(
        self,
        timestamp: float,
    ) -> float:

        return np.clip(
            (
                timestamp
                - self.start_sec
            )
            / (
                self.end_sec
                - self.start_sec
            ),
            0.0,
            1.0,
        )


# ============================================================
# Temporal effect
# ============================================================

def temporal_gain(
    defect_type: str,
    progress: float,
) -> float:

    # -----------------------------
    # 물방울
    # 서서히 맺힌 후 유지
    # -----------------------------

    if defect_type == "droplet":

        return float(
            np.clip(
                progress
                / 0.25,
                0.0,
                1.0,
            )
        )

    # -----------------------------
    # 지속 누수
    # fade-in → hold → fade-out
    # -----------------------------

    if defect_type == "leak":

        if progress < 0.15:

            return (
                progress
                / 0.15
            )

        if progress > 0.85:

            return (
                1.0
                - (
                    progress
                    - 0.85
                )
                / 0.15
            )

        return 1.0

    # -----------------------------
    # splash
    # 약간의 pulsation
    # -----------------------------

    if defect_type == "splash":

        envelope = min(
            1.0,
            progress / 0.10,
            (
                1.0
                - progress
            ) / 0.10,
        )

        pulse = (
            0.75
            + 0.25
            * abs(
                np.sin(
                    progress
                    * np.pi
                    * 8
                )
            )
        )

        return float(
            max(
                0.0,
                envelope
                * pulse,
            )
        )

    return 1.0


# ============================================================
# Canonical asset
# ============================================================

def create_event_asset(
    *,
    event_config: dict,
    template_shape: tuple[int, int],
    anomaly_roi: tuple[
        int,
        int,
        int,
        int,
    ],
    asset_paths: list[Path],
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    Path,
    dict,
]:

    asset_path = asset_paths[
        int(
            rng.integers(
                0,
                len(asset_paths),
            )
        )
    ]

    asset = imread_rgba(
        asset_path
    )

    if asset is None:
        raise RuntimeError(
            asset_path
        )

    asset = trim_transparent(
        asset
    )

    (
        roi_x,
        roi_y,
        roi_w,
        roi_h,
    ) = anomaly_roi

    width_ratio = float(
        rng.uniform(
            event_config[
                "width_ratio_min"
            ],
            event_config[
                "width_ratio_max"
            ],
        )
    )

    target_width = max(
        5,
        int(
            roi_w
            * width_ratio
        ),
    )

    asset = resize_asset(
        asset,
        target_width,
    )

    angle = float(
        rng.uniform(
            event_config[
                "rotation_min"
            ],
            event_config[
                "rotation_max"
            ],
        )
    )

    asset = rotate_rgba(
        asset,
        angle,
    )

    opacity = float(
        rng.uniform(
            event_config[
                "opacity_min"
            ],
            event_config[
                "opacity_max"
            ],
        )
    )

    zone = event_config[
        "zone"
    ]

    zx1, zy1, zx2, zy2 = (
        zone
    )

    anchor_x = (
        roi_x
        + int(
            rng.uniform(
                zx1,
                zx2,
            )
            * roi_w
        )
    )

    anchor_y = (
        roi_y
        + int(
            rng.uniform(
                zy1,
                zy2,
            )
            * roi_h
        )
    )

    asset_h, asset_w = (
        asset.shape[:2]
    )

    if (
        event_config[
            "defect_type"
        ]
        == "leak"
    ):

        origin_x = (
            anchor_x
            - asset_w // 2
        )

        origin_y = (
            anchor_y
            - int(
                asset_h * 0.08
            )
        )

    else:

        origin_x = (
            anchor_x
            - asset_w // 2
        )

        origin_y = (
            anchor_y
            - asset_h // 2
        )

    template_h, template_w = (
        template_shape
    )

    canvas = np.zeros(
        (
            template_h,
            template_w,
            4,
        ),
        dtype=np.uint8,
    )

    x1 = max(
        0,
        origin_x,
    )

    y1 = max(
        0,
        origin_y,
    )

    x2 = min(
        template_w,
        origin_x + asset_w,
    )

    y2 = min(
        template_h,
        origin_y + asset_h,
    )

    if (
        x1 < x2
        and y1 < y2
    ):

        ax1 = (
            x1
            - origin_x
        )

        ay1 = (
            y1
            - origin_y
        )

        ax2 = (
            ax1
            + x2
            - x1
        )

        ay2 = (
            ay1
            + y2
            - y1
        )

        canvas[
            y1:y2,
            x1:x2,
        ] = asset[
            ay1:ay2,
            ax1:ax2,
        ]

    params = {
        "asset":
            str(
                asset_path
            ),
        "width_ratio":
            width_ratio,
        "rotation":
            angle,
        "opacity":
            opacity,
        "anchor":
            [
                anchor_x,
                anchor_y,
            ],
    }

    return (
        canvas,
        asset_path,
        params,
    )


# ============================================================
# Projection
# ============================================================

def composite_projected_asset(
    frame: np.ndarray,
    *,
    search_roi: tuple[
        int,
        int,
        int,
        int,
    ],
    homography: np.ndarray,
    canonical_rgba: np.ndarray,
    gain: float,
    opacity: float,
) -> np.ndarray:

    (
        sx,
        sy,
        sw,
        sh,
    ) = search_roi

    projected_rgb = cv2.warpPerspective(
        canonical_rgba[
            :,
            :,
            :3,
        ],
        homography,
        (
            sw,
            sh,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    projected_alpha = cv2.warpPerspective(
        canonical_rgba[
            :,
            :,
            3,
        ],
        homography,
        (
            sw,
            sh,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    alpha = (
        projected_alpha.astype(
            np.float32
        )
        / 255.0
    )

    alpha *= (
        gain
        * opacity
    )

    alpha = np.clip(
        alpha,
        0,
        1,
    )

    target = frame[
        sy:sy + sh,
        sx:sx + sw,
    ]

    fg = projected_rgb.astype(
        np.float32
    )

    bg = target.astype(
        np.float32
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

    frame[
        sy:sy + sh,
        sx:sx + sw,
    ] = np.clip(
        blended,
        0,
        255,
    ).astype(
        np.uint8
    )

    return frame


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-video",
        required=True,
    )

    parser.add_argument(
        "--config",
        default=(
            "config/"
            "anomaly_test_video.yaml"
        ),
    )

    parser.add_argument(
        "--normal-roi-config",
        default=(
            "config/"
            "normal_roi.yaml"
        ),
    )

    parser.add_argument(
        "--left-template",
        default=(
            "data/templates/left.png"
        ),
    )

    parser.add_argument(
        "--right-template",
        default=(
            "data/templates/right.png"
        ),
    )

    parser.add_argument(
        "--left-metadata",
        default=(
            "data/gate_dataset/"
            "val/left/metadata.csv"
        ),
    )

    parser.add_argument(
        "--right-metadata",
        default=(
            "data/gate_dataset/"
            "val/right/metadata.csv"
        ),
    )

    parser.add_argument(
        "--asset-root",
        default=(
            "data/synthetic_assets/"
            "water"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/videos/"
            "anomaly_test/"
            "anomaly_test.mp4"
        ),
    )

    args = parser.parse_args()

    source_video = resolve_path(
        args.source_video
    )

    config_path = resolve_path(
        args.config
    )

    normal_roi_config = resolve_path(
        args.normal_roi_config
    )

    output_path = resolve_path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        settings = yaml.safe_load(
            stream
        )

    max_gap = float(
        settings[
            "generation"
        ][
            "max_metadata_gap_sec"
        ]
    )

    rng = np.random.default_rng(
        int(
            settings[
                "generation"
            ][
                "seed"
            ]
        )
    )

    # --------------------------------------------------------
    # Template
    # --------------------------------------------------------

    templates = {}

    for roi_name, path in {
        "left":
            args.left_template,
        "right":
            args.right_template,
    }.items():

        image = cv2.imread(
            str(
                resolve_path(
                    path
                )
            )
        )

        if image is None:
            raise FileNotFoundError(
                path
            )

        templates[
            roi_name
        ] = image

    # --------------------------------------------------------
    # Homography lookup
    # --------------------------------------------------------

    lookups = {
        "left":
            HomographyLookup(
                resolve_path(
                    args.left_metadata
                ),
                source_video,
            ),

        "right":
            HomographyLookup(
                resolve_path(
                    args.right_metadata
                ),
                source_video,
            ),
    }

    # --------------------------------------------------------
    # Build events
    # --------------------------------------------------------

    events = []

    event_metadata = []

    asset_root = resolve_path(
        args.asset_root
    )

    for event_config in settings[
        "events"
    ]:

        roi_name = event_config[
            "roi"
        ]

        defect_type = (
            event_config[
                "defect_type"
            ]
        )

        assets = collect_assets(
            asset_root
            / defect_type
        )

        if not assets:
            raise RuntimeError(
                f"No assets: "
                f"{defect_type}"
            )

        anomaly_roi = (
            load_anomaly_roi(
                normal_roi_config,
                roi_name,
            )
        )

        template = templates[
            roi_name
        ]

        (
            canonical,
            asset_path,
            params,
        ) = create_event_asset(
            event_config=(
                event_config
            ),
            template_shape=(
                template.shape[:2]
            ),
            anomaly_roi=(
                anomaly_roi
            ),
            asset_paths=assets,
            rng=rng,
        )

        event = SyntheticEvent(
            event_id=(
                event_config[
                    "id"
                ]
            ),
            roi=roi_name,
            defect_type=(
                defect_type
            ),
            start_sec=float(
                event_config[
                    "start_sec"
                ]
            ),
            end_sec=float(
                event_config[
                    "end_sec"
                ]
            ),
            canonical_rgba=(
                canonical
            ),
            opacity=float(
                params[
                    "opacity"
                ]
            ),
        )

        events.append(
            event
        )

        event_metadata.append(
            {
                "id":
                    event.event_id,
                "roi":
                    event.roi,
                "defect_type":
                    event.defect_type,
                "start_sec":
                    event.start_sec,
                "end_sec":
                    event.end_sec,
                **params,
            }
        )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    capture = cv2.VideoCapture(
        str(
            source_video
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            source_video
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(
            output_path
        ),
        fourcc,
        fps,
        (
            width,
            height,
        ),
    )

    frame_index = 0

    while True:

        ok, frame = (
            capture.read()
        )

        if not ok:
            break

        timestamp = (
            frame_index
            / fps
        )

        for event in events:

            if not event.active(
                timestamp
            ):
                continue

            row = lookups[
                event.roi
            ].nearest(
                timestamp,
                max_gap,
            )

            if row is None:
                continue

            H = row[
                "homography_matrix"
            ]

            search_roi = (
                int(
                    row[
                        "search_x"
                    ]
                ),
                int(
                    row[
                        "search_y"
                    ]
                ),
                int(
                    row[
                        "search_width"
                    ]
                ),
                int(
                    row[
                        "search_height"
                    ]
                ),
            )

            progress = (
                event.progress(
                    timestamp
                )
            )

            gain = temporal_gain(
                event.defect_type,
                progress,
            )

            frame = composite_projected_asset(
                frame,
                search_roi=(
                    search_roi
                ),
                homography=H,
                canonical_rgba=(
                    event.canonical_rgba
                ),
                gain=gain,
                opacity=(
                    event.opacity
                ),
            )

        writer.write(
            frame
        )

        frame_index += 1

    capture.release()
    writer.release()

    sidecar = (
        output_path
        .with_suffix(
            ".events.json"
        )
    )

    sidecar.write_text(
        json.dumps(
            {
                "source_video":
                    str(
                        source_video
                    ),
                "output_video":
                    str(
                        output_path
                    ),
                "fps":
                    fps,
                "events":
                    event_metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"[DONE] {output_path}"
    )

    print(
        f"[EVENTS] {sidecar}"
    )


if __name__ == "__main__":
    main()
