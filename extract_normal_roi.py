from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Models
# ============================================================

@dataclass(frozen=True)
class GateThreshold:
    min_good_matches: int | None = None
    min_inliers: int | None = None
    min_inlier_ratio: float | None = None
    min_area_ratio: float | None = None
    max_area_ratio: float | None = None
    max_abs_rotation: float | None = None


@dataclass(frozen=True)
class RoiConfig:
    name: str
    template_path: Path
    anomaly_roi: tuple[int, int, int, int] | None


# ============================================================
# Path / Image utility
# ============================================================

def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def imread_safe(path: Path) -> np.ndarray | None:
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
        path.suffix
        if path.suffix
        else ".png"
    )

    success, encoded = cv2.imencode(
        suffix,
        image,
    )

    if not success:
        return False

    encoded.tofile(
        str(path)
    )

    return True


def resolve_image_path(
    raw_path: Any,
    metadata_path: Path,
) -> Path | None:

    if raw_path is None:
        return None

    raw_path = str(raw_path).strip()

    if not raw_path:
        return None

    path = Path(raw_path)

    if path.exists():
        return path.resolve()

    if not path.is_absolute():
        candidate = (
            PROJECT_ROOT
            / path
        )

        if candidate.exists():
            return candidate.resolve()

    # fallback:
    # metadata.csv 옆의 images 폴더
    fallback = (
        metadata_path.parent
        / "images"
        / path.name
    )

    if fallback.exists():
        return fallback.resolve()

    return None


# ============================================================
# Config
# ============================================================

def load_normal_roi_config(
    path: Path,
) -> tuple[
    dict[str, RoiConfig],
    float,
]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        data = yaml.safe_load(
            stream
        )

    alignment = data.get(
        "alignment",
        {},
    )

    min_valid_ratio = float(
        alignment.get(
            "min_valid_ratio",
            0.95,
        )
    )

    roi_configs: dict[
        str,
        RoiConfig,
    ] = {}

    for roi_name, values in (
        data.get(
            "rois",
            {}
        ).items()
    ):

        template_path = (
            resolve_project_path(
                values[
                    "template"
                ]
            )
        )

        raw_roi = values.get(
            "anomaly_roi"
        )

        anomaly_roi = None

        if raw_roi is not None:

            if (
                not isinstance(
                    raw_roi,
                    list,
                )
                or len(raw_roi) != 4
            ):
                raise ValueError(
                    f"{roi_name}.anomaly_roi must be "
                    "[x, y, width, height]"
                )

            anomaly_roi = tuple(
                int(v)
                for v in raw_roi
            )

        roi_configs[
            roi_name
        ] = RoiConfig(
            name=roi_name,
            template_path=template_path,
            anomaly_roi=anomaly_roi,
        )

    return (
        roi_configs,
        min_valid_ratio,
    )


def load_gate_threshold(
    yaml_path: Path,
    roi_name: str,
) -> tuple[
    GateThreshold,
    str,
]:

    with yaml_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        data = yaml.safe_load(
            stream
        )

    roi_data = data[
        roi_name
    ]

    values = roi_data.get(
        "thresholds",
        {},
    )

    def as_int(
        key: str,
    ) -> int | None:

        value = values.get(
            key
        )

        if value is None:
            return None

        return int(value)

    def as_float(
        key: str,
    ) -> float | None:

        value = values.get(
            key
        )

        if value is None:
            return None

        return float(value)

    threshold = GateThreshold(
        min_good_matches=as_int(
            "min_good_matches"
        ),
        min_inliers=as_int(
            "min_inliers"
        ),
        min_inlier_ratio=as_float(
            "min_inlier_ratio"
        ),
        min_area_ratio=as_float(
            "min_area_ratio"
        ),
        max_area_ratio=as_float(
            "max_area_ratio"
        ),
        max_abs_rotation=as_float(
            "max_abs_rotation"
        ),
    )

    status_required = (
        roi_data.get(
            "status_required",
            "matched",
        )
    )

    return (
        threshold,
        status_required,
    )


# ============================================================
# Gate
# ============================================================

def valid_number(
    value: Any,
) -> float | None:

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(value):
        return None

    return value


def passes_gate(
    row: pd.Series,
    threshold: GateThreshold,
    status_required: str,
) -> bool:

    if (
        str(
            row.get(
                "match_status",
                "",
            )
        )
        != status_required
    ):
        return False

    checks = [
        (
            "good_matches",
            threshold.min_good_matches,
            "min",
        ),
        (
            "inliers",
            threshold.min_inliers,
            "min",
        ),
        (
            "inlier_ratio",
            threshold.min_inlier_ratio,
            "min",
        ),
        (
            "area_ratio",
            threshold.min_area_ratio,
            "min",
        ),
        (
            "area_ratio",
            threshold.max_area_ratio,
            "max",
        ),
    ]

    for (
        column,
        limit,
        direction,
    ) in checks:

        if limit is None:
            continue

        value = valid_number(
            row.get(
                column
            )
        )

        if value is None:
            return False

        if (
            direction == "min"
            and value < limit
        ):
            return False

        if (
            direction == "max"
            and value > limit
        ):
            return False

    if (
        threshold.max_abs_rotation
        is not None
    ):

        value = valid_number(
            row.get(
                "rotation_deg"
            )
        )

        if value is None:
            return False

        if (
            abs(value)
            > threshold.max_abs_rotation
        ):
            return False

    return True


# ============================================================
# Homography
# ============================================================

def parse_homography(
    raw: Any,
) -> np.ndarray | None:

    if raw is None:
        return None

    try:
        if isinstance(
            raw,
            str,
        ):
            raw = raw.strip()

            if not raw:
                return None

            raw = json.loads(
                raw
            )

        H = np.asarray(
            raw,
            dtype=np.float64,
        ).reshape(
            3,
            3,
        )

    except Exception:
        return None

    if not np.isfinite(
        H
    ).all():
        return None

    if (
        abs(
            np.linalg.det(
                H
            )
        )
        < 1e-12
    ):
        return None

    return H


def align_scene_to_template(
    scene: np.ndarray,
    H_template_to_scene: np.ndarray,
    template_size: tuple[int, int],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    template_width, template_height = (
        template_size
    )

    # localize_planar_object:
    #
    # template -> scene
    #
    # PatchCore에서는 scene -> template가 필요
    H_scene_to_template = (
        np.linalg.inv(
            H_template_to_scene
        )
    )

    aligned = cv2.warpPerspective(
        scene,
        H_scene_to_template,
        (
            template_width,
            template_height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            0,
            0,
            0,
        ),
    )

    # 어느 영역이 실제 scene에서 온 픽셀인지 확인
    source_mask = np.full(
        scene.shape[:2],
        255,
        dtype=np.uint8,
    )

    valid_mask = cv2.warpPerspective(
        source_mask,
        H_scene_to_template,
        (
            template_width,
            template_height,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (
        aligned,
        valid_mask,
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
    ] | None,
) -> np.ndarray:

    if roi is None:
        return image

    x, y, width, height = roi

    image_height, image_width = (
        image.shape[:2]
    )

    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError(
            "Anomaly ROI out of bounds: "
            f"roi={roi}, "
            f"image={image_width}x{image_height}"
        )

    return image[
        y:y + height,
        x:x + width,
    ]


def calculate_valid_ratio(
    valid_mask: np.ndarray,
    roi: tuple[
        int,
        int,
        int,
        int,
    ] | None,
) -> float:

    cropped = crop_roi(
        valid_mask,
        roi,
    )

    return float(
        np.mean(
            cropped > 0
        )
    )


# ============================================================
# Manual-label sanity
# ============================================================

def manual_label_allows(
    row: pd.Series,
) -> bool:

    manual_label = str(
        row.get(
            "manual_label",
            "",
        )
        or ""
    ).strip().lower()

    localization_label = str(
        row.get(
            "localization_label",
            "",
        )
        or ""
    ).strip().lower()

    # 사람이 명확히 제외한 데이터는
    # PatchCore normal dataset에 넣지 않는다.
    if manual_label in {
        "not_visible",
        "ignore",
    }:
        return False

    if localization_label == "bad":
        return False

    # blank / visible은 허용
    return True


# ============================================================
# Main extraction
# ============================================================

def process_roi(
    *,
    roi_name: str,
    split: str,
    dataset_root: Path,
    output_root: Path,
    gate_root: Path,
    roi_config: RoiConfig,
    min_valid_ratio: float,
    save_aligned: bool,
) -> dict[str, Any]:

    metadata_path = (
        dataset_root
        / split
        / roi_name
        / "metadata.csv"
    )

    gate_path = (
        gate_root
        / roi_name
        / "selected_gate.yaml"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            metadata_path
        )

    if not gate_path.exists():
        raise FileNotFoundError(
            gate_path
        )

    template = imread_safe(
        roi_config.template_path
    )

    if template is None:
        raise RuntimeError(
            f"Could not read template: "
            f"{roi_config.template_path}"
        )

    template_height, template_width = (
        template.shape[:2]
    )

    threshold, status_required = (
        load_gate_threshold(
            gate_path,
            roi_name,
        )
    )

    df = pd.read_csv(
        metadata_path,
        encoding="utf-8-sig",
    )

    output_dir = (
        output_root
        / split
        / roi_name
    )

    crop_dir = (
        output_dir
        / "images"
    )

    aligned_dir = (
        output_dir
        / "aligned"
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if save_aligned:
        aligned_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    records: list[
        dict[str, Any]
    ] = []

    summary = {
        "total": 0,
        "gate_false": 0,
        "manual_excluded": 0,
        "missing_image": 0,
        "invalid_homography": 0,
        "alignment_failed": 0,
        "low_valid_ratio": 0,
        "saved": 0,
    }

    for index, row in df.iterrows():

        summary[
            "total"
        ] += 1

        base_record = {
            "source_index":
                int(index),

            "video_name":
                row.get(
                    "video_name",
                    "",
                ),

            "frame_idx":
                row.get(
                    "frame_idx",
                    "",
                ),

            "timestamp_sec":
                row.get(
                    "timestamp_sec",
                    "",
                ),

            "manual_label":
                row.get(
                    "manual_label",
                    "",
                ),

            "localization_label":
                row.get(
                    "localization_label",
                    "",
                ),

            "match_status":
                row.get(
                    "match_status",
                    "",
                ),

            "good_matches":
                row.get(
                    "good_matches",
                    "",
                ),

            "inliers":
                row.get(
                    "inliers",
                    "",
                ),

            "inlier_ratio":
                row.get(
                    "inlier_ratio",
                    "",
                ),

            "area_ratio":
                row.get(
                    "area_ratio",
                    "",
                ),

            "rotation_deg":
                row.get(
                    "rotation_deg",
                    "",
                ),
        }

        # ----------------------------------------------------
        # Fixed Gate
        # ----------------------------------------------------

        if not passes_gate(
            row,
            threshold,
            status_required,
        ):
            summary[
                "gate_false"
            ] += 1

            continue

        # ----------------------------------------------------
        # Human exclusion
        # ----------------------------------------------------

        if not manual_label_allows(
            row
        ):
            summary[
                "manual_excluded"
            ] += 1

            continue

        # ----------------------------------------------------
        # Input image
        # ----------------------------------------------------

        image_path = resolve_image_path(
            row.get(
                "image_path"
            ),
            metadata_path,
        )

        if image_path is None:
            summary[
                "missing_image"
            ] += 1

            continue

        scene = imread_safe(
            image_path
        )

        if scene is None:
            summary[
                "missing_image"
            ] += 1

            continue

        # ----------------------------------------------------
        # Homography
        # ----------------------------------------------------

        H = parse_homography(
            row.get(
                "homography"
            )
        )

        if H is None:
            summary[
                "invalid_homography"
            ] += 1

            continue

        try:
            (
                aligned,
                valid_mask,
            ) = align_scene_to_template(
                scene,
                H,
                (
                    template_width,
                    template_height,
                ),
            )

        except (
            np.linalg.LinAlgError,
            cv2.error,
        ):
            summary[
                "alignment_failed"
            ] += 1

            continue

        # ----------------------------------------------------
        # Valid area
        # ----------------------------------------------------

        valid_ratio = (
            calculate_valid_ratio(
                valid_mask,
                roi_config.anomaly_roi,
            )
        )

        if (
            valid_ratio
            < min_valid_ratio
        ):
            summary[
                "low_valid_ratio"
            ] += 1

            continue

        # ----------------------------------------------------
        # Anomaly ROI
        # ----------------------------------------------------

        normal_crop = crop_roi(
            aligned,
            roi_config.anomaly_roi,
        )

        # ----------------------------------------------------
        # Filename
        # ----------------------------------------------------

        video_stem = Path(
            str(
                row.get(
                    "video_name",
                    "video",
                )
            )
        ).stem

        try:
            frame_idx = int(
                float(
                    row.get(
                        "frame_idx",
                        index,
                    )
                )
            )
        except Exception:
            frame_idx = int(
                index
            )

        filename = (
            f"{video_stem}"
            f"__frame_{frame_idx:06d}.png"
        )

        crop_path = (
            crop_dir
            / filename
        )

        if not imwrite_safe(
            crop_path,
            normal_crop,
        ):
            continue

        aligned_path = ""

        if save_aligned:

            aligned_output_path = (
                aligned_dir
                / filename
            )

            if imwrite_safe(
                aligned_output_path,
                aligned,
            ):
                aligned_path = str(
                    aligned_output_path.relative_to(
                        PROJECT_ROOT
                    )
                )

        summary[
            "saved"
        ] += 1

        records.append(
            {
                **base_record,

                "valid_ratio":
                    valid_ratio,

                "output_image":
                    str(
                        crop_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "aligned_image":
                    aligned_path,

                "anomaly_roi":
                    (
                        json.dumps(
                            roi_config.anomaly_roi
                        )
                        if roi_config.anomaly_roi
                        is not None
                        else ""
                    ),
            }
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    output_metadata = (
        output_dir
        / "metadata.csv"
    )

    if records:

        pd.DataFrame(
            records
        ).to_csv(
            output_metadata,
            index=False,
            encoding="utf-8-sig",
        )

    else:

        pd.DataFrame().to_csv(
            output_metadata,
            index=False,
            encoding="utf-8-sig",
        )

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 65
    )

    print(
        f"{split.upper()} / "
        f"{roi_name.upper()}"
    )

    print(
        "=" * 65
    )

    for key, value in (
        summary.items()
    ):
        print(
            f"{key:20s}: {value}"
        )

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract homography-aligned "
            "normal ROI dataset"
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
        ],
        default="train",
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

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/normal_roi.yaml"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "data/gate_dataset"
        ),
    )

    parser.add_argument(
        "--gate-dir",
        type=Path,
        default=Path(
            "outputs/gate_calibration"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/normal_roi"
        ),
    )

    parser.add_argument(
        "--save-aligned",
        action="store_true",
        help=(
            "Save full template-space "
            "aligned images"
        ),
    )

    args = parser.parse_args()

    config_path = (
        resolve_project_path(
            args.config
        )
    )

    dataset_root = (
        resolve_project_path(
            args.dataset_root
        )
    )

    gate_root = (
        resolve_project_path(
            args.gate_dir
        )
    )

    output_root = (
        resolve_project_path(
            args.output_root
        )
    )

    (
        roi_configs,
        min_valid_ratio,
    ) = load_normal_roi_config(
        config_path
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
        ] = process_roi(
            roi_name=roi_name,
            split=args.split,
            dataset_root=dataset_root,
            output_root=output_root,
            gate_root=gate_root,
            roi_config=roi_configs[
                roi_name
            ],
            min_valid_ratio=(
                min_valid_ratio
            ),
            save_aligned=(
                args.save_aligned
            ),
        )

    summary_path = (
        output_root
        / args.split
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

    print()
    print(
        "[DONE]"
    )

    print(
        f"Output: "
        f"{output_root / args.split}"
    )


if __name__ == "__main__":
    main()
