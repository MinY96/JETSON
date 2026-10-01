from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


from src.anomaly.patchcore import (
    PatchCoreModel,
)

from src.anomaly.preprocessing import (
    AspectPadPreprocessor,
    imread_rgb,
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


def get_device(
    value: str,
) -> torch.device:

    value = value.lower()

    if value == "auto":

        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    return torch.device(
        value
    )


def synchronize(
    device: torch.device,
) -> None:

    if device.type == "cuda":
        torch.cuda.synchronize()


def imread_mask(
    path: Path,
) -> np.ndarray:

    data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read mask: {path}"
        )

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
            f"Could not encode: {path}"
        )

    encoded.tofile(
        str(path)
    )


# ============================================================
# Dataset
# ============================================================

class SyntheticMapDataset(
    Dataset,
):

    def __init__(
        self,
        metadata_path: Path,
        input_size: int,
    ) -> None:

        df = pd.read_csv(
            metadata_path,
            encoding="utf-8-sig",
        )

        required = {
            "synthetic_image",
            "mask_image",
            "defect_type",
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"Missing columns: "
                f"{sorted(missing)}"
            )

        df = df.dropna(
            subset=[
                "synthetic_image",
                "mask_image",
            ]
        )

        self.records = (
            df.to_dict(
                orient="records"
            )
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
            self.records
        )

    def __getitem__(
        self,
        index: int,
    ):

        record = self.records[
            index
        ]

        image_path = resolve_path(
            record[
                "synthetic_image"
            ]
        )

        mask_path = resolve_path(
            record[
                "mask_image"
            ]
        )

        image = imread_rgb(
            image_path
        )

        tensor = self.preprocessor(
            image
        )

        height, width = (
            image.shape[:2]
        )

        return {
            "image":
                tensor,

            "image_path":
                str(
                    image_path
                ),

            "mask_path":
                str(
                    mask_path
                ),

            "defect_type":
                str(
                    record[
                        "defect_type"
                    ]
                ),

            "original_height":
                height,

            "original_width":
                width,
        }


# ============================================================
# Letterbox reverse mapping
# ============================================================

def restore_anomaly_map(
    anomaly_map: np.ndarray,
    *,
    original_width: int,
    original_height: int,
    input_size: int,
) -> np.ndarray:
    """
    AspectPadPreprocessor의 letterbox를 역으로 제거한다.

    model anomaly map
        input_size x input_size

            ↓ padding 제거

        resized content

            ↓ original size로 resize

        original anomaly ROI size
    """

    scale = min(
        input_size
        / original_width,
        input_size
        / original_height,
    )

    resized_width = max(
        1,
        int(
            round(
                original_width
                * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                original_height
                * scale
            )
        ),
    )

    x = (
        input_size
        - resized_width
    ) // 2

    y = (
        input_size
        - resized_height
    ) // 2

    content = anomaly_map[
        y:y + resized_height,
        x:x + resized_width,
    ]

    restored = cv2.resize(
        content,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return restored.astype(
        np.float32
    )


# ============================================================
# Localization metrics
# ============================================================

def top_fraction_metrics(
    anomaly_map: np.ndarray,
    gt_mask: np.ndarray,
    fraction: float,
) -> tuple[
    float,
    float,
]:
    """
    anomaly score 상위 fraction 영역과 GT mask의
    precision / recall.
    """

    flat = anomaly_map.reshape(
        -1
    )

    pixel_count = (
        flat.shape[0]
    )

    top_count = max(
        1,
        int(
            round(
                pixel_count
                * fraction
            )
        ),
    )

    # np.partition으로 전체 sort보다 빠르게 threshold 계산
    threshold_index = (
        pixel_count
        - top_count
    )

    threshold = np.partition(
        flat,
        threshold_index,
    )[
        threshold_index
    ]

    predicted = (
        anomaly_map
        >= threshold
    )

    intersection = int(
        np.sum(
            predicted
            & gt_mask
        )
    )

    predicted_count = int(
        np.sum(
            predicted
        )
    )

    gt_count = int(
        np.sum(
            gt_mask
        )
    )

    precision = (
        intersection
        / predicted_count
        if predicted_count > 0
        else 0.0
    )

    recall = (
        intersection
        / gt_count
        if gt_count > 0
        else 0.0
    )

    return (
        float(
            precision
        ),
        float(
            recall
        ),
    )


def calculate_localization_metrics(
    anomaly_map: np.ndarray,
    gt_mask: np.ndarray,
) -> dict[str, Any]:

    gt_mask = gt_mask.astype(
        bool
    )

    if not np.any(
        gt_mask
    ):
        raise ValueError(
            "GT mask contains no positive pixels."
        )

    outside_mask = (
        ~gt_mask
    )

    # --------------------------------------------------------
    # Peak position
    # --------------------------------------------------------

    peak_index = int(
        np.argmax(
            anomaly_map
        )
    )

    peak_y, peak_x = (
        np.unravel_index(
            peak_index,
            anomaly_map.shape,
        )
    )

    peak_hit = bool(
        gt_mask[
            peak_y,
            peak_x,
        ]
    )

    # --------------------------------------------------------
    # Inside / outside
    # --------------------------------------------------------

    inside_scores = (
        anomaly_map[
            gt_mask
        ]
    )

    outside_scores = (
        anomaly_map[
            outside_mask
        ]
    )

    mean_inside = float(
        np.mean(
            inside_scores
        )
    )

    mean_outside = float(
        np.mean(
            outside_scores
        )
    )

    max_inside = float(
        np.max(
            inside_scores
        )
    )

    max_outside = float(
        np.max(
            outside_scores
        )
    )

    mean_ratio = (
        mean_inside
        / (
            mean_outside
            + 1e-8
        )
    )

    max_ratio = (
        max_inside
        / (
            max_outside
            + 1e-8
        )
    )

    # --------------------------------------------------------
    # Top fraction overlap
    # --------------------------------------------------------

    (
        top1_precision,
        top1_recall,
    ) = top_fraction_metrics(
        anomaly_map,
        gt_mask,
        0.01,
    )

    (
        top5_precision,
        top5_recall,
    ) = top_fraction_metrics(
        anomaly_map,
        gt_mask,
        0.05,
    )

    (
        top10_precision,
        top10_recall,
    ) = top_fraction_metrics(
        anomaly_map,
        gt_mask,
        0.10,
    )

    return {
        "peak_hit":
            peak_hit,

        "peak_x":
            int(
                peak_x
            ),

        "peak_y":
            int(
                peak_y
            ),

        "mean_inside":
            mean_inside,

        "mean_outside":
            mean_outside,

        "mean_inside_outside_ratio":
            float(
                mean_ratio
            ),

        "max_inside":
            max_inside,

        "max_outside":
            max_outside,

        "max_inside_outside_ratio":
            float(
                max_ratio
            ),

        "top1_precision":
            top1_precision,

        "top1_recall":
            top1_recall,

        "top5_precision":
            top5_precision,

        "top5_recall":
            top5_recall,

        "top10_precision":
            top10_precision,

        "top10_recall":
            top10_recall,

        "mask_pixel_ratio":
            float(
                np.mean(
                    gt_mask
                )
            ),
    }


# ============================================================
# Visualization
# ============================================================

def normalize_map(
    anomaly_map: np.ndarray,
) -> np.ndarray:

    minimum = float(
        anomaly_map.min()
    )

    maximum = float(
        anomaly_map.max()
    )

    if maximum <= minimum:

        return np.zeros(
            anomaly_map.shape,
            dtype=np.uint8,
        )

    normalized = (
        (
            anomaly_map
            - minimum
        )
        / (
            maximum
            - minimum
        )
        * 255.0
    )

    return (
        normalized
        .clip(
            0,
            255,
        )
        .astype(
            np.uint8
        )
    )


def make_overlay(
    image_rgb: np.ndarray,
    anomaly_map: np.ndarray,
    gt_mask: np.ndarray,
    *,
    defect_type: str,
    image_score: float,
    metrics: dict[str, Any],
) -> np.ndarray:

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    normalized = normalize_map(
        anomaly_map
    )

    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        image_bgr,
        0.55,
        heatmap,
        0.45,
        0,
    )

    # --------------------------------------------------------
    # GT mask contour
    # --------------------------------------------------------

    mask_u8 = (
        gt_mask.astype(
            np.uint8
        )
        * 255
    )

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Peak
    # --------------------------------------------------------

    peak_x = int(
        metrics[
            "peak_x"
        ]
    )

    peak_y = int(
        metrics[
            "peak_y"
        ]
    )

    cv2.drawMarker(
        overlay,
        (
            peak_x,
            peak_y,
        ),
        (
            0,
            255,
            255,
        ),
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=2,
    )

    # --------------------------------------------------------
    # Information panel
    # --------------------------------------------------------

    height, width = (
        image_bgr.shape[:2]
    )

    panel_height = 105

    panel = np.zeros(
        (
            panel_height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    lines = [
        (
            f"{defect_type} | "
            f"image_score={image_score:.3f}"
        ),
        (
            f"peak_hit={metrics['peak_hit']} | "
            f"mean_ratio="
            f"{metrics['mean_inside_outside_ratio']:.2f}"
        ),
        (
            f"top1 precision="
            f"{metrics['top1_precision']:.3f} | "
            f"top5 precision="
            f"{metrics['top5_precision']:.3f}"
        ),
    ]

    y = 27

    for line in lines:

        cv2.putText(
            panel,
            line,
            (
                10,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                235,
                235,
                235,
            ),
            1,
            cv2.LINE_AA,
        )

        y += 30

    # Original / Overlay / GT mask
    mask_display = cv2.cvtColor(
        mask_u8,
        cv2.COLOR_GRAY2BGR,
    )

    content = np.hstack(
        [
            image_bgr,
            overlay,
            mask_display,
        ]
    )

    # panel도 3배 폭
    panel_full = np.zeros(
        (
            panel_height,
            content.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    panel_full[
        :,
        :width,
    ] = panel

    cv2.putText(
        panel_full,
        "Original",
        (
            10,
            panel_height - 8,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (
            180,
            180,
            180,
        ),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel_full,
        "PatchCore anomaly map + GT contour",
        (
            width + 10,
            panel_height - 8,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (
            180,
            180,
            180,
        ),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel_full,
        "Synthetic GT mask",
        (
            width * 2 + 10,
            panel_height - 8,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (
            180,
            180,
            180,
        ),
        1,
        cv2.LINE_AA,
    )

    return np.vstack(
        [
            panel_full,
            content,
        ]
    )


# ============================================================
# Summary
# ============================================================

def summarize_group(
    group: pd.DataFrame,
) -> dict[str, Any]:

    return {
        "count":
            int(
                len(group)
            ),

        "peak_hit_rate":
            float(
                group[
                    "peak_hit"
                ].mean()
            ),

        "mean_inside_outside_ratio":
            float(
                group[
                    "mean_inside_outside_ratio"
                ].mean()
            ),

        "median_inside_outside_ratio":
            float(
                group[
                    "mean_inside_outside_ratio"
                ].median()
            ),

        "top1_precision":
            float(
                group[
                    "top1_precision"
                ].mean()
            ),

        "top1_recall":
            float(
                group[
                    "top1_recall"
                ].mean()
            ),

        "top5_precision":
            float(
                group[
                    "top5_precision"
                ].mean()
            ),

        "top5_recall":
            float(
                group[
                    "top5_recall"
                ].mean()
            ),

        "top10_precision":
            float(
                group[
                    "top10_precision"
                ].mean()
            ),

        "top10_recall":
            float(
                group[
                    "top10_recall"
                ].mean()
            ),

        "image_score_mean":
            float(
                group[
                    "image_score"
                ].mean()
            ),

        "image_score_min":
            float(
                group[
                    "image_score"
                ].min()
            ),

        "image_score_max":
            float(
                group[
                    "image_score"
                ].max()
            ),
    }


# ============================================================
# Evaluate ROI
# ============================================================

@torch.no_grad()
def evaluate_roi(
    *,
    roi_name: str,
    model_dir: Path,
    synthetic_metadata: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    mask_threshold: int,
    save_overlays: bool,
    max_overlays_per_type: int,
) -> dict:

    model_path = (
        model_dir
        / "patchcore.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    model = PatchCoreModel.load(
        model_path,
        device,
    )

    dataset = SyntheticMapDataset(
        synthetic_metadata,
        input_size=(
            model.config.input_size
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay_counter = defaultdict(
        int
    )

    records = []

    for batch in tqdm(
        loader,
        desc=(
            f"{roi_name} anomaly-map validation"
        ),
    ):

        images = batch[
            "image"
        ]

        synchronize(
            device
        )

        (
            image_scores,
            anomaly_maps,
        ) = model.predict(
            images
        )

        synchronize(
            device
        )

        image_scores = (
            image_scores
            .detach()
            .cpu()
            .numpy()
        )

        anomaly_maps = (
            anomaly_maps
            .detach()
            .cpu()
            .numpy()
        )

        batch_size_actual = (
            len(
                batch[
                    "image_path"
                ]
            )
        )

        for i in range(
            batch_size_actual
        ):

            image_path = Path(
                batch[
                    "image_path"
                ][i]
            )

            mask_path = Path(
                batch[
                    "mask_path"
                ][i]
            )

            defect_type = str(
                batch[
                    "defect_type"
                ][i]
            )

            original_height = int(
                batch[
                    "original_height"
                ][i]
            )

            original_width = int(
                batch[
                    "original_width"
                ][i]
            )

            # ------------------------------------------------
            # Restore anomaly map to original crop coordinates
            # ------------------------------------------------

            anomaly_map = (
                restore_anomaly_map(
                    anomaly_maps[
                        i
                    ],
                    original_width=(
                        original_width
                    ),
                    original_height=(
                        original_height
                    ),
                    input_size=(
                        model.config.input_size
                    ),
                )
            )

            # ------------------------------------------------
            # GT mask
            # ------------------------------------------------

            mask = imread_mask(
                mask_path
            )

            if (
                mask.shape[0]
                != original_height
                or mask.shape[1]
                != original_width
            ):

                mask = cv2.resize(
                    mask,
                    (
                        original_width,
                        original_height,
                    ),
                    interpolation=(
                        cv2.INTER_NEAREST
                    ),
                )

            gt_mask = (
                mask
                >= mask_threshold
            )

            metrics = (
                calculate_localization_metrics(
                    anomaly_map,
                    gt_mask,
                )
            )

            image_score = float(
                image_scores[
                    i
                ]
            )

            record = {
                "roi":
                    roi_name,

                "defect_type":
                    defect_type,

                "image_path":
                    str(
                        image_path
                    ),

                "mask_path":
                    str(
                        mask_path
                    ),

                "image_score":
                    image_score,

                **metrics,
            }

            records.append(
                record
            )

            # ------------------------------------------------
            # Overlay
            # ------------------------------------------------

            if save_overlays:

                count = (
                    overlay_counter[
                        defect_type
                    ]
                )

                if (
                    max_overlays_per_type <= 0
                    or count
                    < max_overlays_per_type
                ):

                    image_rgb = (
                        imread_rgb(
                            image_path
                        )
                    )

                    overlay = make_overlay(
                        image_rgb,
                        anomaly_map,
                        gt_mask,
                        defect_type=(
                            defect_type
                        ),
                        image_score=(
                            image_score
                        ),
                        metrics=metrics,
                    )

                    overlay_path = (
                        output_dir
                        / "overlays"
                        / defect_type
                        / (
                            image_path.stem
                            + ".png"
                        )
                    )

                    imwrite_safe(
                        overlay_path,
                        overlay,
                    )

                    overlay_counter[
                        defect_type
                    ] += 1

    # ========================================================
    # Per-image CSV
    # ========================================================

    result = pd.DataFrame(
        records
    )

    result.to_csv(
        output_dir
        / "per_image_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Summary
    # ========================================================

    summary_rows = []

    overall_summary = (
        summarize_group(
            result
        )
    )

    summary_rows.append(
        {
            "defect_type":
                "ALL",
            **overall_summary,
        }
    )

    summary_json = {
        "roi":
            roi_name,

        "overall":
            overall_summary,

        "by_defect_type":
            {},
    }

    for (
        defect_type,
        group,
    ) in result.groupby(
        "defect_type",
        sort=True,
    ):

        stats = summarize_group(
            group
        )

        summary_json[
            "by_defect_type"
        ][
            defect_type
        ] = stats

        summary_rows.append(
            {
                "defect_type":
                    defect_type,
                **stats,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        output_dir
        / "summary_by_type.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary_json,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Useful inspection lists
    # ========================================================

    result.sort_values(
        [
            "peak_hit",
            "mean_inside_outside_ratio",
        ],
        ascending=[
            True,
            True,
        ],
    ).to_csv(
        output_dir
        / "worst_localization.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missed_peak = result[
        ~result[
            "peak_hit"
        ]
    ]

    missed_peak.to_csv(
        output_dir
        / "peak_missed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Console
    # ========================================================

    print()
    print(
        "=" * 75
    )

    print(
        f"ANOMALY MAP VALIDATION: "
        f"{roi_name.upper()}"
    )

    print(
        "=" * 75
    )

    display_columns = [
        "defect_type",
        "count",
        "peak_hit_rate",
        "mean_inside_outside_ratio",
        "top1_precision",
        "top5_precision",
        "top10_precision",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Peak missed:",
        len(
            missed_peak
        ),
        "/",
        len(
            result
        ),
    )

    return summary_json


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patchcore-config",
        type=Path,
        default=Path(
            "config/patchcore.yaml"
        ),
    )

    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=Path(
            "data/synthetic_ng"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/anomaly_map_validation"
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

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=32,
        help=(
            "Synthetic anti-aliased mask를 "
            "binary GT로 바꾸는 threshold"
        ),
    )

    parser.add_argument(
        "--save-overlays",
        action="store_true",
    )

    parser.add_argument(
        "--max-overlays-per-type",
        type=int,
        default=0,
        help=(
            "0이면 defect type별 모든 overlay 저장"
        ),
    )

    args = parser.parse_args()

    patchcore_config = resolve_path(
        args.patchcore_config
    )

    synthetic_root = resolve_path(
        args.synthetic_root
    )

    output_root = resolve_path(
        args.output_root
    )

    with patchcore_config.open(
        "r",
        encoding="utf-8",
    ) as stream:

        settings = yaml.safe_load(
            stream
        )

    device = get_device(
        args.device
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

        model_dir = resolve_path(
            settings[
                "rois"
            ][
                roi_name
            ][
                "model_dir"
            ]
        )

        synthetic_metadata = (
            synthetic_root
            / "val"
            / roi_name
            / "metadata.csv"
        )

        if not synthetic_metadata.exists():
            raise FileNotFoundError(
                synthetic_metadata
            )

        summary = evaluate_roi(
            roi_name=roi_name,
            model_dir=model_dir,
            synthetic_metadata=(
                synthetic_metadata
            ),
            output_dir=(
                output_root
                / roi_name
            ),
            device=device,
            batch_size=(
                args.batch_size
            ),
            mask_threshold=(
                args.mask_threshold
            ),
            save_overlays=(
                args.save_overlays
            ),
            max_overlays_per_type=(
                args.max_overlays_per_type
            ),
        )

        summaries[
            roi_name
        ] = summary

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summaries,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 75
    )

    print(
        "[DONE] Anomaly map validation completed"
    )

    print(
        f"Output: {output_root}"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()
