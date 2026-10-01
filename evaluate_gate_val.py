from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


VALID_LABELS = {
    "visible",
    "not_visible",
}


# ============================================================
# Dataclasses
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
class GateMetrics:
    tp: int
    tn: int
    fp: int
    fn: int

    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    balanced_accuracy: float


# ============================================================
# Utility
# ============================================================

def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def safe_div(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return float("nan")

    return numerator / denominator


def json_safe(value: Any) -> Any:

    if isinstance(
        value,
        (np.integer,),
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,),
    ):
        value = float(value)

    if (
        isinstance(value, float)
        and math.isnan(value)
    ):
        return None

    return value


# ============================================================
# Load metadata
# ============================================================

def load_metadata(
    csv_path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
    )

    required = {
        "manual_label",
        "match_status",
        "good_matches",
        "inliers",
        "inlier_ratio",
        "area_ratio",
        "rotation_deg",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["manual_label"] = (
        df["manual_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["match_status"] = (
        df["match_status"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if "localization_label" not in df.columns:
        df["localization_label"] = ""

    df["localization_label"] = (
        df["localization_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    numeric_columns = [
        "good_matches",
        "inliers",
        "inlier_ratio",
        "area_ratio",
        "rotation_deg",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ignore / unlabeled 제외
    df = df[
        df["manual_label"].isin(
            VALID_LABELS
        )
    ].copy()

    df["target"] = (
        df["manual_label"]
        .eq("visible")
    )

    return df


# ============================================================
# Load selected gate
# ============================================================

def load_gate_threshold(
    yaml_path: Path,
    roi_name: str,
) -> tuple[GateThreshold, str]:

    with yaml_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        data = yaml.safe_load(stream)

    if not data:
        raise ValueError(
            f"Empty YAML: {yaml_path}"
        )

    if roi_name not in data:
        raise KeyError(
            f"ROI '{roi_name}' not found "
            f"in {yaml_path}"
        )

    roi_config = data[
        roi_name
    ]

    threshold_data = (
        roi_config.get(
            "thresholds",
            {}
        )
    )

    status_required = (
        roi_config.get(
            "status_required",
            "matched",
        )
    )

    def optional_int(
        name: str,
    ) -> int | None:

        value = threshold_data.get(
            name
        )

        if value is None:
            return None

        return int(value)

    def optional_float(
        name: str,
    ) -> float | None:

        value = threshold_data.get(
            name
        )

        if value is None:
            return None

        return float(value)

    threshold = GateThreshold(
        min_good_matches=optional_int(
            "min_good_matches"
        ),
        min_inliers=optional_int(
            "min_inliers"
        ),
        min_inlier_ratio=optional_float(
            "min_inlier_ratio"
        ),
        min_area_ratio=optional_float(
            "min_area_ratio"
        ),
        max_area_ratio=optional_float(
            "max_area_ratio"
        ),
        max_abs_rotation=optional_float(
            "max_abs_rotation"
        ),
    )

    return (
        threshold,
        status_required,
    )


# ============================================================
# Gate
# ============================================================

def apply_gate(
    df: pd.DataFrame,
    threshold: GateThreshold,
    status_required: str,
) -> np.ndarray:

    prediction = (
        df["match_status"]
        .eq(status_required)
        .to_numpy(
            dtype=bool,
            copy=True,
        )
    )

    if (
        threshold.min_good_matches
        is not None
    ):
        values = (
            df["good_matches"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                values
                >= threshold.min_good_matches
            )
        )

        prediction = (
            prediction
            & condition
        )

    if (
        threshold.min_inliers
        is not None
    ):
        values = (
            df["inliers"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                values
                >= threshold.min_inliers
            )
        )

        prediction = (
            prediction
            & condition
        )

    if (
        threshold.min_inlier_ratio
        is not None
    ):
        values = (
            df["inlier_ratio"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                values
                >= threshold.min_inlier_ratio
            )
        )

        prediction = (
            prediction
            & condition
        )

    if (
        threshold.min_area_ratio
        is not None
    ):
        values = (
            df["area_ratio"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                values
                >= threshold.min_area_ratio
            )
        )

        prediction = (
            prediction
            & condition
        )

    if (
        threshold.max_area_ratio
        is not None
    ):
        values = (
            df["area_ratio"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                values
                <= threshold.max_area_ratio
            )
        )

        prediction = (
            prediction
            & condition
        )

    if (
        threshold.max_abs_rotation
        is not None
    ):
        values = (
            df["rotation_deg"]
            .to_numpy(
                dtype=float,
                copy=False,
            )
        )

        condition = (
            np.isfinite(values)
            & (
                np.abs(values)
                <= threshold.max_abs_rotation
            )
        )

        prediction = (
            prediction
            & condition
        )

    return prediction


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> GateMetrics:

    target = np.asarray(
        target,
        dtype=bool,
    )

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    tp = int(
        np.sum(
            target & prediction
        )
    )

    tn = int(
        np.sum(
            ~target & ~prediction
        )
    )

    fp = int(
        np.sum(
            ~target & prediction
        )
    )

    fn = int(
        np.sum(
            target & ~prediction
        )
    )

    accuracy = safe_div(
        tp + tn,
        tp + tn + fp + fn,
    )

    precision = safe_div(
        tp,
        tp + fp,
    )

    recall = safe_div(
        tp,
        tp + fn,
    )

    specificity = safe_div(
        tn,
        tn + fp,
    )

    if (
        np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall > 0
    ):
        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
    else:
        f1 = float("nan")

    balanced_accuracy = np.nanmean(
        [
            recall,
            specificity,
        ]
    )

    return GateMetrics(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=float(
            accuracy
        ),
        precision=float(
            precision
        ),
        recall=float(
            recall
        ),
        specificity=float(
            specificity
        ),
        f1=float(
            f1
        ),
        balanced_accuracy=float(
            balanced_accuracy
        ),
    )


# ============================================================
# Output
# ============================================================

def print_metrics(
    title: str,
    metrics: GateMetrics,
) -> None:

    print()
    print(
        f"[{title}]"
    )

    print(
        f"TP={metrics.tp}  "
        f"TN={metrics.tn}  "
        f"FP={metrics.fp}  "
        f"FN={metrics.fn}"
    )

    print(
        f"Accuracy          : "
        f"{metrics.accuracy:.4f}"
    )

    print(
        f"Precision         : "
        f"{metrics.precision:.4f}"
    )

    print(
        f"Visible Recall    : "
        f"{metrics.recall:.4f}"
    )

    print(
        f"NotVisible Recall : "
        f"{metrics.specificity:.4f}"
    )

    print(
        f"F1                : "
        f"{metrics.f1:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{metrics.balanced_accuracy:.4f}"
    )


def save_error_cases(
    result: pd.DataFrame,
    output_dir: Path,
) -> None:

    missed_visible = result[
        (
            result[
                "manual_label"
            ]
            == "visible"
        )
        & (
            ~result[
                "gate_prediction"
            ]
        )
    ]

    false_visible = result[
        (
            result[
                "manual_label"
            ]
            == "not_visible"
        )
        & (
            result[
                "gate_prediction"
            ]
        )
    ]

    missed_visible.to_csv(
        output_dir
        / "missed_visible.csv",
        index=False,
        encoding="utf-8-sig",
    )

    false_visible.to_csv(
        output_dir
        / "false_visible.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Per video evaluation
# ============================================================

def evaluate_per_video(
    result: pd.DataFrame,
) -> pd.DataFrame:

    if "video_name" not in result.columns:
        return pd.DataFrame()

    rows: list[
        dict[str, Any]
    ] = []

    for video_name, group in (
        result.groupby(
            "video_name",
            sort=True,
        )
    ):

        metrics = calculate_metrics(
            group[
                "target"
            ].to_numpy(
                dtype=bool
            ),
            group[
                "gate_prediction"
            ].to_numpy(
                dtype=bool
            ),
        )

        rows.append(
            {
                "video_name":
                    video_name,

                "samples":
                    len(group),

                "visible":
                    int(
                        group[
                            "target"
                        ].sum()
                    ),

                "not_visible":
                    int(
                        (
                            ~group[
                                "target"
                            ]
                        ).sum()
                    ),

                **asdict(
                    metrics
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ROI evaluation
# ============================================================

def evaluate_roi(
    *,
    roi_name: str,
    dataset_dir: Path,
    gate_root: Path,
    output_root: Path,
) -> dict[str, Any]:

    csv_path = (
        dataset_dir
        / roi_name
        / "metadata.csv"
    )

    gate_yaml = (
        gate_root
        / roi_name
        / "selected_gate.yaml"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            csv_path
        )

    if not gate_yaml.exists():
        raise FileNotFoundError(
            gate_yaml
        )

    df = load_metadata(
        csv_path
    )

    if df.empty:
        raise RuntimeError(
            f"No valid labels found: "
            f"{csv_path}"
        )

    threshold, status_required = (
        load_gate_threshold(
            gate_yaml,
            roi_name,
        )
    )

    print()
    print(
        "=" * 72
    )

    print(
        f"VAL ROI: "
        f"{roi_name.upper()}"
    )

    print(
        "=" * 72
    )

    print(
        f"Labeled samples : "
        f"{len(df)}"
    )

    print(
        f"Visible         : "
        f"{int(df['target'].sum())}"
    )

    print(
        f"Not visible     : "
        f"{int((~df['target']).sum())}"
    )

    print()

    print(
        "[FIXED THRESHOLD]"
    )

    print(
        f"status_required        : "
        f"{status_required}"
    )

    for key, value in (
        asdict(
            threshold
        ).items()
    ):
        print(
            f"{key:22s}: "
            f"{value}"
        )

    target = (
        df["target"]
        .to_numpy(
            dtype=bool
        )
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_prediction = (
        df["match_status"]
        .eq("matched")
        .to_numpy(
            dtype=bool,
            copy=True,
        )
    )

    baseline_metrics = (
        calculate_metrics(
            target,
            baseline_prediction,
        )
    )

    print_metrics(
        "VAL STATUS BASELINE",
        baseline_metrics,
    )

    # --------------------------------------------------------
    # Fixed Gate
    # --------------------------------------------------------

    prediction = apply_gate(
        df,
        threshold,
        status_required,
    )

    metrics = calculate_metrics(
        target,
        prediction,
    )

    print_metrics(
        "VAL FIXED GATE",
        metrics,
    )

    # --------------------------------------------------------
    # Result table
    # --------------------------------------------------------

    result = df.copy()

    result[
        "gate_prediction"
    ] = prediction

    result[
        "gate_prediction_label"
    ] = np.where(
        prediction,
        "visible",
        "not_visible",
    )

    result[
        "correct"
    ] = (
        result[
            "manual_label"
        ]
        == result[
            "gate_prediction_label"
        ]
    )

    output_dir = (
        output_root
        / roi_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_dir
        / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_error_cases(
        result,
        output_dir,
    )

    # --------------------------------------------------------
    # Per video
    # --------------------------------------------------------

    per_video = (
        evaluate_per_video(
            result
        )
    )

    if not per_video.empty:
        per_video.to_csv(
            output_dir
            / "per_video_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print()
        print(
            "[PER VIDEO]"
        )

        display_columns = [
            "video_name",
            "samples",
            "visible",
            "not_visible",
            "recall",
            "specificity",
            "fp",
            "fn",
        ]

        print(
            per_video[
                display_columns
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Localization bad check
    # --------------------------------------------------------

    localization_bad = result[
        (
            result[
                "localization_label"
            ]
            == "bad"
        )
        & (
            result[
                "gate_prediction"
            ]
        )
    ]

    localization_bad.to_csv(
        output_dir
        / "gate_true_localization_bad.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "Gate=True + Localization=bad : "
        f"{len(localization_bad)}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "roi":
            roi_name,

        "samples":
            len(result),

        "visible":
            int(
                result[
                    "target"
                ].sum()
            ),

        "not_visible":
            int(
                (
                    ~result[
                        "target"
                    ]
                ).sum()
            ),

        "threshold":
            {
                key:
                    json_safe(
                        value
                    )
                for key, value
                in asdict(
                    threshold
                ).items()
            },

        "status_required":
            status_required,

        "baseline_metrics":
            {
                key:
                    json_safe(
                        value
                    )
                for key, value
                in asdict(
                    baseline_metrics
                ).items()
            },

        "fixed_gate_metrics":
            {
                key:
                    json_safe(
                        value
                    )
                for key, value
                in asdict(
                    metrics
                ).items()
            },

        "gate_true_localization_bad":
            len(
                localization_bad
            ),
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed Gate thresholds "
            "on validation dataset"
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "data/gate_dataset/val"
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
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/gate_validation"
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

    dataset_dir = (
        resolve_project_path(
            args.dataset_dir
        )
    )

    gate_root = (
        resolve_project_path(
            args.gate_dir
        )
    )

    output_root = (
        resolve_project_path(
            args.output_dir
        )
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
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
        ] = evaluate_roi(
            roi_name=roi_name,
            dataset_dir=dataset_dir,
            gate_root=gate_root,
            output_root=output_root,
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
        "=" * 72
    )

    print(
        "[DONE] Validation evaluation completed"
    )

    print(
        f"Results: {output_root}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
