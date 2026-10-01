from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# Configuration
# ============================================================

METRIC_COLUMNS = [
    "good_matches",
    "inliers",
    "inlier_ratio",
    "area_ratio",
    "rotation_deg",
]

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


def nan_to_none(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None

    return value


# ============================================================
# Load Dataset
# ============================================================

def load_metadata(
    csv_path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
    )

    required_columns = {
        "manual_label",
        "match_status",
        *METRIC_COLUMNS,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns in {csv_path}: "
            f"{sorted(missing)}"
        )

    # ----------------------------------------
    # Normalize labels
    # ----------------------------------------

    df["manual_label"] = (
        df["manual_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
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

    df["match_status"] = (
        df["match_status"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # ----------------------------------------
    # Numeric conversion
    # ----------------------------------------

    for column in METRIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Calibration에서는 visible / not_visible만 사용
    df_labeled = df[
        df["manual_label"].isin(
            VALID_LABELS
        )
    ].copy()

    # visible = positive
    df_labeled["target"] = (
        df_labeled[
            "manual_label"
        ] == "visible"
    )

    return df_labeled


# ============================================================
# Metric calculation
# ============================================================

def calculate_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> GateMetrics:

    target = target.astype(bool)
    prediction = prediction.astype(bool)

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

    f1 = safe_div(
        2 * precision * recall,
        precision + recall,
    )

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
# Baseline: API matched status
# ============================================================

def evaluate_status_baseline(
    df: pd.DataFrame,
) -> GateMetrics:

    target = (
        df["target"]
        .to_numpy(dtype=bool)
    )

    prediction = (
        df["match_status"]
        .eq("matched")
        .to_numpy(dtype=bool)
    )

    return calculate_metrics(
        target,
        prediction,
    )


# ============================================================
# Threshold gate
# ============================================================

def apply_gate(
    df: pd.DataFrame,
    threshold: GateThreshold,
) -> np.ndarray:

    # Homography가 만들어진 matched 상태만 Gate 후보
    prediction = (
        df["match_status"]
        .eq("matched")
        .to_numpy(dtype=bool)
    )

    if threshold.min_good_matches is not None:
        values = (
            df["good_matches"]
            .to_numpy(dtype=float)
        )

        prediction &= (
            np.isfinite(values)
            & (
                values
                >= threshold.min_good_matches
            )
        )

    if threshold.min_inliers is not None:
        values = (
            df["inliers"]
            .to_numpy(dtype=float)
        )

        prediction &= (
            np.isfinite(values)
            & (
                values
                >= threshold.min_inliers
            )
        )

    if threshold.min_inlier_ratio is not None:
        values = (
            df["inlier_ratio"]
            .to_numpy(dtype=float)
        )

        prediction &= (
            np.isfinite(values)
            & (
                values
                >= threshold.min_inlier_ratio
            )
        )

    if threshold.min_area_ratio is not None:
        values = (
            df["area_ratio"]
            .to_numpy(dtype=float)
        )

        prediction &= (
            np.isfinite(values)
            & (
                values
                >= threshold.min_area_ratio
            )
        )

    if threshold.max_area_ratio is not None:
        values = (
            df["area_ratio"]
            .to_numpy(dtype=float)
        )

        prediction &= (
            np.isfinite(values)
            & (
                values
                <= threshold.max_area_ratio
            )
        )

    if threshold.max_abs_rotation is not None:
        values = np.abs(
            df["rotation_deg"]
            .to_numpy(dtype=float)
        )

        prediction &= (
            np.isfinite(values)
            & (
                values
                <= threshold.max_abs_rotation
            )
        )

    return prediction


# ============================================================
# Threshold candidates
# ============================================================

def unique_float_candidates(
    series: pd.Series,
    quantiles: list[float],
    *,
    include_none: bool = True,
    decimals: int = 4,
) -> list[float | None]:

    values = (
        series
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(values) == 0:
        return [None]

    candidates: list[
        float | None
    ] = []

    if include_none:
        candidates.append(None)

    quantile_values = np.quantile(
        values,
        quantiles,
    )

    for value in quantile_values:
        value = round(
            float(value),
            decimals,
        )

        if value not in candidates:
            candidates.append(value)

    return candidates


def unique_int_candidates(
    series: pd.Series,
    quantiles: list[float],
    *,
    include_none: bool = True,
) -> list[int | None]:

    values = (
        series
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(values) == 0:
        return [None]

    candidates: list[
        int | None
    ] = []

    if include_none:
        candidates.append(None)

    quantile_values = np.quantile(
        values,
        quantiles,
    )

    for value in quantile_values:
        candidate = int(
            round(
                float(value)
            )
        )

        if candidate not in candidates:
            candidates.append(
                candidate
            )

    return candidates


def build_candidate_space(
    df: pd.DataFrame,
) -> dict[str, list[Any]]:

    # Homography metric이 실제 존재하는 matched samples를 기준으로
    # threshold 후보를 생성한다.
    matched = df[
        df["match_status"]
        == "matched"
    ]

    lower_quantiles = [
        0.00,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
    ]

    upper_quantiles = [
        0.50,
        0.70,
        0.80,
        0.90,
        0.95,
        1.00,
    ]

    good_matches = (
        unique_int_candidates(
            matched[
                "good_matches"
            ],
            lower_quantiles,
        )
    )

    inliers = (
        unique_int_candidates(
            matched[
                "inliers"
            ],
            lower_quantiles,
        )
    )

    inlier_ratio = (
        unique_float_candidates(
            matched[
                "inlier_ratio"
            ],
            lower_quantiles,
        )
    )

    min_area_ratio = (
        unique_float_candidates(
            matched[
                "area_ratio"
            ],
            [
                0.00,
                0.02,
                0.05,
                0.10,
                0.20,
            ],
        )
    )

    max_area_ratio = (
        unique_float_candidates(
            matched[
                "area_ratio"
            ],
            upper_quantiles,
        )
    )

    abs_rotation = (
        matched[
            "rotation_deg"
        ]
        .abs()
    )

    max_rotation = (
        unique_float_candidates(
            abs_rotation,
            upper_quantiles,
            decimals=2,
        )
    )

    return {
        "min_good_matches":
            good_matches,

        "min_inliers":
            inliers,

        "min_inlier_ratio":
            inlier_ratio,

        "min_area_ratio":
            min_area_ratio,

        "max_area_ratio":
            max_area_ratio,

        "max_abs_rotation":
            max_rotation,
    }


# ============================================================
# Search
# ============================================================

def threshold_complexity(
    threshold: GateThreshold,
) -> int:
    return sum(
        value is not None
        for value in asdict(
            threshold
        ).values()
    )


def search_thresholds(
    df: pd.DataFrame,
    *,
    target_recall: float,
) -> pd.DataFrame:

    space = build_candidate_space(
        df
    )

    target = (
        df["target"]
        .to_numpy(dtype=bool)
    )

    results: list[
        dict[str, Any]
    ] = []

    combinations = product(
        space[
            "min_good_matches"
        ],
        space[
            "min_inliers"
        ],
        space[
            "min_inlier_ratio"
        ],
        space[
            "min_area_ratio"
        ],
        space[
            "max_area_ratio"
        ],
        space[
            "max_abs_rotation"
        ],
    )

    checked = 0

    for (
        min_good_matches,
        min_inliers,
        min_inlier_ratio,
        min_area_ratio,
        max_area_ratio,
        max_abs_rotation,
    ) in combinations:

        # 잘못된 area range 제외
        if (
            min_area_ratio is not None
            and max_area_ratio is not None
            and min_area_ratio
            >= max_area_ratio
        ):
            continue

        threshold = GateThreshold(
            min_good_matches=(
                min_good_matches
            ),
            min_inliers=(
                min_inliers
            ),
            min_inlier_ratio=(
                min_inlier_ratio
            ),
            min_area_ratio=(
                min_area_ratio
            ),
            max_area_ratio=(
                max_area_ratio
            ),
            max_abs_rotation=(
                max_abs_rotation
            ),
        )

        prediction = apply_gate(
            df,
            threshold,
        )

        metrics = calculate_metrics(
            target,
            prediction,
        )

        checked += 1

        if (
            not np.isfinite(
                metrics.recall
            )
            or metrics.recall
            < target_recall
        ):
            continue

        result = {
            **asdict(
                threshold
            ),
            **asdict(
                metrics
            ),
            "complexity":
                threshold_complexity(
                    threshold
                ),
        }

        results.append(
            result
        )

    print(
        f"[SEARCH] evaluated "
        f"{checked:,} threshold combinations"
    )

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(
        results
    )

    # ------------------------------------------------------
    # Selection priority
    #
    # 1. target Recall 조건을 이미 만족
    # 2. Specificity 최대화
    # 3. Precision 최대화
    # 4. Recall 최대화
    # 5. 조건이 단순한 Gate 선호
    # ------------------------------------------------------

    result_df = result_df.sort_values(
        by=[
            "specificity",
            "precision",
            "recall",
            "balanced_accuracy",
            "complexity",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return result_df


# ============================================================
# Analysis
# ============================================================

def print_dataset_summary(
    df: pd.DataFrame,
    roi_name: str,
) -> None:

    print()
    print(
        "=" * 70
    )

    print(
        f"ROI: {roi_name.upper()}"
    )

    print(
        "=" * 70
    )

    print(
        f"Labeled samples : "
        f"{len(df)}"
    )

    print()
    print(
        "[Manual Label]"
    )

    print(
        df[
            "manual_label"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "[Match Status x Manual Label]"
    )

    status_table = pd.crosstab(
        df["match_status"],
        df["manual_label"],
        margins=True,
    )

    print(
        status_table.to_string()
    )

    print()

    if (
        df[
            "localization_label"
        ]
        .ne("")
        .any()
    ):
        print(
            "[Localization Label x Manual Label]"
        )

        localization_table = (
            pd.crosstab(
                df[
                    "localization_label"
                ],
                df[
                    "manual_label"
                ],
                margins=True,
            )
        )

        print(
            localization_table.to_string()
        )

        print()


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


# ============================================================
# Plots
# ============================================================

def save_distribution_plots(
    df: pd.DataFrame,
    output_dir: Path,
    roi_name: str,
) -> None:

    plot_dir = (
        output_dir
        / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for metric in METRIC_COLUMNS:

        visible = (
            df.loc[
                df["manual_label"]
                == "visible",
                metric,
            ]
            .dropna()
            .to_numpy()
        )

        not_visible = (
            df.loc[
                df["manual_label"]
                == "not_visible",
                metric,
            ]
            .dropna()
            .to_numpy()
        )

        if (
            len(visible) == 0
            and len(not_visible) == 0
        ):
            continue

        fig = plt.figure(
            figsize=(8, 5)
        )

        ax = fig.add_subplot(
            111
        )

        data = []
        labels = []

        if len(visible):
            data.append(
                visible
            )
            labels.append(
                "visible"
            )

        if len(not_visible):
            data.append(
                not_visible
            )
            labels.append(
                "not_visible"
            )

        ax.boxplot(
            data,
            labels=labels,
            showfliers=True,
        )

        ax.set_title(
            f"{roi_name.upper()} - {metric}"
        )

        ax.set_ylabel(
            metric
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            plot_dir
            / f"{metric}.png",
            dpi=160,
        )

        plt.close(
            fig
        )


def save_scatter_plot(
    df: pd.DataFrame,
    output_dir: Path,
    roi_name: str,
) -> None:

    plot_dir = (
        output_dir
        / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matched = df[
        df["match_status"]
        == "matched"
    ].copy()

    if matched.empty:
        return

    fig = plt.figure(
        figsize=(8, 6)
    )

    ax = fig.add_subplot(
        111
    )

    for label in [
        "visible",
        "not_visible",
    ]:

        subset = matched[
            matched[
                "manual_label"
            ]
            == label
        ]

        ax.scatter(
            subset[
                "good_matches"
            ],
            subset[
                "inlier_ratio"
            ],
            label=label,
            alpha=0.65,
        )

    ax.set_xlabel(
        "good_matches"
    )

    ax.set_ylabel(
        "inlier_ratio"
    )

    ax.set_title(
        f"{roi_name.upper()} - "
        "Good Matches vs Inlier Ratio"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        plot_dir
        / "good_matches_vs_inlier_ratio.png",
        dpi=160,
    )

    plt.close(
        fig
    )


# ============================================================
# Error case output
# ============================================================

def save_error_cases(
    df: pd.DataFrame,
    prediction: np.ndarray,
    output_dir: Path,
) -> None:

    result = df.copy()

    result[
        "gate_prediction"
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
            "gate_prediction"
        ]
    )

    false_negative = result[
        (
            result[
                "manual_label"
            ]
            == "visible"
        )
        & (
            result[
                "gate_prediction"
            ]
            == "not_visible"
        )
    ]

    false_positive = result[
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
            == "visible"
        )
    ]

    false_negative.to_csv(
        output_dir
        / "missed_visible.csv",
        index=False,
        encoding="utf-8-sig",
    )

    false_positive.to_csv(
        output_dir
        / "false_visible.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# Save selected gate
# ============================================================

def row_to_threshold(
    row: pd.Series,
) -> GateThreshold:

    def optional_int(
        value: Any,
    ) -> int | None:

        if pd.isna(
            value
        ):
            return None

        return int(
            value
        )

    def optional_float(
        value: Any,
    ) -> float | None:

        if pd.isna(
            value
        ):
            return None

        return float(
            value
        )

    return GateThreshold(
        min_good_matches=optional_int(
            row[
                "min_good_matches"
            ]
        ),
        min_inliers=optional_int(
            row[
                "min_inliers"
            ]
        ),
        min_inlier_ratio=optional_float(
            row[
                "min_inlier_ratio"
            ]
        ),
        min_area_ratio=optional_float(
            row[
                "min_area_ratio"
            ]
        ),
        max_area_ratio=optional_float(
            row[
                "max_area_ratio"
            ]
        ),
        max_abs_rotation=optional_float(
            row[
                "max_abs_rotation"
            ]
        ),
    )


def save_selected_gate(
    roi_name: str,
    threshold: GateThreshold,
    metrics: GateMetrics,
    output_dir: Path,
    *,
    target_recall: float,
) -> None:

    payload = {
        roi_name: {
            "thresholds": {
                key: nan_to_none(
                    value
                )
                for key, value
                in asdict(
                    threshold
                ).items()
            },
            "calibration_metrics": {
                key: nan_to_none(
                    value
                )
                for key, value
                in asdict(
                    metrics
                ).items()
            },
            "target_visible_recall":
                target_recall,

            "status_required":
                "matched",
        }
    }

    with (
        output_dir
        / "selected_gate.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as stream:

        yaml.safe_dump(
            payload,
            stream,
            allow_unicode=True,
            sort_keys=False,
        )


# ============================================================
# Process ROI
# ============================================================

def calibrate_roi(
    *,
    roi_name: str,
    csv_path: Path,
    output_root: Path,
    target_recall: float,
    top_k: int,
) -> None:

    output_dir = (
        output_root
        / roi_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_metadata(
        csv_path
    )

    if df.empty:
        raise RuntimeError(
            f"No labeled data found: "
            f"{csv_path}"
        )

    print_dataset_summary(
        df,
        roi_name,
    )

    # --------------------------------------------------------
    # Current API status baseline
    # --------------------------------------------------------

    baseline = (
        evaluate_status_baseline(
            df
        )
    )

    print_metrics(
        "STATUS BASELINE "
        "(matched = visible)",
        baseline,
    )

    # --------------------------------------------------------
    # Cross tables
    # --------------------------------------------------------

    status_table = pd.crosstab(
        df["match_status"],
        df["manual_label"],
        margins=True,
    )

    status_table.to_csv(
        output_dir
        / "status_vs_label.csv",
        encoding="utf-8-sig",
    )

    localization_table = (
        pd.crosstab(
            df[
                "localization_label"
            ],
            df[
                "manual_label"
            ],
            margins=True,
        )
    )

    localization_table.to_csv(
        output_dir
        / "localization_vs_label.csv",
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Distribution statistics
    # --------------------------------------------------------

    stats = (
        df.groupby(
            "manual_label"
        )[
            METRIC_COLUMNS
        ]
        .describe()
    )

    stats.to_csv(
        output_dir
        / "metric_statistics.csv",
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    save_distribution_plots(
        df,
        output_dir,
        roi_name,
    )

    save_scatter_plot(
        df,
        output_dir,
        roi_name,
    )

    # --------------------------------------------------------
    # Threshold search
    # --------------------------------------------------------

    candidates = search_thresholds(
        df,
        target_recall=(
            target_recall
        ),
    )

    if candidates.empty:
        print()
        print(
            "[WARN] No threshold "
            f"satisfied recall >= "
            f"{target_recall:.3f}"
        )

        print(
            "The current matching/localization "
            "stage itself may be missing too many "
            "visible frames."
        )

        return

    candidates.to_csv(
        output_dir
        / "gate_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_candidates = (
        candidates
        .head(
            top_k
        )
        .copy()
    )

    top_candidates.to_csv(
        output_dir
        / "top_gate_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"[TOP {min(top_k, len(top_candidates))} "
        f"GATE CANDIDATES]"
    )

    print()

    display_columns = [
        "min_good_matches",
        "min_inliers",
        "min_inlier_ratio",
        "min_area_ratio",
        "max_area_ratio",
        "max_abs_rotation",

        "recall",
        "specificity",
        "precision",
        "accuracy",

        "tp",
        "tn",
        "fp",
        "fn",
    ]

    print(
        top_candidates[
            display_columns
        ]
        .to_string(
            index=False,
        )
    )

    # --------------------------------------------------------
    # First candidate
    # --------------------------------------------------------

    selected_row = (
        candidates.iloc[0]
    )

    selected_threshold = (
        row_to_threshold(
            selected_row
        )
    )

    prediction = apply_gate(
        df,
        selected_threshold,
    )

    selected_metrics = (
        calculate_metrics(
            df[
                "target"
            ]
            .to_numpy(
                dtype=bool
            ),
            prediction,
        )
    )

    print_metrics(
        "SELECTED CANDIDATE",
        selected_metrics,
    )

    print()
    print(
        "[SELECTED THRESHOLD]"
    )

    for key, value in (
        asdict(
            selected_threshold
        ).items()
    ):
        print(
            f"{key:22s}: {value}"
        )

    save_selected_gate(
        roi_name,
        selected_threshold,
        selected_metrics,
        output_dir,
        target_recall=(
            target_recall
        ),
    )

    save_error_cases(
        df,
        prediction,
        output_dir,
    )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {
        "roi":
            roi_name,

        "labeled_samples":
            len(df),

        "visible_samples":
            int(
                (
                    df[
                        "manual_label"
                    ]
                    == "visible"
                ).sum()
            ),

        "not_visible_samples":
            int(
                (
                    df[
                        "manual_label"
                    ]
                    == "not_visible"
                ).sum()
            ),

        "target_recall":
            target_recall,

        "baseline":
            {
                key:
                    nan_to_none(
                        value
                    )
                for key, value
                in asdict(
                    baseline
                ).items()
            },

        "selected_threshold":
            {
                key:
                    nan_to_none(
                        value
                    )
                for key, value
                in asdict(
                    selected_threshold
                ).items()
            },

        "selected_metrics":
            {
                key:
                    nan_to_none(
                        value
                    )
                for key, value
                in asdict(
                    selected_metrics
                ).items()
            },
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


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Calibrate CCTV inspection gate "
            "from manually labeled metadata"
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "data/gate_dataset/train"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/gate_calibration"
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
        "--target-recall",
        type=float,
        default=0.98,
        help=(
            "Minimum visible recall. "
            "Default: 0.98"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if not (
        0.0
        < args.target_recall
        <= 1.0
    ):
        raise ValueError(
            "--target-recall must be "
            "between 0 and 1"
        )

    dataset_dir = (
        resolve_project_path(
            args.dataset_dir
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

    for roi_name in roi_names:

        csv_path = (
            dataset_dir
            / roi_name
            / "metadata.csv"
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                csv_path
            )

        calibrate_roi(
            roi_name=roi_name,
            csv_path=csv_path,
            output_root=(
                output_root
            ),
            target_recall=(
                args.target_recall
            ),
            top_k=(
                args.top_k
            ),
        )

    print()
    print(
        "=" * 70
    )

    print(
        "[DONE] Gate calibration completed"
    )

    print(
        f"Results: {output_root}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
