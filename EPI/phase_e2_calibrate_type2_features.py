"""
Phase E-2
Type1 / Type2 Motion Feature Calibration
=========================================

Input
-----
1. motion_signals.csv
2. blade_in_out_timestamp.xlsx

Ground Truth Excel columns
--------------------------
type      : type_1 / type_2
start_sec : '55:29, '01:03:25 ...
end_sec   : same format
slot      : 1 ~ 25

Processing
----------
GT interval에 margin을 추가:

    analysis_start = start_sec - margin
    analysis_end   = end_sec   + margin

각 이벤트마다:

- CST motion feature
- Blade ROI 1/2/3 motion feature
- radial feature
- temporal / sequence feature

추출.

Output
------
phase_e2_result/
    event_features.csv
    feature_separation.csv
    roi_summary.csv

이 단계에서는 최종 Type2 rule을 확정하지 않는다.
실제 feature separation을 확인한 뒤 E-2 v2에서
Type2 sequence rule을 calibration한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BLADE_NAMES = [
    "blade1",
    "blade2",
    "blade3",
]


# ============================================================
# Time parser
# ============================================================

def parse_video_time(value) -> float:
    """
    Excel에서 다음과 같이 저장된 값을 초 단위로 변환.

    '55:29
    '43:39
    '01:03:25

    Excel / pandas에서 apostrophe가 제거되어
    55:29 형태로 들어와도 처리한다.
    """

    if pd.isna(value):
        raise ValueError(
            "Timestamp is empty."
        )

    # Excel이 datetime/time으로 읽어버린 경우도 방어
    if hasattr(value, "hour"):
        return (
            value.hour * 3600
            + value.minute * 60
            + value.second
        )

    text = str(value).strip()

    # 사용자가 Excel에서 text 강제를 위해
    # 넣은 apostrophe 제거
    text = text.lstrip("'")

    parts = text.split(":")

    if len(parts) == 2:

        minute = int(parts[0])
        second = float(parts[1])

        return (
            minute * 60
            + second
        )

    if len(parts) == 3:

        hour = int(parts[0])
        minute = int(parts[1])
        second = float(parts[2])

        return (
            hour * 3600
            + minute * 60
            + second
        )

    raise ValueError(
        f"Invalid timestamp: {value}"
    )


def format_video_time(
    sec: float,
) -> str:

    sec = max(
        0.0,
        float(sec),
    )

    hour = int(
        sec // 3600
    )

    minute = int(
        (sec % 3600) // 60
    )

    second = (
        sec % 60
    )

    return (
        f"{hour:02d}:"
        f"{minute:02d}:"
        f"{second:06.3f}"
    )


# ============================================================
# Basic statistics
# ============================================================

def safe_percentile(
    values,
    q,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            q,
        )
    )


def safe_mean(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def safe_max(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.max(values)
    )


def safe_min(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.min(values)
    )


# ============================================================
# Peak
# ============================================================

def find_max_peak(
    event_df,
    column,
):

    if len(event_df) == 0:
        return np.nan, np.nan

    values = event_df[
        column
    ].to_numpy(
        dtype=np.float64
    )

    valid = np.isfinite(
        values
    )

    if not np.any(valid):
        return np.nan, np.nan

    valid_idx = np.where(
        valid
    )[0]

    local_idx = valid_idx[
        np.argmax(
            values[valid]
        )
    ]

    row = event_df.iloc[
        local_idx
    ]

    return (
        float(row[column]),
        float(row["time_sec"]),
    )


def find_min_peak(
    event_df,
    column,
):

    if len(event_df) == 0:
        return np.nan, np.nan

    values = event_df[
        column
    ].to_numpy(
        dtype=np.float64
    )

    valid = np.isfinite(
        values
    )

    if not np.any(valid):
        return np.nan, np.nan

    valid_idx = np.where(
        valid
    )[0]

    local_idx = valid_idx[
        np.argmin(
            values[valid]
        )
    ]

    row = event_df.iloc[
        local_idx
    ]

    return (
        float(row[column]),
        float(row["time_sec"]),
    )


def find_abs_peak(
    event_df,
    column,
):

    if len(event_df) == 0:
        return np.nan, np.nan

    values = event_df[
        column
    ].to_numpy(
        dtype=np.float64
    )

    valid = np.isfinite(
        values
    )

    if not np.any(valid):
        return np.nan, np.nan

    valid_idx = np.where(
        valid
    )[0]

    local_idx = valid_idx[
        np.argmax(
            np.abs(
                values[valid]
            )
        )
    ]

    row = event_df.iloc[
        local_idx
    ]

    return (
        float(row[column]),
        float(row["time_sec"]),
    )


# ============================================================
# CST Features
# ============================================================

def extract_cst_features(
    event_df,
):

    dy = event_df[
        "cst_dy"
    ].to_numpy(
        dtype=np.float64
    )

    abs_dy = np.abs(
        dy
    )

    (
        dy_max,
        dy_max_time,
    ) = find_max_peak(
        event_df,
        "cst_dy",
    )

    (
        dy_min,
        dy_min_time,
    ) = find_min_peak(
        event_df,
        "cst_dy",
    )

    (
        dy_abs_peak,
        dy_abs_peak_time,
    ) = find_abs_peak(
        event_df,
        "cst_dy",
    )

    response = event_df[
        "cst_phase_response"
    ].to_numpy(
        dtype=np.float64
    )

    return {

        "cst_dy_max":
            dy_max,

        "cst_dy_max_time":
            dy_max_time,

        "cst_dy_min":
            dy_min,

        "cst_dy_min_time":
            dy_min_time,

        "cst_dy_abs_peak":
            abs(dy_abs_peak),

        "cst_dy_abs_peak_signed":
            dy_abs_peak,

        "cst_dy_abs_peak_time":
            dy_abs_peak_time,

        "cst_dy_abs_mean":
            safe_mean(
                abs_dy
            ),

        "cst_dy_abs_p90":
            safe_percentile(
                abs_dy,
                90,
            ),

        "cst_dy_abs_p95":
            safe_percentile(
                abs_dy,
                95,
            ),

        "cst_phase_response_mean":
            safe_mean(
                response
            ),

        "cst_phase_response_p50":
            safe_percentile(
                response,
                50,
            ),
    }


# ============================================================
# Blade Features
# ============================================================

def extract_blade_features(
    event_df,
    blade_name,
):

    prefix = (
        f"{blade_name}_"
    )

    motion_col = (
        prefix
        + "motion_ratio"
    )

    diff_col = (
        prefix
        + "mean_diff"
    )

    radial_col = (
        prefix
        + "radial_score"
    )

    radial_abs_col = (
        prefix
        + "radial_abs"
    )

    active_col = (
        prefix
        + "active_ratio"
    )

    magnitude_col = (
        prefix
        + "median_magnitude"
    )

    flow_x_col = (
        prefix
        + "flow_x"
    )

    flow_y_col = (
        prefix
        + "flow_y"
    )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    motion = event_df[
        motion_col
    ].to_numpy(
        dtype=np.float64
    )

    mean_diff = event_df[
        diff_col
    ].to_numpy(
        dtype=np.float64
    )

    radial = event_df[
        radial_col
    ].to_numpy(
        dtype=np.float64
    )

    radial_abs = event_df[
        radial_abs_col
    ].to_numpy(
        dtype=np.float64
    )

    active = event_df[
        active_col
    ].to_numpy(
        dtype=np.float64
    )

    magnitude = event_df[
        magnitude_col
    ].to_numpy(
        dtype=np.float64
    )

    flow_x = event_df[
        flow_x_col
    ].to_numpy(
        dtype=np.float64
    )

    flow_y = event_df[
        flow_y_col
    ].to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Peak timing
    # --------------------------------------------------------

    (
        radial_max,
        radial_max_time,
    ) = find_max_peak(
        event_df,
        radial_col,
    )

    (
        radial_min,
        radial_min_time,
    ) = find_min_peak(
        event_df,
        radial_col,
    )

    (
        motion_max,
        motion_max_time,
    ) = find_max_peak(
        event_df,
        motion_col,
    )

    # --------------------------------------------------------
    # Event temporal split
    #
    # 전체 event를 앞/뒤 절반으로 나눠서
    # IN / OUT 후보 신호를 별도로 본다.
    # --------------------------------------------------------

    start_time = float(
        event_df[
            "time_sec"
        ].iloc[0]
    )

    end_time = float(
        event_df[
            "time_sec"
        ].iloc[-1]
    )

    mid_time = (
        start_time
        +
        end_time
    ) / 2.0

    first_half = event_df[
        event_df[
            "time_sec"
        ] <= mid_time
    ]

    second_half = event_df[
        event_df[
            "time_sec"
        ] > mid_time
    ]

    if len(first_half):

        first_radial = (
            first_half[
                radial_col
            ].to_numpy(
                dtype=np.float64
            )
        )

        first_motion = (
            first_half[
                motion_col
            ].to_numpy(
                dtype=np.float64
            )
        )

    else:

        first_radial = np.array(
            []
        )

        first_motion = np.array(
            []
        )

    if len(second_half):

        second_radial = (
            second_half[
                radial_col
            ].to_numpy(
                dtype=np.float64
            )
        )

        second_motion = (
            second_half[
                motion_col
            ].to_numpy(
                dtype=np.float64
            )
        )

    else:

        second_radial = np.array(
            []
        )

        second_motion = np.array(
            []
        )

    return {

        # Motion
        f"{blade_name}_motion_max":
            safe_max(
                motion
            ),

        f"{blade_name}_motion_p95":
            safe_percentile(
                motion,
                95,
            ),

        f"{blade_name}_motion_mean":
            safe_mean(
                motion
            ),

        f"{blade_name}_motion_peak_time":
            motion_max_time,

        # Difference
        f"{blade_name}_diff_max":
            safe_max(
                mean_diff
            ),

        f"{blade_name}_diff_p95":
            safe_percentile(
                mean_diff,
                95,
            ),

        # Radial
        f"{blade_name}_radial_max":
            radial_max,

        f"{blade_name}_radial_max_time":
            radial_max_time,

        f"{blade_name}_radial_min":
            radial_min,

        f"{blade_name}_radial_min_time":
            radial_min_time,

        f"{blade_name}_radial_abs_max":
            safe_max(
                radial_abs
            ),

        f"{blade_name}_radial_abs_p95":
            safe_percentile(
                radial_abs,
                95,
            ),

        # Flow
        f"{blade_name}_flow_x_abs_p95":
            safe_percentile(
                np.abs(flow_x),
                95,
            ),

        f"{blade_name}_flow_y_abs_p95":
            safe_percentile(
                np.abs(flow_y),
                95,
            ),

        # Optical flow activity
        f"{blade_name}_active_max":
            safe_max(
                active
            ),

        f"{blade_name}_active_p95":
            safe_percentile(
                active,
                95,
            ),

        f"{blade_name}_magnitude_max":
            safe_max(
                magnitude
            ),

        f"{blade_name}_magnitude_p95":
            safe_percentile(
                magnitude,
                95,
            ),

        # --------------------------------------------
        # Temporal features
        # --------------------------------------------

        f"{blade_name}_first_radial_max":
            safe_max(
                first_radial
            ),

        f"{blade_name}_first_radial_min":
            safe_min(
                first_radial
            ),

        f"{blade_name}_second_radial_max":
            safe_max(
                second_radial
            ),

        f"{blade_name}_second_radial_min":
            safe_min(
                second_radial
            ),

        f"{blade_name}_first_motion_max":
            safe_max(
                first_motion
            ),

        f"{blade_name}_second_motion_max":
            safe_max(
                second_motion
            ),
    }


# ============================================================
# Sequence Features
# ============================================================

def add_sequence_features(
    features,
    blade_name,
):

    cst_time = features[
        "cst_dy_abs_peak_time"
    ]

    radial_max_time = features[
        f"{blade_name}_radial_max_time"
    ]

    radial_min_time = features[
        f"{blade_name}_radial_min_time"
    ]

    # --------------------------------------------------------
    # 시간 차이
    # --------------------------------------------------------

    features[
        f"{blade_name}_radial_max_to_cst_sec"
    ] = (
        cst_time
        - radial_max_time
    )

    features[
        f"{blade_name}_cst_to_radial_min_sec"
    ] = (
        radial_min_time
        - cst_time
    )

    # --------------------------------------------------------
    # 순서
    #
    # radial positive
    #      ↓
    # CST movement
    #      ↓
    # radial negative
    #
    # 이 가정이 실제 데이터에서 맞는지 검증하기 위한 feature.
    # 아직 판정에는 사용하지 않는다.
    # --------------------------------------------------------

    features[
        f"{blade_name}_sequence_pos_cst_neg"
    ] = int(
        radial_max_time
        < cst_time
        < radial_min_time
    )

    return features


# ============================================================
# Event Feature Extraction
# ============================================================

def extract_event_features(
    signal_df,
    gt_row,
    margin_sec,
    event_id,
):

    gt_start = (
        gt_row[
            "start_sec_value"
        ]
    )

    gt_end = (
        gt_row[
            "end_sec_value"
        ]
    )

    analysis_start = max(
        0.0,
        gt_start
        - margin_sec,
    )

    analysis_end = (
        gt_end
        + margin_sec
    )

    event_df = signal_df[
        (
            signal_df[
                "time_sec"
            ]
            >= analysis_start
        )
        &
        (
            signal_df[
                "time_sec"
            ]
            <= analysis_end
        )
    ].copy()

    if len(event_df) == 0:

        raise RuntimeError(
            f"No signal data for "
            f"event {event_id}: "
            f"{analysis_start:.3f}"
            f" ~ "
            f"{analysis_end:.3f}"
        )

    features = {

        "event_id":
            event_id,

        "type":
            gt_row["type"],

        "slot":
            gt_row["slot"],

        # Ground Truth
        "gt_start_sec":
            gt_start,

        "gt_end_sec":
            gt_end,

        "gt_duration_sec":
            gt_end
            - gt_start,

        "gt_start_time":
            format_video_time(
                gt_start
            ),

        "gt_end_time":
            format_video_time(
                gt_end
            ),

        # Margin 적용 분석 구간
        "analysis_start_sec":
            analysis_start,

        "analysis_end_sec":
            analysis_end,

        "analysis_start_time":
            format_video_time(
                analysis_start
            ),

        "analysis_end_time":
            format_video_time(
                analysis_end
            ),

        "analysis_duration_sec":
            analysis_end
            - analysis_start,

        "num_signal_frames":
            len(event_df),
    }

    # --------------------------------------------------------
    # CST
    # --------------------------------------------------------

    features.update(
        extract_cst_features(
            event_df
        )
    )

    # --------------------------------------------------------
    # Blade
    # --------------------------------------------------------

    for blade_name in BLADE_NAMES:

        features.update(
            extract_blade_features(
                event_df,
                blade_name,
            )
        )

        features = (
            add_sequence_features(
                features,
                blade_name,
            )
        )

    return features


# ============================================================
# Separation metric
# ============================================================

def calculate_separation(
    event_df,
):

    """
    각 numeric feature가 Type1 / Type2를
    얼마나 분리하는지 간단히 평가한다.

    effect_size:
        |mean2 - mean1| / pooled_std

    값이 클수록 두 class 분리가 잘 됨.

    direction:
        type2_higher
        type2_lower
    """

    ignored = {
        "event_id",
        "slot",

        "gt_start_sec",
        "gt_end_sec",
        "gt_duration_sec",

        "analysis_start_sec",
        "analysis_end_sec",
        "analysis_duration_sec",

        "num_signal_frames",
    }

    rows = []

    numeric_columns = (
        event_df
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
    )

    for column in numeric_columns:

        if column in ignored:
            continue

        type1 = (
            event_df[
                event_df["type"]
                == "type_1"
            ][column]
            .dropna()
            .to_numpy(
                dtype=np.float64
            )
        )

        type2 = (
            event_df[
                event_df["type"]
                == "type_2"
            ][column]
            .dropna()
            .to_numpy(
                dtype=np.float64
            )
        )

        if (
            len(type1) < 2
            or
            len(type2) < 2
        ):
            continue

        mean1 = float(
            np.mean(type1)
        )

        mean2 = float(
            np.mean(type2)
        )

        std1 = float(
            np.std(
                type1,
                ddof=1,
            )
        )

        std2 = float(
            np.std(
                type2,
                ddof=1,
            )
        )

        pooled_std = np.sqrt(
            (
                std1 ** 2
                +
                std2 ** 2
            )
            / 2.0
        )

        if pooled_std > 1e-12:

            effect_size = (
                abs(
                    mean2
                    - mean1
                )
                /
                pooled_std
            )

        else:

            effect_size = 0.0

        direction = (
            "type2_higher"
            if mean2 > mean1
            else "type2_lower"
        )

        rows.append({

            "feature":
                column,

            "type1_mean":
                mean1,

            "type1_std":
                std1,

            "type1_min":
                float(
                    np.min(type1)
                ),

            "type1_max":
                float(
                    np.max(type1)
                ),

            "type2_mean":
                mean2,

            "type2_std":
                std2,

            "type2_min":
                float(
                    np.min(type2)
                ),

            "type2_max":
                float(
                    np.max(type2)
                ),

            "effect_size":
                effect_size,

            "direction":
                direction,
        })

    result = pd.DataFrame(
        rows
    )

    if len(result):

        result = (
            result
            .sort_values(
                "effect_size",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    return result


# ============================================================
# ROI summary
# ============================================================

def build_roi_summary(
    separation_df,
):

    rows = []

    for blade_name in BLADE_NAMES:

        roi_features = (
            separation_df[
                separation_df[
                    "feature"
                ].str.startswith(
                    f"{blade_name}_"
                )
            ]
        )

        if len(roi_features) == 0:
            continue

        top = (
            roi_features
            .sort_values(
                "effect_size",
                ascending=False,
            )
            .head(10)
        )

        rows.append({

            "blade_roi":
                blade_name,

            "best_feature":
                top.iloc[0][
                    "feature"
                ],

            "best_effect_size":
                top.iloc[0][
                    "effect_size"
                ],

            "top5_mean_effect_size":
                top.head(5)[
                    "effect_size"
                ].mean(),

            "top10_mean_effect_size":
                top[
                    "effect_size"
                ].mean(),
        })

    result = pd.DataFrame(
        rows
    )

    if len(result):

        result = (
            result
            .sort_values(
                "top5_mean_effect_size",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--signals",
        required=True,
        help="motion_signals.csv",
    )

    parser.add_argument(
        "--timestamps",
        required=True,
        help="blade_in_out_timestamp.xlsx",
    )

    parser.add_argument(
        "--output",
        default="phase_e2_result",
    )

    parser.add_argument(
        "--margin-sec",
        type=float,
        default=0.5,
        help=(
            "Margin added before start_sec "
            "and after end_sec"
        ),
    )

    args = parser.parse_args()

    signal_path = Path(
        args.signals
    )

    timestamp_path = Path(
        args.timestamps
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load signal
    # ========================================================

    print(
        "Loading motion signals..."
    )

    signal_df = pd.read_csv(
        signal_path
    )

    signal_df = (
        signal_df
        .sort_values(
            "time_sec"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Signal rows: "
        f"{len(signal_df):,}"
    )

    # ========================================================
    # Validate signal columns
    # ========================================================

    required_signal_columns = [
        "frame_idx",
        "time_sec",

        "cst_dx",
        "cst_dy",
        "cst_phase_response",
    ]

    for blade in BLADE_NAMES:

        for suffix in [
            "motion_ratio",
            "mean_diff",
            "flow_x",
            "flow_y",
            "radial_score",
            "radial_abs",
            "active_ratio",
            "median_magnitude",
        ]:

            required_signal_columns.append(
                f"{blade}_{suffix}"
            )

    missing = [
        c
        for c
        in required_signal_columns
        if c not in signal_df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns in "
            "motion_signals.csv:\n"
            + "\n".join(
                missing
            )
        )

    # ========================================================
    # Load Ground Truth
    # ========================================================

    print(
        "Loading timestamp Ground Truth..."
    )

    gt_df = pd.read_excel(
        timestamp_path,
        dtype={
            "type": str,
            "start_sec": str,
            "end_sec": str,
        },
    )

    required_gt_columns = {
        "type",
        "start_sec",
        "end_sec",
        "slot",
    }

    missing_gt = (
        required_gt_columns
        - set(
            gt_df.columns
        )
    )

    if missing_gt:

        raise ValueError(
            "Missing columns in timestamp Excel: "
            + str(
                missing_gt
            )
        )

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    gt_df["type"] = (
        gt_df["type"]
        .str.strip()
        .str.lower()
    )

    gt_df = gt_df[
        gt_df["type"].isin(
            [
                "type_1",
                "type_2",
            ]
        )
    ].copy()

    # --------------------------------------------------------
    # Parse timestamp
    # --------------------------------------------------------

    gt_df[
        "start_sec_value"
    ] = gt_df[
        "start_sec"
    ].apply(
        parse_video_time
    )

    gt_df[
        "end_sec_value"
    ] = gt_df[
        "end_sec"
    ].apply(
        parse_video_time
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    invalid = gt_df[
        gt_df[
            "end_sec_value"
        ]
        <=
        gt_df[
            "start_sec_value"
        ]
    ]

    if len(invalid):

        print()
        print(
            "Invalid timestamp rows:"
        )

        print(
            invalid[
                [
                    "type",
                    "start_sec",
                    "end_sec",
                    "slot",
                ]
            ]
        )

        raise ValueError(
            "end_sec must be "
            "greater than start_sec."
        )

    print()
    print(
        "Ground Truth:"
    )

    print(
        gt_df[
            "type"
        ].value_counts()
    )

    print(
        f"Margin: "
        f"{args.margin_sec:.3f} sec"
    )

    # ========================================================
    # Extract Event Features
    # ========================================================

    event_features = []

    for idx, gt_row in (
        gt_df.iterrows()
    ):

        event_id = (
            f"{gt_row['type']}_"
            f"slot{int(gt_row['slot']):02d}"
        )

        print(
            f"Extracting "
            f"{event_id} "
            f"{gt_row['start_sec']} "
            f"~ "
            f"{gt_row['end_sec']}"
        )

        features = (
            extract_event_features(
                signal_df,
                gt_row,
                args.margin_sec,
                event_id,
            )
        )

        event_features.append(
            features
        )

    event_df = pd.DataFrame(
        event_features
    )

    # ========================================================
    # Save Event Features
    # ========================================================

    event_path = (
        output_dir
        / "event_features.csv"
    )

    event_df.to_csv(
        event_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Feature Separation
    # ========================================================

    separation_df = (
        calculate_separation(
            event_df
        )
    )

    separation_path = (
        output_dir
        / "feature_separation.csv"
    )

    separation_df.to_csv(
        separation_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # ROI Summary
    # ========================================================

    roi_summary = (
        build_roi_summary(
            separation_df
        )
    )

    roi_path = (
        output_dir
        / "roi_summary.csv"
    )

    roi_summary.to_csv(
        roi_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Console Summary
    # ========================================================

    print()
    print(
        "========================================="
    )
    print(
        "Phase E-2 Complete"
    )
    print(
        "========================================="
    )

    print(
        f"Events             : "
        f"{len(event_df)}"
    )

    print(
        f"Margin             : "
        f"{args.margin_sec:.3f} sec"
    )

    print(
        f"Event features     : "
        f"{event_path}"
    )

    print(
        f"Feature separation : "
        f"{separation_path}"
    )

    print(
        f"ROI summary        : "
        f"{roi_path}"
    )

    # --------------------------------------------------------
    # Top features
    # --------------------------------------------------------

    if len(separation_df):

        print()
        print(
            "Top 20 separation features"
        )
        print(
            "-----------------------------------------"
        )

        print(
            separation_df[
                [
                    "feature",
                    "effect_size",
                    "direction",
                    "type1_mean",
                    "type2_mean",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # ROI ranking
    # --------------------------------------------------------

    if len(roi_summary):

        print()
        print(
            "Blade ROI ranking"
        )
        print(
            "-----------------------------------------"
        )

        print(
            roi_summary.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
