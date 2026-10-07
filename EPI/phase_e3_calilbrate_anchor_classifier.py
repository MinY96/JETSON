"""
phase_e3_calibrate_anchor_classifier.py

Purpose
-------
Phase E-3에서 검출한 CST anchor들을 Ground Truth와 매칭하고,
각 anchor 주변의 motion signal feature를 추출하여

    Type1 vs Type2

분류에 가장 적합한 threshold rule을 자동 탐색한다.

Input
-----
1. motion_signals.csv
2. blade_in_out_timestamp.xlsx
3. detected_events.csv
   - phase_e3_extract_type2_clips.py --no-export 결과

Output
------
phase_e3_anchor_calibration/
    anchor_features.csv
    single_rule_results.csv
    two_rule_results.csv
    best_rules.csv

Optimization priority
---------------------
1. Type2 Recall = 1.0
2. Precision 최대
3. False Positive 최소
4. Rule 단순성
"""

from __future__ import annotations

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


BLADE_NAMES = [
    "blade1",
    "blade2",
    "blade3",
]


# ============================================================
# Configuration
# ============================================================

class Config:

    # Anchor 주변 feature 분석 범위
    #
    # Type2 sequence 전체를 충분히 포함시키기 위해
    # CST peak 기준 앞/뒤 3초
    feature_pre_sec = 3.0
    feature_post_sec = 3.0

    # GT event와 anchor matching 허용 margin
    gt_match_margin_sec = 1.0

    # threshold 후보 개수
    threshold_quantiles = 100

    # 2개 rule 조합 탐색 시
    # single rule 상위 N개 feature만 사용
    top_features_for_pair_search = 20

    # 최소 Type2 Recall
    min_recall = 1.0


# ============================================================
# Time parser
# ============================================================

def parse_video_time(value) -> float:

    if pd.isna(value):
        raise ValueError("Timestamp is empty")

    if hasattr(value, "hour"):

        return (
            value.hour * 3600
            + value.minute * 60
            + value.second
        )

    text = str(value).strip()
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


# ============================================================
# Load Ground Truth
# ============================================================

def load_gt(path):

    df = pd.read_excel(
        path,
        dtype={
            "type": str,
            "start_sec": str,
            "end_sec": str,
        },
    )

    df["type"] = (
        df["type"]
        .str.strip()
        .str.lower()
    )

    df["gt_start_sec"] = (
        df["start_sec"]
        .apply(parse_video_time)
    )

    df["gt_end_sec"] = (
        df["end_sec"]
        .apply(parse_video_time)
    )

    return (
        df
        .sort_values("gt_start_sec")
        .reset_index(drop=True)
    )


# ============================================================
# GT ↔ Anchor matching
# ============================================================

def match_anchors_to_gt(
    anchors,
    gt,
    cfg,
):

    """
    CST peak timestamp를 이용하여 GT event와 매칭.

    우선순위:
        1. GT start/end 안에 anchor 존재
        2. ± margin 안에 존재
        3. 가장 가까운 GT 선택

    하나의 GT는 하나의 anchor와만 매칭.
    """

    candidates = []

    for anchor_idx, anchor in anchors.iterrows():

        anchor_time = float(
            anchor["cst_peak_sec"]
        )

        for gt_idx, gt_row in gt.iterrows():

            start = float(
                gt_row["gt_start_sec"]
            )

            end = float(
                gt_row["gt_end_sec"]
            )

            if (
                anchor_time
                <
                start
                - cfg.gt_match_margin_sec
            ):
                continue

            if (
                anchor_time
                >
                end
                + cfg.gt_match_margin_sec
            ):
                continue

            # GT 구간 내부이면 distance=0 취급
            if start <= anchor_time <= end:

                distance = 0.0

            else:

                distance = min(
                    abs(anchor_time - start),
                    abs(anchor_time - end),
                )

            candidates.append(
                (
                    distance,
                    anchor_idx,
                    gt_idx,
                )
            )

    # 가장 가까운 것부터 greedy matching
    candidates.sort(
        key=lambda x: x[0]
    )

    used_anchor = set()
    used_gt = set()

    matches = {}

    for (
        distance,
        anchor_idx,
        gt_idx,
    ) in candidates:

        if anchor_idx in used_anchor:
            continue

        if gt_idx in used_gt:
            continue

        used_anchor.add(
            anchor_idx
        )

        used_gt.add(
            gt_idx
        )

        matches[
            anchor_idx
        ] = (
            gt_idx,
            distance,
        )

    return matches


# ============================================================
# Basic statistics
# ============================================================

def clean_array(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return values[
        np.isfinite(values)
    ]


def stat_mean(values):

    x = clean_array(values)

    if len(x) == 0:
        return np.nan

    return float(
        np.mean(x)
    )


def stat_min(values):

    x = clean_array(values)

    if len(x) == 0:
        return np.nan

    return float(
        np.min(x)
    )


def stat_max(values):

    x = clean_array(values)

    if len(x) == 0:
        return np.nan

    return float(
        np.max(x)
    )


def stat_percentile(
    values,
    q,
):

    x = clean_array(values)

    if len(x) == 0:
        return np.nan

    return float(
        np.percentile(
            x,
            q,
        )
    )


# ============================================================
# Peak helpers
# ============================================================

def max_peak(
    df,
    column,
):

    values = df[
        column
    ].to_numpy(
        dtype=np.float64
    )

    valid = np.isfinite(
        values
    )

    if not np.any(valid):
        return np.nan, np.nan

    valid_indices = np.where(
        valid
    )[0]

    local_idx = valid_indices[
        np.argmax(
            values[valid]
        )
    ]

    row = df.iloc[
        local_idx
    ]

    return (
        float(row[column]),
        float(row["time_sec"]),
    )


def min_peak(
    df,
    column,
):

    values = df[
        column
    ].to_numpy(
        dtype=np.float64
    )

    valid = np.isfinite(
        values
    )

    if not np.any(valid):
        return np.nan, np.nan

    valid_indices = np.where(
        valid
    )[0]

    local_idx = valid_indices[
        np.argmin(
            values[valid]
        )
    ]

    row = df.iloc[
        local_idx
    ]

    return (
        float(row[column]),
        float(row["time_sec"]),
    )


# ============================================================
# Anchor feature extraction
# ============================================================

def extract_anchor_features(
    signals,
    anchor,
    gt_row,
    cfg,
):

    anchor_time = float(
        anchor["cst_peak_sec"]
    )

    start_time = max(
        0.0,
        anchor_time
        - cfg.feature_pre_sec,
    )

    end_time = (
        anchor_time
        + cfg.feature_post_sec
    )

    window = signals[
        (
            signals["time_sec"]
            >= start_time
        )
        &
        (
            signals["time_sec"]
            <= end_time
        )
    ].copy()

    if len(window) == 0:
        return None

    result = {

        "event_id":
            anchor["event_id"],

        "type":
            gt_row["type"],

        "slot":
            gt_row["slot"],

        "anchor_sec":
            anchor_time,

        "gt_start_sec":
            gt_row[
                "gt_start_sec"
            ],

        "gt_end_sec":
            gt_row[
                "gt_end_sec"
            ],

        "window_start_sec":
            start_time,

        "window_end_sec":
            end_time,

        "cst_anchor_peak_dy":
            anchor[
                "cst_peak_dy"
            ],
    }

    # ========================================================
    # CST
    # ========================================================

    cst_dy = (
        window[
            "cst_dy"
        ].to_numpy(
            dtype=np.float64
        )
    )

    cst_abs = np.abs(
        cst_dy
    )

    result.update({

        "cst_dy_min":
            stat_min(
                cst_dy
            ),

        "cst_dy_max":
            stat_max(
                cst_dy
            ),

        "cst_abs_max":
            stat_max(
                cst_abs
            ),

        "cst_abs_mean":
            stat_mean(
                cst_abs
            ),

        "cst_abs_p90":
            stat_percentile(
                cst_abs,
                90,
            ),

        "cst_abs_p95":
            stat_percentile(
                cst_abs,
                95,
            ),

        "cst_phase_mean":
            stat_mean(
                window[
                    "cst_phase_response"
                ]
            ),

        "cst_phase_p50":
            stat_percentile(
                window[
                    "cst_phase_response"
                ],
                50,
            ),
    })

    # ========================================================
    # Blade
    # ========================================================

    for blade in BLADE_NAMES:

        prefix = (
            f"{blade}_"
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

        # ----------------------------------------------------
        # 전체 window
        # ----------------------------------------------------

        result[
            f"{blade}_motion_max"
        ] = stat_max(
            window[motion_col]
        )

        result[
            f"{blade}_motion_mean"
        ] = stat_mean(
            window[motion_col]
        )

        result[
            f"{blade}_motion_p90"
        ] = stat_percentile(
            window[motion_col],
            90,
        )

        result[
            f"{blade}_motion_p95"
        ] = stat_percentile(
            window[motion_col],
            95,
        )

        result[
            f"{blade}_diff_max"
        ] = stat_max(
            window[diff_col]
        )

        result[
            f"{blade}_diff_p95"
        ] = stat_percentile(
            window[diff_col],
            95,
        )

        radial_max, radial_max_time = (
            max_peak(
                window,
                radial_col,
            )
        )

        radial_min, radial_min_time = (
            min_peak(
                window,
                radial_col,
            )
        )

        result[
            f"{blade}_radial_max"
        ] = radial_max

        result[
            f"{blade}_radial_min"
        ] = radial_min

        result[
            f"{blade}_radial_max_rel_sec"
        ] = (
            radial_max_time
            - anchor_time
        )

        result[
            f"{blade}_radial_min_rel_sec"
        ] = (
            radial_min_time
            - anchor_time
        )

        result[
            f"{blade}_radial_abs_max"
        ] = stat_max(
            window[
                radial_abs_col
            ]
        )

        result[
            f"{blade}_radial_abs_p95"
        ] = stat_percentile(
            window[
                radial_abs_col
            ],
            95,
        )

        result[
            f"{blade}_active_max"
        ] = stat_max(
            window[
                active_col
            ]
        )

        result[
            f"{blade}_active_p95"
        ] = stat_percentile(
            window[
                active_col
            ],
            95,
        )

        result[
            f"{blade}_magnitude_max"
        ] = stat_max(
            window[
                magnitude_col
            ]
        )

        result[
            f"{blade}_magnitude_p95"
        ] = stat_percentile(
            window[
                magnitude_col
            ],
            95,
        )

        result[
            f"{blade}_flow_x_abs_p95"
        ] = stat_percentile(
            np.abs(
                window[
                    flow_x_col
                ]
            ),
            95,
        )

        result[
            f"{blade}_flow_y_abs_p95"
        ] = stat_percentile(
            np.abs(
                window[
                    flow_y_col
                ]
            ),
            95,
        )

        # ----------------------------------------------------
        # Anchor 전/후 분리
        # ----------------------------------------------------

        before = window[
            window[
                "time_sec"
            ]
            <
            anchor_time
        ]

        after = window[
            window[
                "time_sec"
            ]
            >=
            anchor_time
        ]

        result[
            f"{blade}_before_motion_max"
        ] = stat_max(
            before[
                motion_col
            ]
        )

        result[
            f"{blade}_after_motion_max"
        ] = stat_max(
            after[
                motion_col
            ]
        )

        result[
            f"{blade}_before_radial_max"
        ] = stat_max(
            before[
                radial_col
            ]
        )

        result[
            f"{blade}_before_radial_min"
        ] = stat_min(
            before[
                radial_col
            ]
        )

        result[
            f"{blade}_after_radial_max"
        ] = stat_max(
            after[
                radial_col
            ]
        )

        result[
            f"{blade}_after_radial_min"
        ] = stat_min(
            after[
                radial_col
            ]
        )

        result[
            f"{blade}_before_active_max"
        ] = stat_max(
            before[
                active_col
            ]
        )

        result[
            f"{blade}_after_active_max"
        ] = stat_max(
            after[
                active_col
            ]
        )

    return result


# ============================================================
# Classification metrics
# ============================================================

def evaluate_prediction(
    y_true,
    prediction,
):

    y_true = np.asarray(
        y_true,
        dtype=bool,
    )

    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    tp = int(
        np.sum(
            y_true
            &
            prediction
        )
    )

    tn = int(
        np.sum(
            ~y_true
            &
            ~prediction
        )
    )

    fp = int(
        np.sum(
            ~y_true
            &
            prediction
        )
    )

    fn = int(
        np.sum(
            y_true
            &
            ~prediction
        )
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    accuracy = (
        (tp + tn)
        /
        len(y_true)
        if len(y_true)
        else 0.0
    )

    return {

        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        "recall":
            recall,

        "precision":
            precision,

        "specificity":
            specificity,

        "accuracy":
            accuracy,
    }


# ============================================================
# Threshold candidates
# ============================================================

def build_thresholds(
    values,
    n_quantiles,
):

    values = clean_array(
        values
    )

    if len(values) == 0:
        return []

    unique = np.unique(
        values
    )

    # 데이터가 적으면 실제 값 사이 midpoint 사용
    if len(unique) <= 100:

        thresholds = []

        # 양 끝 threshold도 포함
        thresholds.append(
            unique[0]
            - 1e-9
        )

        for a, b in zip(
            unique[:-1],
            unique[1:],
        ):

            thresholds.append(
                (a + b) / 2.0
            )

        thresholds.append(
            unique[-1]
            + 1e-9
        )

        return thresholds

    quantiles = np.linspace(
        0.0,
        1.0,
        n_quantiles,
    )

    return np.unique(
        np.quantile(
            values,
            quantiles,
        )
    ).tolist()


# ============================================================
# Single rule search
# ============================================================

def search_single_rules(
    df,
    cfg,
):

    y_true = (
        df["type"]
        ==
        "type_2"
    ).to_numpy()

    ignore = {
        "event_id",
        "slot",

        "anchor_sec",

        "gt_start_sec",
        "gt_end_sec",

        "window_start_sec",
        "window_end_sec",
    }

    numeric_columns = (
        df
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
    )

    rows = []

    for feature in numeric_columns:

        if feature in ignore:
            continue

        values = (
            df[feature]
            .to_numpy(
                dtype=np.float64
            )
        )

        if np.any(
            ~np.isfinite(values)
        ):
            continue

        thresholds = (
            build_thresholds(
                values,
                cfg.threshold_quantiles,
            )
        )

        for threshold in thresholds:

            # --------------------------------------------
            # feature >= threshold
            # --------------------------------------------

            pred = (
                values
                >=
                threshold
            )

            metrics = (
                evaluate_prediction(
                    y_true,
                    pred,
                )
            )

            rows.append({

                "feature":
                    feature,

                "op":
                    ">=",

                "threshold":
                    threshold,

                **metrics,
            })

            # --------------------------------------------
            # feature <= threshold
            # --------------------------------------------

            pred = (
                values
                <=
                threshold
            )

            metrics = (
                evaluate_prediction(
                    y_true,
                    pred,
                )
            )

            rows.append({

                "feature":
                    feature,

                "op":
                    "<=",

                "threshold":
                    threshold,

                **metrics,
            })

    result = pd.DataFrame(
        rows
    )

    if len(result):

        result = result.sort_values(
            [
                "recall",
                "precision",
                "specificity",
                "accuracy",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        ).reset_index(
            drop=True
        )

    return result


# ============================================================
# Rule apply
# ============================================================

def apply_rule(
    df,
    feature,
    op,
    threshold,
):

    values = (
        df[feature]
        .to_numpy(
            dtype=np.float64
        )
    )

    if op == ">=":

        return (
            values
            >=
            threshold
        )

    if op == "<=":

        return (
            values
            <=
            threshold
        )

    raise ValueError(
        f"Unsupported op: {op}"
    )


# ============================================================
# Select candidate rules
# ============================================================

def select_feature_candidates(
    single_rules,
    cfg,
):

    """
    각 feature별 가장 좋은 Recall=1 rule 하나를 선택.

    동일 feature threshold 수십 개가 pair search에
    중복 투입되는 것을 방지.
    """

    perfect_recall = (
        single_rules[
            single_rules[
                "recall"
            ]
            >=
            cfg.min_recall
        ]
    )

    if len(perfect_recall) == 0:

        perfect_recall = (
            single_rules.copy()
        )

    candidates = []

    used_features = set()

    for _, row in (
        perfect_recall.iterrows()
    ):

        feature = row[
            "feature"
        ]

        if feature in used_features:
            continue

        candidates.append(
            row
        )

        used_features.add(
            feature
        )

        if (
            len(candidates)
            >=
            cfg.top_features_for_pair_search
        ):
            break

    return pd.DataFrame(
        candidates
    )


# ============================================================
# Two-rule AND search
# ============================================================

def search_two_rules(
    df,
    candidate_rules,
    cfg,
):

    y_true = (
        df["type"]
        ==
        "type_2"
    ).to_numpy()

    rows = []

    records = (
        candidate_rules
        .to_dict(
            "records"
        )
    )

    for rule1, rule2 in combinations(
        records,
        2,
    ):

        # 같은 feature는 skip
        if (
            rule1["feature"]
            ==
            rule2["feature"]
        ):
            continue

        pred1 = apply_rule(
            df,

            rule1[
                "feature"
            ],

            rule1[
                "op"
            ],

            rule1[
                "threshold"
            ],
        )

        pred2 = apply_rule(
            df,

            rule2[
                "feature"
            ],

            rule2[
                "op"
            ],

            rule2[
                "threshold"
            ],
        )

        prediction = (
            pred1
            &
            pred2
        )

        metrics = (
            evaluate_prediction(
                y_true,
                prediction,
            )
        )

        rows.append({

            "feature1":
                rule1[
                    "feature"
                ],

            "op1":
                rule1[
                    "op"
                ],

            "threshold1":
                rule1[
                    "threshold"
                ],

            "feature2":
                rule2[
                    "feature"
                ],

            "op2":
                rule2[
                    "op"
                ],

            "threshold2":
                rule2[
                    "threshold"
                ],

            **metrics,
        })

    result = pd.DataFrame(
        rows
    )

    if len(result):

        result = result.sort_values(
            [
                "recall",
                "precision",
                "specificity",
                "accuracy",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        ).reset_index(
            drop=True
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
        "--anchors",
        required=True,
        help="Phase E3 detected_events.csv",
    )

    parser.add_argument(
        "--output",
        default=
            "phase_e3_anchor_calibration",
    )

    parser.add_argument(
        "--pre-sec",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--post-sec",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()

    cfg = Config()

    cfg.feature_pre_sec = (
        args.pre_sec
    )

    cfg.feature_post_sec = (
        args.post_sec
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load
    # ========================================================

    print(
        "Loading signals..."
    )

    signals = pd.read_csv(
        args.signals
    )

    signals = (
        signals
        .sort_values(
            "time_sec"
        )
        .reset_index(
            drop=True
        )
    )

    anchors = pd.read_csv(
        args.anchors
    )

    gt = load_gt(
        args.timestamps
    )

    print(
        f"Signals : "
        f"{len(signals):,}"
    )

    print(
        f"Anchors : "
        f"{len(anchors)}"
    )

    print(
        f"GT      : "
        f"{len(gt)}"
    )

    # ========================================================
    # Matching
    # ========================================================

    matches = (
        match_anchors_to_gt(
            anchors,
            gt,
            cfg,
        )
    )

    print()
    print(
        "========================================"
    )
    print(
        "Anchor / GT Matching"
    )
    print(
        "========================================"
    )

    print(
        f"Matched anchors : "
        f"{len(matches)}/"
        f"{len(anchors)}"
    )

    unmatched_anchor_count = (
        len(anchors)
        -
        len(matches)
    )

    unmatched_gt_count = (
        len(gt)
        -
        len(
            set(
                gt_idx
                for gt_idx, _
                in matches.values()
            )
        )
    )

    print(
        f"Unmatched anchor : "
        f"{unmatched_anchor_count}"
    )

    print(
        f"Unmatched GT     : "
        f"{unmatched_gt_count}"
    )

    # ========================================================
    # Feature extraction
    # ========================================================

    rows = []

    for anchor_idx, (
        gt_idx,
        distance,
    ) in matches.items():

        anchor = anchors.loc[
            anchor_idx
        ]

        gt_row = gt.loc[
            gt_idx
        ]

        features = (
            extract_anchor_features(
                signals,
                anchor,
                gt_row,
                cfg,
            )
        )

        if features is None:
            continue

        features[
            "gt_match_distance_sec"
        ] = distance

        rows.append(
            features
        )

    feature_df = pd.DataFrame(
        rows
    )

    feature_df = (
        feature_df
        .sort_values(
            "anchor_sec"
        )
        .reset_index(
            drop=True
        )
    )

    feature_path = (
        output_dir
        /
        "anchor_features.csv"
    )

    feature_df.to_csv(
        feature_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Class distribution
    # ========================================================

    print()
    print(
        "Class distribution"
    )
    print(
        "----------------------------------------"
    )

    print(
        feature_df[
            "type"
        ].value_counts()
    )

    # ========================================================
    # Single Rule Search
    # ========================================================

    print()
    print(
        "Searching single rules..."
    )

    single_rules = (
        search_single_rules(
            feature_df,
            cfg,
        )
    )

    single_path = (
        output_dir
        /
        "single_rule_results.csv"
    )

    single_rules.to_csv(
        single_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Candidate Features
    # ========================================================

    candidates = (
        select_feature_candidates(
            single_rules,
            cfg,
        )
    )

    # ========================================================
    # Two Rule Search
    # ========================================================

    print(
        "Searching two-rule combinations..."
    )

    two_rules = (
        search_two_rules(
            feature_df,
            candidates,
            cfg,
        )
    )

    two_path = (
        output_dir
        /
        "two_rule_results.csv"
    )

    two_rules.to_csv(
        two_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print(
        "========================================"
    )
    print(
        "Calibration Result"
    )
    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Best single rule
    # --------------------------------------------------------

    if len(single_rules):

        perfect_single = (
            single_rules[
                single_rules[
                    "recall"
                ]
                >=
                cfg.min_recall
            ]
        )

        if len(perfect_single):

            best = (
                perfect_single.iloc[0]
            )

            print()
            print(
                "BEST SINGLE RULE"
            )

            print(
                "----------------------------------------"
            )

            print(
                f"{best['feature']} "
                f"{best['op']} "
                f"{best['threshold']:.8f}"
            )

            print(
                f"TP={int(best['tp'])} "
                f"TN={int(best['tn'])} "
                f"FP={int(best['fp'])} "
                f"FN={int(best['fn'])}"
            )

            print(
                f"Recall    = "
                f"{best['recall']:.4f}"
            )

            print(
                f"Precision = "
                f"{best['precision']:.4f}"
            )

    # --------------------------------------------------------
    # Best 2-rule
    # --------------------------------------------------------

    if len(two_rules):

        perfect_pair = (
            two_rules[
                two_rules[
                    "recall"
                ]
                >=
                cfg.min_recall
            ]
        )

        if len(perfect_pair):

            best = (
                perfect_pair.iloc[0]
            )

            print()
            print(
                "BEST TWO-RULE"
            )

            print(
                "----------------------------------------"
            )

            print(
                f"{best['feature1']} "
                f"{best['op1']} "
                f"{best['threshold1']:.8f}"
            )

            print(
                "AND"
            )

            print(
                f"{best['feature2']} "
                f"{best['op2']} "
                f"{best['threshold2']:.8f}"
            )

            print(
                f"TP={int(best['tp'])} "
                f"TN={int(best['tn'])} "
                f"FP={int(best['fp'])} "
                f"FN={int(best['fn'])}"
            )

            print(
                f"Recall    = "
                f"{best['recall']:.4f}"
            )

            print(
                f"Precision = "
                f"{best['precision']:.4f}"
            )

    # ========================================================
    # Top rules
    # ========================================================

    print()
    print(
        "Top 10 Single Rules"
    )
    print(
        "----------------------------------------"
    )

    if len(single_rules):

        print(
            single_rules[
                [
                    "feature",
                    "op",
                    "threshold",
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                    "recall",
                    "precision",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Top 10 Two-Rules"
    )
    print(
        "----------------------------------------"
    )

    if len(two_rules):

        print(
            two_rules[
                [
                    "feature1",
                    "op1",
                    "threshold1",

                    "feature2",
                    "op2",
                    "threshold2",

                    "tp",
                    "tn",
                    "fp",
                    "fn",

                    "recall",
                    "precision",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Output"
    )
    print(
        "----------------------------------------"
    )

    print(
        feature_path
    )

    print(
        single_path
    )

    print(
        two_path
    )


if __name__ == "__main__":
    main()
