"""
phase_e3_extract_type2_clips.py

Phase E-3 v3
============

목적
----
Phase E-1의 motion_signals.csv를 이용하여
장시간 CCTV 영상에서 Type2 이벤트를 자동 검출하고
Phase A에서 사용할 Type2 clip을 추출한다.

Type2 sequence
--------------
Blade IN
    ↓
CST UP
    ↓
CST STOP
    ↓
Blade OUT

Phase E-3에서는 정확한 CST STOP/measurement frame은 찾지 않는다.
정확한 measurement frame 검출은 Phase A에서 수행한다.

핵심 변경사항
-------------
기존 방식:
    특정 cst_dy band에 들어온 frame을 먼저 선택
        ↓
    grouping

문제:
    큰 CST movement도 -14 → -7 → -5 → 0처럼 이동하면서
    Type2 band를 통과하기 때문에 가짜 anchor 발생.

현재 방식:
    cst_dy <= -2인 모든 CST negative movement 검출
        ↓
    temporal grouping
        ↓
    각 movement event 전체에서 min(cst_dy) 계산
        ↓
    event peak(min dy)를 이용해 Type2 선별

현재 데이터:
    Type2 event peak:
        약 -4.875 ~ -5.186

    다른 큰 CST movement:
        약 -10.86 이하

따라서 기본 Type2 event 조건:
    -8 < event_min_dy <= -2

Input
-----
motion_signals.csv
blade_in_out_timestamp.xlsx
original.mp4

Output
------
output/
    detected_events.csv
    gt_comparison.csv
    clips/
        type2_001.mp4
        ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:

    # ========================================================
    # CST movement detection
    # ========================================================

    # 모든 의미 있는 negative CST movement 검출
    cst_motion_threshold: float = -2.0

    # 가까운 CST motion frame을 하나의 event로 연결
    cst_group_gap_sec: float = 0.40

    # 최소 event duration
    cst_min_event_sec: float = 0.03

    # ========================================================
    # Type2 event classification
    # ========================================================

    # 중요:
    # 이 threshold는 frame에 적용하는 것이 아니라
    # CST movement event 전체의 min(cst_dy)에 적용한다.
    #
    # 실제 Type2:
    #     약 -4.875 ~ -5.186
    #
    # 다른 큰 CST movement:
    #     약 -10.86 이하

    cst_type2_peak_min: float = -8.0
    cst_type2_peak_max: float = -2.0

    # ========================================================
    # Blade boundary
    # ========================================================

    blade_name: str = "blade2"

    # CST event 이전 Blade IN 탐색 범위
    blade_in_search_sec: float = 3.0

    # CST event 이후 Blade OUT 탐색 범위
    blade_out_search_sec: float = 3.0

    # Blade local adaptive threshold
    blade_motion_quantile: float = 0.75

    blade_motion_min_threshold: float = 0.015

    blade_min_motion_sec: float = 0.06

    blade_group_gap_sec: float = 0.20

    # ========================================================
    # Clip
    # ========================================================

    pre_margin_sec: float = 0.50

    post_margin_sec: float = 0.50

    # Blade boundary 검출 실패 시
    # CST event 기준 fallback
    fallback_pre_sec: float = 3.0

    fallback_post_sec: float = 3.0

    # ========================================================
    # GT
    # ========================================================

    gt_match_tolerance_sec: float = 3.0


# ============================================================
# Time utilities
# ============================================================

def parse_video_time(value) -> float:

    if pd.isna(value):
        raise ValueError(
            "Timestamp is empty"
        )

    # datetime.time 등
    if hasattr(value, "hour"):

        return (
            value.hour * 3600
            + value.minute * 60
            + value.second
        )

    text = (
        str(value)
        .strip()
        .lstrip("'")
    )

    parts = text.split(":")

    # MM:SS
    if len(parts) == 2:

        minute = int(
            parts[0]
        )

        second = float(
            parts[1]
        )

        return (
            minute * 60
            + second
        )

    # HH:MM:SS
    if len(parts) == 3:

        hour = int(
            parts[0]
        )

        minute = int(
            parts[1]
        )

        second = float(
            parts[2]
        )

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
        (sec % 3600)
        // 60
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
# Temporal grouping
# ============================================================

def find_boolean_groups(
    df: pd.DataFrame,
    mask: np.ndarray,
    max_gap_sec: float,
    min_duration_sec: float,
):
    """
    True mask를 시간 기준으로 group화한다.

    True frame 사이의 시간 차이가 max_gap_sec 이하이면
    동일 movement event로 연결한다.
    """

    indices = np.where(
        mask
    )[0]

    if len(indices) == 0:
        return []

    groups = []

    current = [
        indices[0]
    ]

    for idx in indices[1:]:

        prev_idx = (
            current[-1]
        )

        prev_time = float(
            df.iloc[
                prev_idx
            ]["time_sec"]
        )

        curr_time = float(
            df.iloc[
                idx
            ]["time_sec"]
        )

        gap = (
            curr_time
            - prev_time
        )

        if gap <= max_gap_sec:

            current.append(
                idx
            )

        else:

            groups.append(
                current
            )

            current = [
                idx
            ]

    groups.append(
        current
    )

    result = []

    for group in groups:

        start_idx = (
            group[0]
        )

        end_idx = (
            group[-1]
        )

        start_time = float(
            df.iloc[
                start_idx
            ]["time_sec"]
        )

        end_time = float(
            df.iloc[
                end_idx
            ]["time_sec"]
        )

        duration = (
            end_time
            - start_time
        )

        # 15 FPS에서는 실제 짧은 movement가
        # 1~2 frame일 수 있으므로
        # duration 또는 frame count 조건 사용
        if (
            duration
            >= min_duration_sec
            or
            len(group) >= 2
        ):

            result.append({

                "start_idx":
                    start_idx,

                "end_idx":
                    end_idx,

                "start_time":
                    start_time,

                "end_time":
                    end_time,

                "duration":
                    duration,

                "indices":
                    group,
            })

    return result


# ============================================================
# CST movement detection
# ============================================================

def detect_cst_up_events(
    signals: pd.DataFrame,
    cfg: Config,
):
    """
    2-stage CST detection.

    Stage 1
    -------
    cst_dy <= cst_motion_threshold인 frame을 검출하고
    시간적으로 하나의 CST movement로 묶는다.

    Stage 2
    -------
    각 movement 전체에서 실제 minimum cst_dy를 계산한다.

    Type2 event:
        cst_type2_peak_min
            <
        event_min_dy
            <=
        cst_type2_peak_max

    중요
    ----
    Type2 peak band를 frame에 직접 적용하지 않는다.

    큰 CST movement도 이동 중 -6~-4 등의 영역을
    통과하기 때문에 frame-level band filtering을 하면
    false candidate가 발생한다.
    """

    dy = (
        signals[
            "cst_dy"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # ========================================================
    # Stage 1
    # 모든 negative CST movement 검출
    # ========================================================

    motion_mask = (
        dy
        <=
        cfg.cst_motion_threshold
    )

    groups = find_boolean_groups(
        signals,
        motion_mask,
        max_gap_sec=
            cfg.cst_group_gap_sec,
        min_duration_sec=
            cfg.cst_min_event_sec,
    )

    print()
    print(
        f"Raw CST movement events: "
        f"{len(groups)}"
    )

    all_events = []

    type2_events = []

    # ========================================================
    # Stage 2
    # 각 movement 전체 peak 계산
    # ========================================================

    for (
        raw_event_id,
        group,
    ) in enumerate(
        groups,
        start=1,
    ):

        start_idx = (
            group[
                "start_idx"
            ]
        )

        end_idx = (
            group[
                "end_idx"
            ]
        )

        local = signals.iloc[
            start_idx:
            end_idx + 1
        ]

        # ----------------------------------------------------
        # movement event 전체에서 가장 작은 cst_dy
        # ----------------------------------------------------

        peak_idx = (
            local[
                "cst_dy"
            ]
            .idxmin()
        )

        peak_row = (
            signals.loc[
                peak_idx
            ]
        )

        peak_dy = float(
            peak_row[
                "cst_dy"
            ]
        )

        peak_time = float(
            peak_row[
                "time_sec"
            ]
        )

        event = {

            "raw_event_id":
                raw_event_id,

            "cst_start_time":
                float(
                    group[
                        "start_time"
                    ]
                ),

            "cst_end_time":
                float(
                    group[
                        "end_time"
                    ]
                ),

            "cst_peak_time":
                peak_time,

            "cst_peak_dy":
                peak_dy,

            "cst_peak_frame":
                int(
                    peak_row[
                        "frame_idx"
                    ]
                ),

            "cst_duration_sec":
                float(
                    group[
                        "duration"
                    ]
                ),
        }

        all_events.append(
            event
        )

        # ====================================================
        # Type2 event classification
        # ====================================================

        is_type2 = (
            peak_dy
            >
            cfg.cst_type2_peak_min
            and
            peak_dy
            <=
            cfg.cst_type2_peak_max
        )

        if is_type2:

            type2_events.append(
                event
            )

    # ========================================================
    # Diagnostic
    # ========================================================

    print()
    print(
        "Raw CST movement peaks"
    )

    print(
        "----------------------------------------"
    )

    type2_raw_ids = {
        event[
            "raw_event_id"
        ]
        for event
        in type2_events
    }

    for event in all_events:

        if (
            event[
                "raw_event_id"
            ]
            in type2_raw_ids
        ):

            label = (
                "TYPE2"
            )

        else:

            label = (
                "REJECT"
            )

        print(
            f"#{event['raw_event_id']:02d} "
            f"{format_video_time(event['cst_peak_time'])} "
            f"dy={event['cst_peak_dy']:.4f} "
            f"[{label}]"
        )

    print()
    print(
        f"Type2 CST-UP anchors: "
        f"{len(type2_events)}"
    )

    return type2_events


# ============================================================
# Blade threshold
# ============================================================

def calculate_local_motion_threshold(
    values,
    cfg: Config,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:

        return (
            cfg.blade_motion_min_threshold
        )

    threshold = float(
        np.quantile(
            values,
            cfg.blade_motion_quantile,
        )
    )

    return max(
        threshold,
        cfg.blade_motion_min_threshold,
    )


# ============================================================
# Blade motion groups
# ============================================================

def find_blade_motion_groups(
    signals: pd.DataFrame,
    start_time: float,
    end_time: float,
    cfg: Config,
):

    blade_col = (
        f"{cfg.blade_name}"
        "_motion_ratio"
    )

    window = (
        signals[
            (
                signals[
                    "time_sec"
                ]
                >=
                start_time
            )
            &
            (
                signals[
                    "time_sec"
                ]
                <=
                end_time
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(window) == 0:

        return (
            [],
            np.nan,
        )

    threshold = (
        calculate_local_motion_threshold(
            window[
                blade_col
            ].to_numpy(),
            cfg,
        )
    )

    mask = (
        window[
            blade_col
        ]
        .to_numpy(
            dtype=np.float64
        )
        >=
        threshold
    )

    groups = find_boolean_groups(
        window,
        mask,
        max_gap_sec=
            cfg.blade_group_gap_sec,
        min_duration_sec=
            cfg.blade_min_motion_sec,
    )

    result = []

    for group in groups:

        start_idx = (
            group[
                "start_idx"
            ]
        )

        end_idx = (
            group[
                "end_idx"
            ]
        )

        local = window.iloc[
            start_idx:
            end_idx + 1
        ]

        peak_local_idx = (
            local[
                blade_col
            ]
            .idxmax()
        )

        peak_row = (
            window.loc[
                peak_local_idx
            ]
        )

        result.append({

            "start_time":
                float(
                    group[
                        "start_time"
                    ]
                ),

            "end_time":
                float(
                    group[
                        "end_time"
                    ]
                ),

            "duration":
                float(
                    group[
                        "duration"
                    ]
                ),

            "peak_time":
                float(
                    peak_row[
                        "time_sec"
                    ]
                ),

            "peak_motion":
                float(
                    peak_row[
                        blade_col
                    ]
                ),

            "threshold":
                threshold,
        })

    return (
        result,
        threshold,
    )


# ============================================================
# Blade IN
# ============================================================

def find_blade_in(
    signals: pd.DataFrame,
    cst_start_time: float,
    cfg: Config,
):

    search_start = max(
        0.0,
        cst_start_time
        -
        cfg.blade_in_search_sec,
    )

    search_end = (
        cst_start_time
    )

    (
        groups,
        threshold,
    ) = find_blade_motion_groups(
        signals,
        search_start,
        search_end,
        cfg,
    )

    if not groups:

        return (
            None,
            threshold,
        )

    # CST movement 직전의 가장 가까운 Blade motion
    candidate = max(
        groups,
        key=lambda g:
            g[
                "end_time"
            ],
    )

    return (
        candidate,
        threshold,
    )


# ============================================================
# Blade OUT
# ============================================================

def find_blade_out(
    signals: pd.DataFrame,
    cst_end_time: float,
    cfg: Config,
):

    search_start = (
        cst_end_time
    )

    search_end = (
        cst_end_time
        +
        cfg.blade_out_search_sec
    )

    (
        groups,
        threshold,
    ) = find_blade_motion_groups(
        signals,
        search_start,
        search_end,
        cfg,
    )

    if not groups:

        return (
            None,
            threshold,
        )

    # CST movement 이후 가장 먼저 나타난 Blade motion
    candidate = min(
        groups,
        key=lambda g:
            g[
                "start_time"
            ],
    )

    return (
        candidate,
        threshold,
    )


# ============================================================
# Build Type2 events
# ============================================================

def build_type2_events(
    signals: pd.DataFrame,
    cst_events,
    cfg: Config,
):

    detected = []

    for (
        event_id,
        cst,
    ) in enumerate(
        cst_events,
        start=1,
    ):

        cst_start = float(
            cst[
                "cst_start_time"
            ]
        )

        cst_end = float(
            cst[
                "cst_end_time"
            ]
        )

        # ====================================================
        # Blade IN
        # ====================================================

        (
            blade_in,
            blade_in_threshold,
        ) = find_blade_in(
            signals,
            cst_start,
            cfg,
        )

        # ====================================================
        # Blade OUT
        # ====================================================

        (
            blade_out,
            blade_out_threshold,
        ) = find_blade_out(
            signals,
            cst_end,
            cfg,
        )

        # ====================================================
        # Clip start
        # ====================================================

        if blade_in is not None:

            clip_start = (
                blade_in[
                    "start_time"
                ]
                -
                cfg.pre_margin_sec
            )

            blade_in_source = (
                "blade"
            )

        else:

            clip_start = (
                cst_start
                -
                cfg.fallback_pre_sec
            )

            blade_in_source = (
                "fallback"
            )

        clip_start = max(
            0.0,
            clip_start,
        )

        # ====================================================
        # Clip end
        # ====================================================

        if blade_out is not None:

            clip_end = (
                blade_out[
                    "end_time"
                ]
                +
                cfg.post_margin_sec
            )

            blade_out_source = (
                "blade"
            )

        else:

            clip_end = (
                cst_end
                +
                cfg.fallback_post_sec
            )

            blade_out_source = (
                "fallback"
            )

        # 잘못된 boundary 보호
        if clip_end <= clip_start:

            clip_start = max(
                0.0,
                cst_start
                -
                cfg.fallback_pre_sec,
            )

            clip_end = (
                cst_end
                +
                cfg.fallback_post_sec
            )

            blade_in_source = (
                "fallback"
            )

            blade_out_source = (
                "fallback"
            )

        # ====================================================
        # Sequence score
        # ====================================================

        score = 1.0

        if blade_in is None:
            score -= 0.25

        if blade_out is None:
            score -= 0.25

        score = max(
            0.0,
            score,
        )

        # ====================================================
        # Result
        # ====================================================

        detected.append({

            "event_id":
                event_id,

            "raw_cst_event_id":
                int(
                    cst[
                        "raw_event_id"
                    ]
                ),

            # ------------------------------------------------
            # CST
            # ------------------------------------------------

            "cst_start_sec":
                cst_start,

            "cst_peak_sec":
                float(
                    cst[
                        "cst_peak_time"
                    ]
                ),

            "cst_end_sec":
                cst_end,

            "cst_peak_dy":
                float(
                    cst[
                        "cst_peak_dy"
                    ]
                ),

            "cst_peak_frame":
                int(
                    cst[
                        "cst_peak_frame"
                    ]
                ),

            "cst_duration_sec":
                float(
                    cst[
                        "cst_duration_sec"
                    ]
                ),

            # ------------------------------------------------
            # Blade IN
            # ------------------------------------------------

            "blade_in_start_sec":
                (
                    blade_in[
                        "start_time"
                    ]
                    if blade_in
                    else np.nan
                ),

            "blade_in_end_sec":
                (
                    blade_in[
                        "end_time"
                    ]
                    if blade_in
                    else np.nan
                ),

            "blade_in_peak_sec":
                (
                    blade_in[
                        "peak_time"
                    ]
                    if blade_in
                    else np.nan
                ),

            "blade_in_peak_motion":
                (
                    blade_in[
                        "peak_motion"
                    ]
                    if blade_in
                    else np.nan
                ),

            "blade_in_threshold":
                blade_in_threshold,

            # ------------------------------------------------
            # Blade OUT
            # ------------------------------------------------

            "blade_out_start_sec":
                (
                    blade_out[
                        "start_time"
                    ]
                    if blade_out
                    else np.nan
                ),

            "blade_out_end_sec":
                (
                    blade_out[
                        "end_time"
                    ]
                    if blade_out
                    else np.nan
                ),

            "blade_out_peak_sec":
                (
                    blade_out[
                        "peak_time"
                    ]
                    if blade_out
                    else np.nan
                ),

            "blade_out_peak_motion":
                (
                    blade_out[
                        "peak_motion"
                    ]
                    if blade_out
                    else np.nan
                ),

            "blade_out_threshold":
                blade_out_threshold,

            # ------------------------------------------------
            # Clip
            # ------------------------------------------------

            "clip_start_sec":
                clip_start,

            "clip_end_sec":
                clip_end,

            "clip_duration_sec":
                (
                    clip_end
                    -
                    clip_start
                ),

            "blade_in_source":
                blade_in_source,

            "blade_out_source":
                blade_out_source,

            "sequence_score":
                score,
        })

    return detected


# ============================================================
# Ground Truth
# ============================================================

def load_gt(
    path,
):

    gt = pd.read_excel(
        path,
        dtype={
            "type": str,
            "start_sec": str,
            "end_sec": str,
        },
    )

    gt[
        "type"
    ] = (
        gt[
            "type"
        ]
        .str.strip()
        .str.lower()
    )

    gt[
        "gt_start_sec"
    ] = (
        gt[
            "start_sec"
        ]
        .apply(
            parse_video_time
        )
    )

    gt[
        "gt_end_sec"
    ] = (
        gt[
            "end_sec"
        ]
        .apply(
            parse_video_time
        )
    )

    return gt


# ============================================================
# GT comparison
# ============================================================

def compare_with_gt(
    detected_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    cfg: Config,
):

    gt2 = (
        gt_df[
            gt_df[
                "type"
            ]
            ==
            "type_2"
        ]
        .copy()
        .sort_values(
            "gt_start_sec"
        )
        .reset_index(
            drop=True
        )
    )

    rows = []

    used_detected = set()

    for _, gt in (
        gt2.iterrows()
    ):

        gt_start = float(
            gt[
                "gt_start_sec"
            ]
        )

        gt_end = float(
            gt[
                "gt_end_sec"
            ]
        )

        gt_center = (
            gt_start
            +
            gt_end
        ) / 2.0

        best_idx = None
        best_distance = None

        for (
            det_idx,
            det,
        ) in detected_df.iterrows():

            if (
                det_idx
                in used_detected
            ):
                continue

            anchor = float(
                det[
                    "cst_peak_sec"
                ]
            )

            if (
                anchor
                <
                gt_start
                -
                cfg.gt_match_tolerance_sec
            ):
                continue

            if (
                anchor
                >
                gt_end
                +
                cfg.gt_match_tolerance_sec
            ):
                continue

            distance = abs(
                anchor
                -
                gt_center
            )

            if (
                best_distance is None
                or
                distance
                <
                best_distance
            ):

                best_idx = (
                    det_idx
                )

                best_distance = (
                    distance
                )

        # ====================================================
        # Miss
        # ====================================================

        if best_idx is None:

            rows.append({

                "slot":
                    gt["slot"],

                "gt_start_sec":
                    gt_start,

                "gt_end_sec":
                    gt_end,

                "matched":
                    False,

                "detected_event_id":
                    np.nan,

                "cst_peak_sec":
                    np.nan,

                "time_error_sec":
                    np.nan,

                "clip_start_sec":
                    np.nan,

                "clip_end_sec":
                    np.nan,
            })

            continue

        # ====================================================
        # Match
        # ====================================================

        used_detected.add(
            best_idx
        )

        det = (
            detected_df.loc[
                best_idx
            ]
        )

        rows.append({

            "slot":
                gt["slot"],

            "gt_start_sec":
                gt_start,

            "gt_end_sec":
                gt_end,

            "matched":
                True,

            "detected_event_id":
                int(
                    det[
                        "event_id"
                    ]
                ),

            "cst_peak_sec":
                float(
                    det[
                        "cst_peak_sec"
                    ]
                ),

            "time_error_sec":
                (
                    float(
                        det[
                            "cst_peak_sec"
                        ]
                    )
                    -
                    gt_center
                ),

            "clip_start_sec":
                float(
                    det[
                        "clip_start_sec"
                    ]
                ),

            "clip_end_sec":
                float(
                    det[
                        "clip_end_sec"
                    ]
                ),
        })

    comparison = pd.DataFrame(
        rows
    )

    return (
        comparison,
        used_detected,
    )


# ============================================================
# Clip export
# ============================================================

def export_clips(
    video_path: Path,
    detected_df: pd.DataFrame,
    clip_dir: Path,
):

    clip_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration_sec = (
        total_frames / fps
        if fps > 0
        else 0.0
    )

    print()
    print(
        "Video"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"FPS        : "
        f"{fps:.3f}"
    )

    print(
        f"Resolution : "
        f"{width}x{height}"
    )

    print(
        f"Frames     : "
        f"{total_frames:,}"
    )

    print(
        f"Duration   : "
        f"{format_video_time(duration_sec)}"
    )

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    for _, event in (
        detected_df.iterrows()
    ):

        event_id = int(
            event[
                "event_id"
            ]
        )

        start_sec = max(
            0.0,
            float(
                event[
                    "clip_start_sec"
                ]
            ),
        )

        end_sec = min(
            duration_sec,
            float(
                event[
                    "clip_end_sec"
                ]
            ),
        )

        start_frame = max(
            0,
            int(
                np.floor(
                    start_sec
                    *
                    fps
                )
            ),
        )

        end_frame = min(
            total_frames - 1,
            int(
                np.ceil(
                    end_sec
                    *
                    fps
                )
            ),
        )

        output_path = (
            clip_dir
            /
            f"type2_{event_id:03d}.mp4"
        )

        print(
            f"[CLIP {event_id:03d}] "
            f"{format_video_time(start_sec)} "
            f"~ "
            f"{format_video_time(end_sec)} "
            f"| "
            f"{end_sec - start_sec:.2f}s "
            f"| frame "
            f"{start_frame}~{end_frame}"
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame,
        )

        writer = (
            cv2.VideoWriter(
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
        )

        if not writer.isOpened():

            cap.release()

            raise RuntimeError(
                f"Cannot create video: "
                f"{output_path}"
            )

        frame_idx = (
            start_frame
        )

        while (
            frame_idx
            <=
            end_frame
        ):

            ok, frame = (
                cap.read()
            )

            if (
                not ok
                or
                frame is None
            ):
                break

            writer.write(
                frame
            )

            frame_idx += 1

        writer.release()

    cap.release()


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Phase E-3 v3: "
                "event-level Type2 CST detection "
                "and clip extraction"
            )
        )
    )

    # ========================================================
    # Input
    # ========================================================

    parser.add_argument(
        "--signals",
        required=True,
        help=(
            "Phase E-1 "
            "motion_signals.csv"
        ),
    )

    parser.add_argument(
        "--timestamps",
        required=True,
        help=(
            "Ground-truth Excel"
        ),
    )

    parser.add_argument(
        "--video",
        required=True,
        help=(
            "Original long MP4"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "phase_e3_result"
        ),
    )

    # ========================================================
    # CST
    # ========================================================

    parser.add_argument(
        "--cst-motion-threshold",
        type=float,
        default=-2.0,
        help=(
            "Raw CST negative movement "
            "threshold"
        ),
    )

    parser.add_argument(
        "--cst-type2-peak-min",
        type=float,
        default=-8.0,
        help=(
            "Type2 event minimum dy "
            "lower bound (exclusive)"
        ),
    )

    parser.add_argument(
        "--cst-type2-peak-max",
        type=float,
        default=-2.0,
        help=(
            "Type2 event minimum dy "
            "upper bound (inclusive)"
        ),
    )

    # ========================================================
    # Clip
    # ========================================================

    parser.add_argument(
        "--pre-margin",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--post-margin",
        type=float,
        default=0.5,
    )

    # ========================================================
    # Validation
    # ========================================================

    parser.add_argument(
        "--no-export",
        action="store_true",
        help=(
            "Detection/GT evaluation만 수행하고 "
            "MP4는 저장하지 않는다."
        ),
    )

    args = (
        parser.parse_args()
    )

    # ========================================================
    # Config
    # ========================================================

    cfg = Config(

        cst_motion_threshold=
            args.cst_motion_threshold,

        cst_type2_peak_min=
            args.cst_type2_peak_min,

        cst_type2_peak_max=
            args.cst_type2_peak_max,

        pre_margin_sec=
            args.pre_margin,

        post_margin_sec=
            args.post_margin,
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

    signals = pd.read_csv(
        args.signals
    )

    required_columns = [

        "frame_idx",

        "time_sec",

        "cst_dy",

        (
            f"{cfg.blade_name}"
            "_motion_ratio"
        ),
    ]

    missing_columns = [

        column

        for column
        in required_columns

        if column
        not in signals.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing signal columns: "
            f"{missing_columns}"
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

    print(
        f"Signal rows: "
        f"{len(signals):,}"
    )

    # ========================================================
    # Config diagnostic
    # ========================================================

    print()

    print(
        "CST detection"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Motion threshold : "
        f"cst_dy <= "
        f"{cfg.cst_motion_threshold:.3f}"
    )

    print(
        f"Type2 event peak : "
        f"{cfg.cst_type2_peak_min:.3f} "
        f"< event_min_dy <= "
        f"{cfg.cst_type2_peak_max:.3f}"
    )

    # ========================================================
    # CST event detection
    # ========================================================

    cst_events = (
        detect_cst_up_events(
            signals,
            cfg,
        )
    )

    # ========================================================
    # Build Type2 event
    # ========================================================

    detected = (
        build_type2_events(
            signals,
            cst_events,
            cfg,
        )
    )

    detected_df = (
        pd.DataFrame(
            detected
        )
    )

    # ========================================================
    # Human-readable timestamps
    # ========================================================

    if len(
        detected_df
    ):

        time_columns = [

            "cst_start_sec",

            "cst_peak_sec",

            "cst_end_sec",

            "clip_start_sec",

            "clip_end_sec",
        ]

        for column in (
            time_columns
        ):

            detected_df[
                column.replace(
                    "_sec",
                    "_time",
                )
            ] = (
                detected_df[
                    column
                ]
                .apply(
                    format_video_time
                )
            )

    # ========================================================
    # Save detected events
    # ========================================================

    detected_path = (
        output_dir
        /
        "detected_events.csv"
    )

    detected_df.to_csv(
        detected_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # GT
    # ========================================================

    gt = load_gt(
        args.timestamps
    )

    gt_type2_count = int(
        (
            gt[
                "type"
            ]
            ==
            "type_2"
        )
        .sum()
    )

    # ========================================================
    # Compare GT
    # ========================================================

    (
        comparison,
        matched_detected_indices,
    ) = compare_with_gt(
        detected_df,
        gt,
        cfg,
    )

    comparison_path = (
        output_dir
        /
        "gt_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    matched_count = int(
        comparison[
            "matched"
        ]
        .sum()
    )

    detected_count = len(
        detected_df
    )

    extra_count = (
        detected_count
        -
        len(
            matched_detected_indices
        )
    )

    recall = (
        matched_count
        /
        gt_type2_count

        if gt_type2_count

        else 0.0
    )

    precision = (
        len(
            matched_detected_indices
        )
        /
        detected_count

        if detected_count

        else 0.0
    )

    # ========================================================
    # Boundary statistics
    # ========================================================

    if detected_count:

        blade_in_found = int(
            detected_df[
                "blade_in_start_sec"
            ]
            .notna()
            .sum()
        )

        blade_out_found = int(
            detected_df[
                "blade_out_start_sec"
            ]
            .notna()
            .sum()
        )

        both_found = int(
            (
                detected_df[
                    "blade_in_start_sec"
                ]
                .notna()
                &
                detected_df[
                    "blade_out_start_sec"
                ]
                .notna()
            )
            .sum()
        )

        fallback_in = int(
            (
                detected_df[
                    "blade_in_source"
                ]
                ==
                "fallback"
            )
            .sum()
        )

        fallback_out = int(
            (
                detected_df[
                    "blade_out_source"
                ]
                ==
                "fallback"
            )
            .sum()
        )

    else:

        blade_in_found = 0

        blade_out_found = 0

        both_found = 0

        fallback_in = 0

        fallback_out = 0

    # ========================================================
    # Result
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "Phase E-3 v3 Detection Result"
    )

    print(
        "========================================"
    )

    print(
        f"GT Type2            : "
        f"{gt_type2_count}"
    )

    print(
        f"Detected candidates : "
        f"{detected_count}"
    )

    print(
        f"Matched Type2       : "
        f"{matched_count}"
    )

    print(
        f"Missed Type2        : "
        f"{gt_type2_count - matched_count}"
    )

    print(
        f"Extra candidates    : "
        f"{extra_count}"
    )

    print(
        f"Recall              : "
        f"{recall:.4f}"
    )

    print(
        f"Precision           : "
        f"{precision:.4f}"
    )

    # ========================================================
    # Boundary
    # ========================================================

    print()

    print(
        "Sequence boundary"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Blade IN found      : "
        f"{blade_in_found}/"
        f"{detected_count}"
    )

    print(
        f"Blade OUT found     : "
        f"{blade_out_found}/"
        f"{detected_count}"
    )

    print(
        f"Both found          : "
        f"{both_found}/"
        f"{detected_count}"
    )

    print(
        f"Blade IN fallback   : "
        f"{fallback_in}"
    )

    print(
        f"Blade OUT fallback  : "
        f"{fallback_out}"
    )

    # ========================================================
    # Detected Type2
    # ========================================================

    if detected_count:

        print()

        print(
            "Detected Type2 CST-UP anchors"
        )

        print(
            "----------------------------------------"
        )

        for _, row in (
            detected_df.iterrows()
        ):

            print(
                f"#{int(row['event_id']):02d} "
                f"raw=#{int(row['raw_cst_event_id']):02d} "
                f"{format_video_time(row['cst_peak_sec'])} "
                f"dy={row['cst_peak_dy']:.4f} "
                f"| IN={row['blade_in_source']} "
                f"| OUT={row['blade_out_source']} "
                f"| clip={row['clip_duration_sec']:.2f}s"
            )

    # ========================================================
    # Missed GT
    # ========================================================

    missed = (
        comparison[
            ~comparison[
                "matched"
            ]
        ]
    )

    if len(
        missed
    ):

        print()

        print(
            "MISSED Type2"
        )

        print(
            "----------------------------------------"
        )

        print(
            missed[
                [
                    "slot",
                    "gt_start_sec",
                    "gt_end_sec",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ========================================================
    # Extra
    # ========================================================

    if detected_count:

        extra_indices = (
            set(
                detected_df.index
            )
            -
            set(
                matched_detected_indices
            )
        )

        if extra_indices:

            print()

            print(
                "EXTRA candidates"
            )

            print(
                "----------------------------------------"
            )

            extra_df = (
                detected_df.loc[
                    sorted(
                        extra_indices
                    )
                ]
            )

            print(
                extra_df[
                    [
                        "event_id",
                        "raw_cst_event_id",
                        "cst_peak_sec",
                        "cst_peak_dy",
                    ]
                ]
                .to_string(
                    index=False
                )
            )

    # ========================================================
    # Output
    # ========================================================

    print()

    print(
        "Output"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Detected CSV : "
        f"{detected_path}"
    )

    print(
        f"GT comparison: "
        f"{comparison_path}"
    )

    # ========================================================
    # Export
    # ========================================================

    if (
        not args.no_export
        and
        detected_count
    ):

        print()

        print(
            "Exporting Type2 clips..."
        )

        export_clips(
            Path(
                args.video
            ),
            detected_df,
            output_dir
            /
            "clips",
        )

        print()

        print(
            "Clip export complete."
        )

    elif args.no_export:

        print()

        print(
            "[INFO] --no-export: "
            "clip export skipped."
        )


if __name__ == "__main__":
    main()
