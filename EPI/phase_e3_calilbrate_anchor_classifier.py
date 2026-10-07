"""
phase_e3_extract_type2_clips.py

Phase E-3 v2
============

목적
----
Phase E-1에서 생성한 motion_signals.csv를 이용하여
원본 장시간 CCTV 영상에서 Type2 이벤트를 자동 검출하고
Phase A에서 사용할 표준 Type2 clip을 추출한다.

Type2 실제 sequence
-------------------
1. 작업 완료 wafer를 Blade가 CST로 가지고 옴
2. Blade가 끝까지 들어옴
3. CST가 wafer 홈 안착을 위해 살짝 상승
4. wafer가 CST 홈에 안착
5. CST 상승 종료
6. 잠시 정지
7. Blade OUT

Phase E-3에서는 정확한 STOP/측정 시점까지 찾지 않는다.

검출 구조
---------
Blade IN
   ↓
Type2 CST-UP anchor
   ↓
Blade OUT

CST STOP 및 정확한 measurement frame 검출은
다음 Phase A에서 수행한다.

중요한 실험 결과
----------------
전체 CST-UP candidate 49개를 분석한 결과:

Type2 target CST-UP:
    약 -4.875 ~ -5.186

다른 큰 CST movement:
    약 -10.86 이하
    일부 -14, -16, -31 수준

따라서 Type2 CST-UP을 다음 band로 검출:

    -8.0 < cst_dy <= -2.0

Blade ROI:
    blade2 = (309, 379, 1593, 236)

Blade signal은 Type2 분류용이 아니라
clip start/end boundary 탐색용으로만 사용한다.

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
    # Type2 CST-UP band
    # ========================================================

    # Type2:
    #     약 -4.875 ~ -5.186
    #
    # 다른 큰 CST movement:
    #     약 -10.86 이하
    #
    # 따라서:
    #
    #     -8 < cst_dy <= -2
    #
    # 영역만 Type2 CST-UP candidate로 사용

    cst_up_min_dy: float = -8.0
    cst_up_max_dy: float = -2.0

    # 가까운 CST-UP frame들을 하나의 event로 묶음
    cst_group_gap_sec: float = 0.40

    # 너무 짧은 단발 noise 제거
    cst_min_event_sec: float = 0.03

    # ========================================================
    # Blade boundary
    # ========================================================

    # E-2 결과 기준 가장 안정적인 Blade ROI
    blade_name: str = "blade2"

    # CST-UP 시작 전 Blade IN 탐색 범위
    blade_in_search_sec: float = 3.0

    # CST-UP 종료 후 Blade OUT 탐색 범위
    blade_out_search_sec: float = 3.0

    # local adaptive threshold
    blade_motion_quantile: float = 0.75

    # threshold가 지나치게 낮아지는 것 방지
    blade_motion_min_threshold: float = 0.015

    # Blade motion 최소 지속 시간
    blade_min_motion_sec: float = 0.06

    # motion 사이 짧은 gap은 하나로 연결
    blade_group_gap_sec: float = 0.20

    # ========================================================
    # Clip margin
    # ========================================================

    # Blade IN 시작보다 조금 앞
    pre_margin_sec: float = 0.50

    # Blade OUT 종료보다 조금 뒤
    post_margin_sec: float = 0.50

    # Blade IN/OUT 검출 실패 시
    # CST anchor 기준 fallback
    fallback_pre_sec: float = 3.0
    fallback_post_sec: float = 3.0

    # ========================================================
    # GT evaluation
    # ========================================================

    gt_match_tolerance_sec: float = 3.0


# ============================================================
# Time utilities
# ============================================================

def parse_video_time(value) -> float:
    """
    Excel timestamp를 초 단위로 변환.

    지원:
        03:45
        43:39
        01:03:25
        '01:03:25
    """

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


def format_video_time(sec: float) -> str:

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
# Boolean temporal grouping
# ============================================================

def find_boolean_groups(
    df: pd.DataFrame,
    mask: np.ndarray,
    max_gap_sec: float,
    min_duration_sec: float,
):
    """
    True mask를 시간 기준으로 group화한다.

    중간 False 구간이 max_gap_sec 이하이면
    동일 event로 연결한다.
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

        if (
            gap
            <= max_gap_sec
        ):
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

        # 15 FPS에서는 짧은 실제 motion이
        # 1~2 frame일 수 있으므로 frame 수 조건도 허용
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
# Type2 CST-UP Detection
# ============================================================

def detect_cst_up_events(
    signals: pd.DataFrame,
    cfg: Config,
):
    """
    Type2 특유의 작은 CST-UP만 검출한다.

    현재 분석 결과:

        Type2:
            약 -4.875 ~ -5.186

        다른 CST movement:
            약 -10.86 이하

    따라서:

        cst_up_min_dy < cst_dy <= cst_up_max_dy

    즉 기본값:

        -8 < cst_dy <= -2

    를 만족하는 frame만 사용한다.
    """

    dy = (
        signals[
            "cst_dy"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    mask = (
        (dy > cfg.cst_up_min_dy)
        &
        (dy <= cfg.cst_up_max_dy)
    )

    groups = find_boolean_groups(
        signals,
        mask,
        max_gap_sec=
            cfg.cst_group_gap_sec,
        min_duration_sec=
            cfg.cst_min_event_sec,
    )

    events = []

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

        local = signals.iloc[
            start_idx:
            end_idx + 1
        ]

        # group 안에서 가장 강한 UP
        # = 가장 작은 cst_dy
        min_local_idx = (
            local[
                "cst_dy"
            ]
            .idxmin()
        )

        peak_row = (
            signals.loc[
                min_local_idx
            ]
        )

        events.append({

            "cst_start_time":
                group[
                    "start_time"
                ],

            "cst_end_time":
                group[
                    "end_time"
                ],

            "cst_peak_time":
                float(
                    peak_row[
                        "time_sec"
                    ]
                ),

            "cst_peak_dy":
                float(
                    peak_row[
                        "cst_dy"
                    ]
                ),

            "cst_peak_frame":
                int(
                    peak_row[
                        "frame_idx"
                    ]
                ),
        })

    return events


# ============================================================
# Blade motion threshold
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
        f"{cfg.blade_name}_"
        "motion_ratio"
    )

    window = (
        signals[
            (
                signals[
                    "time_sec"
                ]
                >= start_time
            )
            &
            (
                signals[
                    "time_sec"
                ]
                <= end_time
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(window) == 0:
        return [], np.nan

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
        ].to_numpy(
            dtype=np.float64
        )
        >= threshold
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
                group[
                    "start_time"
                ],

            "end_time":
                group[
                    "end_time"
                ],

            "duration":
                group[
                    "duration"
                ],

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
    """
    CST-UP 시작 전 blade motion을 탐색한다.

    radial sign은 사용하지 않는다.

    CST-UP에 가장 가까운 motion group을
    Blade IN 후보로 사용한다.
    """

    search_start = max(
        0.0,
        cst_start_time
        - cfg.blade_in_search_sec,
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
        return None, threshold

    # CST-UP 직전 가장 가까운 motion
    candidate = max(
        groups,
        key=lambda g:
            g["end_time"]
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
    """
    CST-UP 종료 후 blade motion을 탐색한다.

    CST-UP 이후 최초 motion group을
    Blade OUT 후보로 사용한다.
    """

    search_start = (
        cst_end_time
    )

    search_end = (
        cst_end_time
        + cfg.blade_out_search_sec
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
        return None, threshold

    candidate = min(
        groups,
        key=lambda g:
            g["start_time"]
    )

    return (
        candidate,
        threshold,
    )


# ============================================================
# Build Type2 Events
# ============================================================

def build_type2_events(
    signals: pd.DataFrame,
    cst_events,
    cfg: Config,
):
    """
    Type2 CST-UP anchor마다 Blade IN/OUT boundary를 찾는다.

    Blade boundary 검출 실패 시에도
    event 자체를 버리지 않는다.

    이유:
        Type2 검출의 핵심은 CST-UP anchor이고,
        Blade는 clip boundary 보조 신호이기 때문이다.
    """

    detected = []

    for event_id, cst in enumerate(
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

        # 혹시 잘못된 boundary가 만들어지는 것 방지
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
                clip_end
                - clip_start,

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

    gt["type"] = (
        gt["type"]
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
# GT Comparison
# ============================================================

def compare_with_gt(
    detected_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    cfg: Config,
):
    """
    검출된 Type2 candidate와
    GT Type2만 비교한다.

    GT는 detection에 사용하지 않고
    성능 평가에만 사용한다.
    """

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

    for _, gt in gt2.iterrows():

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

        for det_idx, det in (
            detected_df.iterrows()
        ):

            if det_idx in used_detected:
                continue

            anchor = float(
                det[
                    "cst_peak_sec"
                ]
            )

            # GT interval + tolerance
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

            # anchor가 GT 내부면
            # GT center와의 거리로 가장 가까운 것 선택
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
# Video Clip Export
# ============================================================

def export_clips(
    video_path: Path,
    detected_df: pd.DataFrame,
    clip_dir: Path,
):
    """
    OpenCV로 원본 영상의 정확한 frame 구간을 decode하여
    Type2 clip을 저장한다.

    ffmpeg stream-copy 방식과 달리 keyframe 위치에
    clip 시작점이 밀리는 문제를 피하기 위해
    frame 단위 decode/write를 사용한다.
    """

    clip_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
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
        f"FPS        : {fps:.3f}"
    )
    print(
        f"Resolution : {width}x{height}"
    )
    print(
        f"Frames     : {total_frames:,}"
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
                    * fps
                )
            ),
        )

        end_frame = min(
            total_frames - 1,
            int(
                np.ceil(
                    end_sec
                    * fps
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
            f"{end_sec-start_sec:.2f}s "
            f"| "
            f"frame "
            f"{start_frame}~{end_frame}"
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame,
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (
                width,
                height,
            ),
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
            <= end_frame
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

    parser = argparse.ArgumentParser(
        description=(
            "Phase E-3 v2: "
            "Type2 CST-UP detection and clip extraction"
        )
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    parser.add_argument(
        "--signals",
        required=True,
        help="Phase E-1 motion_signals.csv",
    )

    parser.add_argument(
        "--timestamps",
        required=True,
        help="Ground-truth Excel",
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Original long MP4",
    )

    parser.add_argument(
        "--output",
        default="phase_e3_result",
    )

    # --------------------------------------------------------
    # CST band
    # --------------------------------------------------------

    parser.add_argument(
        "--cst-up-min-dy",
        type=float,
        default=-8.0,
        help=(
            "Type2 CST-UP lower bound "
            "(exclusive)"
        ),
    )

    parser.add_argument(
        "--cst-up-max-dy",
        type=float,
        default=-2.0,
        help=(
            "Type2 CST-UP upper bound "
            "(inclusive)"
        ),
    )

    # --------------------------------------------------------
    # Clip margin
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Debug / validation
    # --------------------------------------------------------

    parser.add_argument(
        "--no-export",
        action="store_true",
        help=(
            "Detection/GT evaluation만 수행하고 "
            "MP4 clip은 저장하지 않는다."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Config
    # ========================================================

    cfg = Config(

        cst_up_min_dy=
            args.cst_up_min_dy,

        cst_up_max_dy=
            args.cst_up_max_dy,

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
    # Load Signals
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
        f"{cfg.blade_name}_motion_ratio",
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

    print()
    print(
        "Type2 CST-UP band"
    )
    print(
        "----------------------------------------"
    )

    print(
        f"{cfg.cst_up_min_dy:.3f} "
        f"< cst_dy <= "
        f"{cfg.cst_up_max_dy:.3f}"
    )

    # ========================================================
    # Detect Type2 CST-UP
    # ========================================================

    cst_events = (
        detect_cst_up_events(
            signals,
            cfg,
        )
    )

    print()
    print(
        f"CST UP anchors: "
        f"{len(cst_events)}"
    )

    # ========================================================
    # Build events
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
    # Human-readable time
    # ========================================================

    if len(detected_df):

        time_columns = [
            "cst_start_sec",
            "cst_peak_sec",
            "cst_end_sec",
            "clip_start_sec",
            "clip_end_sec",
        ]

        for column in time_columns:

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
    # Save detection CSV
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
    # Load GT
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
        ).sum()
    )

    # ========================================================
    # GT evaluation
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
        ].sum()
    )

    detected_count = (
        len(detected_df)
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
    # Sequence statistics
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

        full_boundary_found = int(
            (
                detected_df[
                    "blade_in_start_sec"
                ].notna()
                &
                detected_df[
                    "blade_out_start_sec"
                ].notna()
            ).sum()
        )

        fallback_in_count = int(
            (
                detected_df[
                    "blade_in_source"
                ]
                ==
                "fallback"
            ).sum()
        )

        fallback_out_count = int(
            (
                detected_df[
                    "blade_out_source"
                ]
                ==
                "fallback"
            ).sum()
        )

    else:

        blade_in_found = 0
        blade_out_found = 0
        full_boundary_found = 0
        fallback_in_count = 0
        fallback_out_count = 0

    # ========================================================
    # Console result
    # ========================================================

    print()
    print(
        "========================================"
    )
    print(
        "Phase E-3 v2 Detection Result"
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
        f"{full_boundary_found}/"
        f"{detected_count}"
    )

    print(
        f"Blade IN fallback   : "
        f"{fallback_in_count}"
    )

    print(
        f"Blade OUT fallback  : "
        f"{fallback_out_count}"
    )

    # ========================================================
    # Candidate details
    # ========================================================

    if detected_count:

        print()
        print(
            "Detected CST-UP anchors"
        )
        print(
            "----------------------------------------"
        )

        for _, row in (
            detected_df.iterrows()
        ):

            print(
                f"#{int(row['event_id']):02d} "
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

    if len(missed):

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
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # Extra candidates
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
                        "cst_peak_sec",
                        "cst_peak_dy",
                    ]
                ].to_string(
                    index=False
                )
            )

    # ========================================================
    # Output paths
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
    # Export clips
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
