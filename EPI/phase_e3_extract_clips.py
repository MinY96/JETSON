"""
phase_e3_extract_type2_clips.py

Phase E-3
=========
motion_signals.csv에서 CST UP을 anchor로 Type2 후보를 검출하고,
blade2 motion signal을 이용하여 Blade IN / OUT 경계를 추정한 뒤
원본 MP4에서 표준 Type2 clip을 추출한다.

Detection philosophy
--------------------
1. Type2 판정의 핵심 anchor:
       CST UP

2. Blade signal의 역할:
       Type2 판정 X
       clip start/end boundary 탐색 O

3. Sequence:
       Blade IN
          ↓
       CST UP
          ↓
       CST STOP
          ↓
       Blade OUT

4. Recall 우선:
       GT Type2 25/25 검출을 우선한다.

Input
-----
- motion_signals.csv
- blade_in_out_timestamp.xlsx
- original MP4

Output
------
phase_e3_result/
    clips/
        type2_001.mp4
        ...
    detected_events.csv
    gt_comparison.csv
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

    # --------------------------------------------------------
    # CST UP
    #
    # E-2:
    #
    # Type1:
    #     cst_dy_min
    #     +0.3766 ~ +0.4788
    #
    # Type2:
    #     cst_dy_min
    #     -5.1863 ~ -4.8754
    #
    # 따라서 -2.0은 상당히 보수적인 high-recall threshold.
    # --------------------------------------------------------

    cst_up_threshold: float = -2.0

    # CST UP candidate들을 하나의 event로 묶을 시간
    cst_group_gap_sec: float = 0.40

    # 너무 짧은 단발성 noise 제거
    cst_min_event_sec: float = 0.03

    # --------------------------------------------------------
    # Blade boundary search
    # --------------------------------------------------------

    blade_name: str = "blade2"

    # CST anchor 기준 탐색 범위
    blade_in_search_sec: float = 3.0
    blade_out_search_sec: float = 3.0

    # motion_ratio threshold는 고정값 대신
    # local adaptive threshold 사용
    blade_motion_quantile: float = 0.75

    # 너무 낮은 threshold 방지
    blade_motion_min_threshold: float = 0.015

    # 연속 motion group 최소 길이
    blade_min_motion_sec: float = 0.06

    # motion group 사이 gap merge
    blade_group_gap_sec: float = 0.20

    # --------------------------------------------------------
    # CST STOP
    # --------------------------------------------------------

    cst_stop_abs_threshold: float = 0.30

    cst_stop_min_sec: float = 0.05

    # --------------------------------------------------------
    # Final clip margin
    # --------------------------------------------------------

    pre_margin_sec: float = 0.50
    post_margin_sec: float = 0.50

    # Blade boundary 검출 실패 시 fallback
    fallback_pre_sec: float = 2.5
    fallback_post_sec: float = 2.5

    # --------------------------------------------------------
    # GT matching
    # --------------------------------------------------------

    gt_match_tolerance_sec: float = 3.0


# ============================================================
# Time utility
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
# Group boolean mask
# ============================================================

def find_boolean_groups(
    df: pd.DataFrame,
    mask: np.ndarray,
    max_gap_sec: float,
    min_duration_sec: float,
):

    """
    True mask를 temporal group으로 묶는다.

    예:
        True True False True True

    False 구간이 max_gap_sec 이하이면
    하나의 group으로 merge.
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

        prev_idx = current[-1]

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

        if (
            curr_time
            - prev_time
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

        start_idx = group[0]
        end_idx = group[-1]

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

        # 15 FPS에서는 single frame event도
        # duration=0이므로 최소 frame 수 관점도 같이 허용
        if (
            duration >= min_duration_sec
            or len(group) >= 2
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
# CST UP Detection
# ============================================================

def detect_cst_up_events(
    signals: pd.DataFrame,
    cfg: Config,
):

    dy = signals[
        "cst_dy"
    ].to_numpy(
        dtype=np.float64
    )

    mask = (
        dy
        <= cfg.cst_up_threshold
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
            group["start_idx"]
        )

        end_idx = (
            group["end_idx"]
        )

        # group 범위 내 실제 minimum dy
        local = signals.iloc[
            start_idx:
            end_idx + 1
        ]

        min_local_idx = (
            local[
                "cst_dy"
            ].idxmin()
        )

        peak_row = signals.loc[
            min_local_idx
        ]

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
# Local Blade Motion Threshold
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
# Blade Motion Groups
# ============================================================

def find_blade_motion_groups(
    signals,
    start_time,
    end_time,
    cfg,
):

    blade_col = (
        f"{cfg.blade_name}_"
        "motion_ratio"
    )

    window = signals[
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
    ].copy()

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
        ].to_numpy()
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
            group["start_idx"]
        )

        end_idx = (
            group["end_idx"]
        )

        local = window.iloc[
            start_idx:
            end_idx + 1
        ]

        peak_idx = (
            local[
                blade_col
            ].idxmax()
        )

        peak_row = window.loc[
            peak_idx
        ]

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
    signals,
    cst_start_time,
    cfg,
):

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

    # CST UP 직전에 끝나는 motion group을
    # Blade IN 후보로 사용.
    #
    # peak magnitude가 가장 큰 group보다
    # temporal proximity를 우선한다.
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
# CST STOP
# ============================================================

def find_cst_stop(
    signals,
    cst_end_time,
    search_end_time,
    cfg,
):

    window = signals[
        (
            signals[
                "time_sec"
            ]
            >= cst_end_time
        )
        &
        (
            signals[
                "time_sec"
            ]
            <= search_end_time
        )
    ].copy()

    if len(window) == 0:
        return None

    mask = (
        np.abs(
            window[
                "cst_dy"
            ].to_numpy(
                dtype=np.float64
            )
        )
        <=
        cfg.cst_stop_abs_threshold
    )

    groups = find_boolean_groups(
        window,
        mask,
        max_gap_sec=0.10,
        min_duration_sec=
            cfg.cst_stop_min_sec,
    )

    if not groups:
        return None

    # CST UP 직후 최초 stable group
    return groups[0]


# ============================================================
# Blade OUT
# ============================================================

def find_blade_out(
    signals,
    search_start_time,
    cfg,
):

    search_end = (
        search_start_time
        + cfg.blade_out_search_sec
    )

    (
        groups,
        threshold,
    ) = find_blade_motion_groups(
        signals,
        search_start_time,
        search_end,
        cfg,
    )

    if not groups:

        return None, threshold

    # CST STOP 이후 최초 motion group을
    # Blade OUT 후보로 사용.
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
    signals,
    cst_events,
    cfg,
):

    detected = []

    for idx, cst in enumerate(
        cst_events,
        start=1,
    ):

        cst_start = (
            cst[
                "cst_start_time"
            ]
        )

        cst_end = (
            cst[
                "cst_end_time"
            ]
        )

        # ----------------------------------------------------
        # Blade IN
        # ----------------------------------------------------

        (
            blade_in,
            blade_in_threshold,
        ) = find_blade_in(
            signals,
            cst_start,
            cfg,
        )

        # ----------------------------------------------------
        # CST STOP
        # ----------------------------------------------------

        stop_search_end = (
            cst_end
            + cfg.blade_out_search_sec
        )

        cst_stop = (
            find_cst_stop(
                signals,
                cst_end,
                stop_search_end,
                cfg,
            )
        )

        # ----------------------------------------------------
        # Blade OUT search start
        # ----------------------------------------------------

        if cst_stop is not None:

            blade_out_search_start = (
                cst_stop[
                    "end_time"
                ]
            )

        else:

            blade_out_search_start = (
                cst_end
            )

        # ----------------------------------------------------
        # Blade OUT
        # ----------------------------------------------------

        (
            blade_out,
            blade_out_threshold,
        ) = find_blade_out(
            signals,
            blade_out_search_start,
            cfg,
        )

        # ----------------------------------------------------
        # Clip start
        # ----------------------------------------------------

        if blade_in is not None:

            clip_start = (
                blade_in[
                    "start_time"
                ]
                -
                cfg.pre_margin_sec
            )

            in_source = (
                "blade"
            )

        else:

            clip_start = (
                cst_start
                -
                cfg.fallback_pre_sec
            )

            in_source = (
                "fallback"
            )

        clip_start = max(
            0.0,
            clip_start,
        )

        # ----------------------------------------------------
        # Clip end
        # ----------------------------------------------------

        if blade_out is not None:

            clip_end = (
                blade_out[
                    "end_time"
                ]
                +
                cfg.post_margin_sec
            )

            out_source = (
                "blade"
            )

        else:

            clip_end = (
                cst_end
                +
                cfg.fallback_post_sec
            )

            out_source = (
                "fallback"
            )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        score = 1.0

        if blade_in is None:
            score -= 0.20

        if cst_stop is None:
            score -= 0.20

        if blade_out is None:
            score -= 0.20

        score = max(
            0.0,
            score,
        )

        detected.append({

            "event_id":
                idx,

            "cst_start_sec":
                cst_start,

            "cst_peak_sec":
                cst[
                    "cst_peak_time"
                ],

            "cst_end_sec":
                cst_end,

            "cst_peak_dy":
                cst[
                    "cst_peak_dy"
                ],

            "cst_peak_frame":
                cst[
                    "cst_peak_frame"
                ],

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

            "cst_stop_start_sec":
                (
                    cst_stop[
                        "start_time"
                    ]
                    if cst_stop
                    else np.nan
                ),

            "cst_stop_end_sec":
                (
                    cst_stop[
                        "end_time"
                    ]
                    if cst_stop
                    else np.nan
                ),

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

            "clip_start_sec":
                clip_start,

            "clip_end_sec":
                clip_end,

            "clip_duration_sec":
                clip_end
                - clip_start,

            "blade_in_source":
                in_source,

            "blade_out_source":
                out_source,

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
    ] = gt[
        "start_sec"
    ].apply(
        parse_video_time
    )

    gt[
        "gt_end_sec"
    ] = gt[
        "end_sec"
    ].apply(
        parse_video_time
    )

    return gt


# ============================================================
# GT Comparison
# ============================================================

def compare_with_gt(
    detected_df,
    gt_df,
    cfg,
):

    gt2 = gt_df[
        gt_df[
            "type"
        ]
        ==
        "type_2"
    ].copy()

    gt2 = gt2.sort_values(
        "gt_start_sec"
    ).reset_index(
        drop=True
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

            # anchor가 GT 구간 또는 tolerance
            # 범위에 들어오는지 검사
            if (
                anchor
                <
                gt_start
                -
                cfg.gt_match_tolerance_sec
                or
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

        used_detected.add(
            best_idx
        )

        det = detected_df.loc[
            best_idx
        ]

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
                det[
                    "event_id"
                ],

            "cst_peak_sec":
                det[
                    "cst_peak_sec"
                ],

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
                det[
                    "clip_start_sec"
                ],

            "clip_end_sec":
                det[
                    "clip_end_sec"
                ],
        })

    comparison = pd.DataFrame(
        rows
    )

    return (
        comparison,
        used_detected,
    )


# ============================================================
# Clip Export
# ============================================================

def export_clips(
    video_path,
    detected_df,
    clip_dir,
):

    clip_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(video_path)
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

        start_sec = float(
            event[
                "clip_start_sec"
            ]
        )

        end_sec = float(
            event[
                "clip_end_sec"
            ]
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

        end_frame = int(
            np.ceil(
                end_sec
                * fps
            )
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
            f"({end_sec-start_sec:.2f}s)"
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

            raise RuntimeError(
                f"Cannot create: "
                f"{output_path}"
            )

        frame_idx = (
            start_frame
        )

        while (
            frame_idx
            <= end_frame
        ):

            ok, frame = cap.read()

            if not ok:
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--signals",
        required=True,
    )

    parser.add_argument(
        "--timestamps",
        required=True,
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="phase_e3_result",
    )

    parser.add_argument(
        "--cst-up-threshold",
        type=float,
        default=-2.0,
    )

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

    parser.add_argument(
        "--no-export",
        action="store_true",
        help=(
            "Detection/GT comparison만 수행하고 "
            "MP4 clip은 저장하지 않음"
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = Config(
        cst_up_threshold=
            args.cst_up_threshold,

        pre_margin_sec=
            args.pre_margin,

        post_margin_sec=
            args.post_margin,
    )

    # ========================================================
    # Load signals
    # ========================================================

    print(
        "Loading motion signals..."
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

    print(
        f"Signal rows: "
        f"{len(signals):,}"
    )

    # ========================================================
    # Detect CST UP
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
    # Build Type2
    # ========================================================

    detected = (
        build_type2_events(
            signals,
            cst_events,
            cfg,
        )
    )

    detected_df = pd.DataFrame(
        detected
    )

    # --------------------------------------------------------
    # Time strings
    # --------------------------------------------------------

    if len(detected_df):

        for column in [
            "cst_peak_sec",
            "clip_start_sec",
            "clip_end_sec",
        ]:

            detected_df[
                column.replace(
                    "_sec",
                    "_time",
                )
            ] = (
                detected_df[
                    column
                ].apply(
                    format_video_time
                )
            )

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
        ).sum()
    )

    comparison, matched_ids = (
        compare_with_gt(
            detected_df,
            gt,
            cfg,
        )
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

    false_positive_count = (
        len(detected_df)
        -
        len(matched_ids)
    )

    recall = (
        matched_count
        /
        gt_type2_count
        if gt_type2_count
        else 0.0
    )

    precision = (
        len(matched_ids)
        /
        len(detected_df)
        if len(detected_df)
        else 0.0
    )

    # ========================================================
    # Console result
    # ========================================================

    print()
    print(
        "========================================"
    )
    print(
        "Phase E-3 Detection Result"
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
        f"{len(detected_df)}"
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
        f"{false_positive_count}"
    )

    print(
        f"Recall              : "
        f"{recall:.4f}"
    )

    print(
        f"Precision           : "
        f"{precision:.4f}"
    )

    # --------------------------------------------------------
    # Sequence completeness
    # --------------------------------------------------------

    if len(detected_df):

        blade_in_ok = int(
            detected_df[
                "blade_in_start_sec"
            ].notna().sum()
        )

        stop_ok = int(
            detected_df[
                "cst_stop_start_sec"
            ].notna().sum()
        )

        blade_out_ok = int(
            detected_df[
                "blade_out_start_sec"
            ].notna().sum()
        )

        print()
        print(
            "Sequence detection"
        )
        print(
            "----------------------------------------"
        )

        print(
            f"Blade IN found       : "
            f"{blade_in_ok}/"
            f"{len(detected_df)}"
        )

        print(
            f"CST STOP found       : "
            f"{stop_ok}/"
            f"{len(detected_df)}"
        )

        print(
            f"Blade OUT found      : "
            f"{blade_out_ok}/"
            f"{len(detected_df)}"
        )

    print()
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
        len(detected_df)
    ):

        print()
        print(
            "Exporting clips..."
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

        print(
            "Clip export complete."
        )


if __name__ == "__main__":
    main()
