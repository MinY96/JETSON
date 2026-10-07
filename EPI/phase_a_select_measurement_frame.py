"""
phase_a_select_measurement_frame.py

Phase A
=======

목적
----
Phase E-3에서 추출한 Type2 clip 각각에 대해

    Blade IN
        ↓
    CST UP
        ↓
    CST STOP
        ↓
    [MEASUREMENT FRAME]
        ↓
    Blade OUT

중 measurement frame 1장을 자동 선택한다.

Phase A에서는 wafer / blade geometry 검출을 수행하지 않는다.
그 작업은 Phase B에서 수행한다.

Input
-----
Phase E-3 clips/

    type2_001.mp4
    type2_002.mp4
    ...
    type2_025.mp4

Output
------
phase_a_result/

    measurement_frames/
        type2_001_measurement.jpg
        ...
        type2_025_measurement.jpg

    debug/
        type2_001_debug.csv
        ...

    measurement_frames.csv


핵심 로직
---------
1. CST ROI에서 phase correlation으로 frame-to-frame dy 계산
2. negative dy가 강한 CST-UP 구간 탐색
3. CST-UP 종료 이후 abs(dy)가 작은 stable 구간 탐색
4. stable 상태가 연속 N frame 이상 유지되는지 확인
5. stable 구간 안쪽의 frame을 measurement frame으로 선택
6. measurement frame을 JPG로 저장


중요
----
Phase A는 "정확한 측정 시점"만 선택한다.

상부 wafer / Blade / 하부 wafer 검출,
gap 계산,
ratio 계산은 Phase B/C에서 수행한다.
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
    # CST ROI
    #
    # Phase E-1과 동일한 ROI
    # x, y, w, h
    # --------------------------------------------------------

    cst_roi: tuple = (
        2039,
        110,
        149,
        654,
    )

    # --------------------------------------------------------
    # Phase correlation preprocessing
    # --------------------------------------------------------

    cst_blur_ksize: int = 5

    # --------------------------------------------------------
    # CST UP detection
    #
    # Type2에서 cst_dy peak가 약 -5 부근이었으므로
    # -2 이하를 CST movement로 본다.
    # --------------------------------------------------------

    cst_up_threshold: float = -2.0

    # --------------------------------------------------------
    # CST STOP / Stable
    #
    # abs(dy) <= 이 값이면 정지 후보
    # --------------------------------------------------------

    stable_abs_dy_threshold: float = 0.30

    # 최소 stable 시간
    # 15 FPS 기준:
    # 0.20 sec ≈ 3 frames
    # --------------------------------------------------------

    stable_min_sec: float = 0.20

    # --------------------------------------------------------
    # Measurement frame
    #
    # stable 구간 시작 후 얼마 뒤를 선택할지
    #
    # 0.10 sec면 15 FPS 기준 약 1~2 frame 뒤
    # --------------------------------------------------------

    measurement_offset_sec: float = 0.10

    # --------------------------------------------------------
    # Search
    #
    # CST-UP peak 이후 이 시간 내에서 STOP 탐색
    # --------------------------------------------------------

    stop_search_sec: float = 1.50

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    save_debug_csv: bool = True

    save_overlay_image: bool = True


# ============================================================
# ROI
# ============================================================

def crop_roi(
    frame: np.ndarray,
    roi: tuple,
) -> np.ndarray:

    x, y, w, h = roi

    frame_h, frame_w = (
        frame.shape[:2]
    )

    x1 = max(
        0,
        int(x),
    )

    y1 = max(
        0,
        int(y),
    )

    x2 = min(
        frame_w,
        int(x + w),
    )

    y2 = min(
        frame_h,
        int(y + h),
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):
        raise ValueError(
            f"Invalid ROI: {roi} "
            f"for frame "
            f"{frame_w}x{frame_h}"
        )

    return frame[
        y1:y2,
        x1:x2
    ]


# ============================================================
# CST preprocessing
# ============================================================

def preprocess_cst(
    frame: np.ndarray,
    cfg: Config,
) -> np.ndarray:

    roi = crop_roi(
        frame,
        cfg.cst_roi,
    )

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY,
    )

    ksize = int(
        cfg.cst_blur_ksize
    )

    if ksize > 1:

        if ksize % 2 == 0:
            ksize += 1

        gray = cv2.GaussianBlur(
            gray,
            (
                ksize,
                ksize,
            ),
            0,
        )

    return gray.astype(
        np.float32
    )


# ============================================================
# Hanning window
# ============================================================

def create_hanning_window(
    image: np.ndarray,
):

    h, w = (
        image.shape[:2]
    )

    return cv2.createHanningWindow(
        (
            w,
            h,
        ),
        cv2.CV_32F,
    )


# ============================================================
# CST motion signal
# ============================================================

def calculate_cst_signal(
    clip_path: Path,
    cfg: Config,
):

    cap = cv2.VideoCapture(
        str(
            clip_path
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open clip: "
            f"{clip_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
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

    rows = []

    prev_cst = None
    hanning = None

    frame_idx = 0

    while True:

        ok, frame = (
            cap.read()
        )

        if (
            not ok
            or
            frame is None
        ):
            break

        cst = preprocess_cst(
            frame,
            cfg,
        )

        if hanning is None:

            hanning = (
                create_hanning_window(
                    cst
                )
            )

        # ----------------------------------------------------
        # 첫 frame
        # ----------------------------------------------------

        if prev_cst is None:

            dx = 0.0
            dy = 0.0
            response = 1.0

        else:

            (
                shift,
                response,
            ) = cv2.phaseCorrelate(
                prev_cst,
                cst,
                hanning,
            )

            dx = float(
                shift[0]
            )

            dy = float(
                shift[1]
            )

            response = float(
                response
            )

        time_sec = (
            frame_idx / fps
            if fps > 0
            else 0.0
        )

        rows.append({

            "frame_idx":
                frame_idx,

            "time_sec":
                time_sec,

            "cst_dx":
                dx,

            "cst_dy":
                dy,

            "phase_response":
                response,
        })

        prev_cst = cst

        frame_idx += 1

    cap.release()

    signal_df = pd.DataFrame(
        rows
    )

    metadata = {

        "fps":
            fps,

        "frame_count":
            frame_count,

        "width":
            width,

        "height":
            height,
    }

    return (
        signal_df,
        metadata,
    )


# ============================================================
# CST UP peak
# ============================================================

def find_cst_up_peak(
    signal_df: pd.DataFrame,
    cfg: Config,
):

    if signal_df.empty:
        return None

    candidates = (
        signal_df[
            signal_df[
                "cst_dy"
            ]
            <=
            cfg.cst_up_threshold
        ]
    )

    if candidates.empty:
        return None

    peak_idx = (
        candidates[
            "cst_dy"
        ]
        .idxmin()
    )

    row = (
        signal_df.loc[
            peak_idx
        ]
    )

    return {

        "frame_idx":
            int(
                row[
                    "frame_idx"
                ]
            ),

        "time_sec":
            float(
                row[
                    "time_sec"
                ]
            ),

        "dy":
            float(
                row[
                    "cst_dy"
                ]
            ),
    }


# ============================================================
# Stable sequence
# ============================================================

def find_stable_sequence(
    signal_df: pd.DataFrame,
    peak_frame_idx: int,
    fps: float,
    cfg: Config,
):
    """
    CST-UP peak 이후 최초의 연속 stable 구간을 찾는다.

    stable:
        abs(cst_dy) <= threshold

    단일 frame이 아니라 stable_min_sec 이상
    연속 유지되어야 STOP으로 인정한다.
    """

    if fps <= 0:
        return None

    stable_frames = max(
        2,
        int(
            np.ceil(
                cfg.stable_min_sec
                *
                fps
            )
        ),
    )

    max_search_frames = max(
        stable_frames,
        int(
            np.ceil(
                cfg.stop_search_sec
                *
                fps
            )
        ),
    )

    start_pos = (
        peak_frame_idx + 1
    )

    end_pos = min(
        len(
            signal_df
        ),
        start_pos
        +
        max_search_frames,
    )

    stable_count = 0
    stable_start_pos = None

    for pos in range(
        start_pos,
        end_pos,
    ):

        dy = float(
            signal_df.iloc[
                pos
            ][
                "cst_dy"
            ]
        )

        is_stable = (
            abs(dy)
            <=
            cfg.stable_abs_dy_threshold
        )

        if is_stable:

            if stable_count == 0:

                stable_start_pos = (
                    pos
                )

            stable_count += 1

            if (
                stable_count
                >=
                stable_frames
            ):

                stable_end_pos = (
                    pos
                )

                return {

                    "start_pos":
                        stable_start_pos,

                    "end_pos":
                        stable_end_pos,

                    "required_frames":
                        stable_frames,

                    "start_frame_idx":
                        int(
                            signal_df.iloc[
                                stable_start_pos
                            ][
                                "frame_idx"
                            ]
                        ),

                    "end_frame_idx":
                        int(
                            signal_df.iloc[
                                stable_end_pos
                            ][
                                "frame_idx"
                            ]
                        ),

                    "start_time_sec":
                        float(
                            signal_df.iloc[
                                stable_start_pos
                            ][
                                "time_sec"
                            ]
                        ),

                    "end_time_sec":
                        float(
                            signal_df.iloc[
                                stable_end_pos
                            ][
                                "time_sec"
                            ]
                        ),
                }

        else:

            stable_count = 0
            stable_start_pos = None

    return None


# ============================================================
# Measurement frame
# ============================================================

def choose_measurement_frame(
    signal_df: pd.DataFrame,
    stable,
    fps: float,
    cfg: Config,
):

    if stable is None:
        return None

    offset_frames = int(
        round(
            cfg.measurement_offset_sec
            *
            fps
        )
    )

    measurement_pos = (
        stable[
            "start_pos"
        ]
        +
        offset_frames
    )

    # 최소한 검증된 stable sequence 안에서 선택
    measurement_pos = min(
        measurement_pos,
        stable[
            "end_pos"
        ],
    )

    row = (
        signal_df.iloc[
            measurement_pos
        ]
    )

    return {

        "pos":
            measurement_pos,

        "frame_idx":
            int(
                row[
                    "frame_idx"
                ]
            ),

        "time_sec":
            float(
                row[
                    "time_sec"
                ]
            ),

        "cst_dy":
            float(
                row[
                    "cst_dy"
                ]
            ),

        "phase_response":
            float(
                row[
                    "phase_response"
                ]
            ),
    }


# ============================================================
# Read exact frame
# ============================================================

def read_frame(
    clip_path: Path,
    frame_idx: int,
):

    cap = cv2.VideoCapture(
        str(
            clip_path
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open clip: "
            f"{clip_path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(
            frame_idx
        ),
    )

    ok, frame = (
        cap.read()
    )

    cap.release()

    if (
        not ok
        or
        frame is None
    ):

        return None

    return frame


# ============================================================
# Overlay
# ============================================================

def draw_debug_overlay(
    frame: np.ndarray,
    clip_name: str,
    measurement,
    peak,
    stable,
    cfg: Config,
):

    output = (
        frame.copy()
    )

    x, y, w, h = (
        cfg.cst_roi
    )

    # CST ROI
    cv2.rectangle(
        output,
        (
            int(x),
            int(y),
        ),
        (
            int(x + w),
            int(y + h),
        ),
        (
            0,
            255,
            255,
        ),
        2,
    )

    lines = [

        f"Phase A Measurement",

        f"Clip: {clip_name}",

        (
            f"Frame: "
            f"{measurement['frame_idx']}"
        ),

        (
            f"Clip time: "
            f"{measurement['time_sec']:.3f}s"
        ),

        (
            f"CST dy: "
            f"{measurement['cst_dy']:.4f}"
        ),

        (
            f"CST peak: "
            f"{peak['time_sec']:.3f}s "
            f"(dy={peak['dy']:.4f})"
        ),

        (
            f"Stable: "
            f"{stable['start_time_sec']:.3f}"
            f"~"
            f"{stable['end_time_sec']:.3f}s"
        ),
    ]

    y_text = 45

    for line in lines:

        cv2.putText(
            output,
            line,
            (
                30,
                y_text,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                0,
                255,
                0,
            ),
            2,
            cv2.LINE_AA,
        )

        y_text += 35

    return output


# ============================================================
# Analyze one clip
# ============================================================

def analyze_clip(
    clip_path: Path,
    measurement_dir: Path,
    overlay_dir: Path,
    debug_dir: Path,
    cfg: Config,
):

    # ========================================================
    # Signal
    # ========================================================

    (
        signal_df,
        metadata,
    ) = calculate_cst_signal(
        clip_path,
        cfg,
    )

    fps = float(
        metadata[
            "fps"
        ]
    )

    # ========================================================
    # CST UP
    # ========================================================

    peak = find_cst_up_peak(
        signal_df,
        cfg,
    )

    if peak is None:

        return {
            "clip":
                clip_path.name,

            "status":
                "cst_up_not_found",
        }

    # ========================================================
    # Stable
    # ========================================================

    stable = find_stable_sequence(
        signal_df,
        peak[
            "frame_idx"
        ],
        fps,
        cfg,
    )

    if stable is None:

        return {
            "clip":
                clip_path.name,

            "status":
                "stable_not_found",

            "fps":
                fps,

            "cst_peak_frame":
                peak[
                    "frame_idx"
                ],

            "cst_peak_sec":
                peak[
                    "time_sec"
                ],

            "cst_peak_dy":
                peak[
                    "dy"
                ],
        }

    # ========================================================
    # Measurement
    # ========================================================

    measurement = (
        choose_measurement_frame(
            signal_df,
            stable,
            fps,
            cfg,
        )
    )

    if measurement is None:

        return {
            "clip":
                clip_path.name,

            "status":
                "measurement_not_found",
        }

    # ========================================================
    # Read frame
    # ========================================================

    frame = read_frame(
        clip_path,
        measurement[
            "frame_idx"
        ],
    )

    if frame is None:

        return {
            "clip":
                clip_path.name,

            "status":
                "frame_read_failed",
        }

    # ========================================================
    # Save measurement frame
    # ========================================================

    measurement_path = (
        measurement_dir
        /
        (
            f"{clip_path.stem}"
            "_measurement.jpg"
        )
    )

    cv2.imwrite(
        str(
            measurement_path
        ),
        frame,
    )

    # ========================================================
    # Overlay
    # ========================================================

    overlay_path = None

    if cfg.save_overlay_image:

        overlay = (
            draw_debug_overlay(
                frame,
                clip_path.name,
                measurement,
                peak,
                stable,
                cfg,
            )
        )

        overlay_path = (
            overlay_dir
            /
            (
                f"{clip_path.stem}"
                "_measurement_debug.jpg"
            )
        )

        cv2.imwrite(
            str(
                overlay_path
            ),
            overlay,
        )

    # ========================================================
    # Debug CSV
    # ========================================================

    if cfg.save_debug_csv:

        debug_df = (
            signal_df.copy()
        )

        debug_df[
            "is_cst_up"
        ] = (
            debug_df[
                "cst_dy"
            ]
            <=
            cfg.cst_up_threshold
        )

        debug_df[
            "is_stable"
        ] = (
            debug_df[
                "cst_dy"
            ]
            .abs()
            <=
            cfg.stable_abs_dy_threshold
        )

        debug_df[
            "is_measurement"
        ] = False

        debug_df.loc[
            debug_df[
                "frame_idx"
            ]
            ==
            measurement[
                "frame_idx"
            ],
            "is_measurement",
        ] = True

        debug_path = (
            debug_dir
            /
            (
                f"{clip_path.stem}"
                "_debug.csv"
            )
        )

        debug_df.to_csv(
            debug_path,
            index=False,
            encoding="utf-8-sig",
        )

    # ========================================================
    # Result
    # ========================================================

    stable_duration = (
        stable[
            "end_time_sec"
        ]
        -
        stable[
            "start_time_sec"
        ]
    )

    return {

        "clip":
            clip_path.name,

        "status":
            "ok",

        "fps":
            fps,

        "frame_count":
            metadata[
                "frame_count"
            ],

        "cst_peak_frame":
            peak[
                "frame_idx"
            ],

        "cst_peak_sec":
            peak[
                "time_sec"
            ],

        "cst_peak_dy":
            peak[
                "dy"
            ],

        "stable_start_frame":
            stable[
                "start_frame_idx"
            ],

        "stable_end_frame":
            stable[
                "end_frame_idx"
            ],

        "stable_start_sec":
            stable[
                "start_time_sec"
            ],

        "stable_end_sec":
            stable[
                "end_time_sec"
            ],

        "stable_duration_sec":
            stable_duration,

        "measurement_frame_idx":
            measurement[
                "frame_idx"
            ],

        "measurement_sec":
            measurement[
                "time_sec"
            ],

        "measurement_cst_dy":
            measurement[
                "cst_dy"
            ],

        "measurement_phase_response":
            measurement[
                "phase_response"
            ],

        "measurement_image":
            str(
                measurement_path
            ),

        "overlay_image":
            (
                str(
                    overlay_path
                )
                if overlay_path
                else ""
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase A - "
            "Select measurement frame "
            "from Type2 clips"
        )
    )

    parser.add_argument(
        "--clips",
        required=True,
        help=(
            "Phase E-3 clips directory"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "phase_a_result"
        ),
    )

    parser.add_argument(
        "--stable-threshold",
        type=float,
        default=0.30,
        help=(
            "abs(cst_dy) threshold "
            "for CST stable state"
        ),
    )

    parser.add_argument(
        "--stable-sec",
        type=float,
        default=0.20,
        help=(
            "Minimum continuous stable time"
        ),
    )

    parser.add_argument(
        "--measurement-offset",
        type=float,
        default=0.10,
        help=(
            "Measurement frame offset "
            "from stable start"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Config
    # ========================================================

    cfg = Config(
        stable_abs_dy_threshold=
            args.stable_threshold,

        stable_min_sec=
            args.stable_sec,

        measurement_offset_sec=
            args.measurement_offset,
    )

    clips_dir = Path(
        args.clips
    )

    output_dir = Path(
        args.output
    )

    measurement_dir = (
        output_dir
        /
        "measurement_frames"
    )

    overlay_dir = (
        output_dir
        /
        "overlay"
    )

    debug_dir = (
        output_dir
        /
        "debug"
    )

    for directory in [
        output_dir,
        measurement_dir,
        overlay_dir,
        debug_dir,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Find clips
    # ========================================================

    clips = sorted(
        clips_dir.glob(
            "*.mp4"
        )
    )

    if not clips:

        raise FileNotFoundError(
            f"No MP4 clips found: "
            f"{clips_dir}"
        )

    print(
        "========================================"
    )

    print(
        "Phase A - Measurement Frame Selection"
    )

    print(
        "========================================"
    )

    print(
        f"Clips             : "
        f"{len(clips)}"
    )

    print(
        f"CST UP threshold  : "
        f"{cfg.cst_up_threshold:.3f}"
    )

    print(
        f"Stable threshold  : "
        f"|dy| <= "
        f"{cfg.stable_abs_dy_threshold:.3f}"
    )

    print(
        f"Stable duration   : "
        f"{cfg.stable_min_sec:.3f}s"
    )

    print(
        f"Measure offset    : "
        f"{cfg.measurement_offset_sec:.3f}s"
    )

    print()

    # ========================================================
    # Analyze
    # ========================================================

    results = []

    for (
        index,
        clip_path,
    ) in enumerate(
        clips,
        start=1,
    ):

        print(
            f"[{index:02d}/{len(clips):02d}] "
            f"{clip_path.name}"
        )

        try:

            result = (
                analyze_clip(
                    clip_path,
                    measurement_dir,
                    overlay_dir,
                    debug_dir,
                    cfg,
                )
            )

        except Exception as exc:

            result = {

                "clip":
                    clip_path.name,

                "status":
                    "error",

                "error":
                    str(
                        exc
                    ),
            }

        results.append(
            result
        )

        status = (
            result[
                "status"
            ]
        )

        if status == "ok":

            print(
                f"    CST peak    : "
                f"{result['cst_peak_sec']:.3f}s "
                f"dy="
                f"{result['cst_peak_dy']:.4f}"
            )

            print(
                f"    Stable      : "
                f"{result['stable_start_sec']:.3f}s "
                f"~ "
                f"{result['stable_end_sec']:.3f}s"
            )

            print(
                f"    Measurement : "
                f"frame "
                f"{result['measurement_frame_idx']} "
                f"@ "
                f"{result['measurement_sec']:.3f}s"
            )

        else:

            print(
                f"    FAILED: "
                f"{status}"
            )

    # ========================================================
    # Save result
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    result_path = (
        output_dir
        /
        "measurement_frames.csv"
    )

    result_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Summary
    # ========================================================

    ok_count = int(
        (
            result_df[
                "status"
            ]
            ==
            "ok"
        )
        .sum()
    )

    fail_count = (
        len(
            result_df
        )
        -
        ok_count
    )

    print()

    print(
        "========================================"
    )

    print(
        "Phase A Result"
    )

    print(
        "========================================"
    )

    print(
        f"Total   : "
        f"{len(result_df)}"
    )

    print(
        f"Success : "
        f"{ok_count}"
    )

    print(
        f"Failed  : "
        f"{fail_count}"
    )

    print()

    print(
        f"CSV     : "
        f"{result_path}"
    )

    print(
        f"Images  : "
        f"{measurement_dir}"
    )

    print(
        f"Overlay : "
        f"{overlay_dir}"
    )

    print(
        f"Debug   : "
        f"{debug_dir}"
    )


if __name__ == "__main__":
    main()
