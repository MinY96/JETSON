"""
Phase A - Type 2 Measurement Frame Detector
===========================================

목적
----
이미 Type 2로 분류된 3~5초 길이의 clip에서

    Blade IN
        ->
    CST UP
        ->
    CST STOP
        ->
    [ MEASUREMENT ]
        ->
    Blade OUT

시퀀스를 이용하여 gap 측정에 사용할 최적의 프레임 1장을 선택한다.

중요
----
- Wafer line detection 없음
- Blade line detection 없음
- Gap measurement 없음
- Timing detection만 수행

출력
----
각 clip별 폴더:

    measurement_frame.jpg
    timing_debug.csv
    timing_debug.mp4

전체:

    phase_a_results.csv
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Config
# ============================================================

@dataclass
class Config:

    # 분석 속도 / 안정성을 위한 resize
    analysis_width: int = 640

    # --------------------------------------------------------
    # CST motion
    # --------------------------------------------------------

    # phase correlation 결과 중 너무 작은 이동은 noise로 취급
    cst_motion_noise_px: float = 0.08

    # CST가 실제 이동 중이라고 판단할 최소 |dy|
    cst_move_threshold: float = 0.18

    # CST 정지 판단 threshold
    cst_stable_threshold: float = 0.10

    # phase correlation 신뢰도 최소값
    phase_response_min: float = 0.05

    # CST 이동 최소 지속시간
    cst_min_move_sec: float = 0.06

    # --------------------------------------------------------
    # Blade motion
    # --------------------------------------------------------

    # absdiff threshold
    blade_diff_pixel_threshold: int = 12

    # Blade ROI 중 몇 % 이상 pixel이 변하면 motion으로 볼지
    blade_motion_ratio_threshold: float = 0.015

    # Blade OUT 최소 지속시간
    blade_out_min_sec: float = 0.08

    # --------------------------------------------------------
    # Sequence
    # --------------------------------------------------------

    # clip 앞부분은 탐색에서 제외
    search_start_ratio: float = 0.10

    # clip 끝부분도 약간 제외
    search_end_ratio: float = 0.95

    # CST UP 종료 후 Blade OUT이 나타날 수 있는 최대 시간
    max_stop_to_blade_out_sec: float = 0.8

    # CST가 멈춘 후 최소 안정 시간
    min_stable_sec: float = 0.05

    # Blade OUT 직전 몇 초까지 measurement window 후보로 볼지
    measurement_lookback_sec: float = 0.35

    # Blade OUT 바로 직전은 움직임이 시작될 수 있으므로 제외
    blade_out_guard_sec: float = 0.03

    # --------------------------------------------------------
    # Measurement frame score
    # --------------------------------------------------------

    # motion penalty
    cst_motion_weight: float = 1.0
    blade_motion_weight: float = 2.0

    # sharpness reward
    sharpness_weight: float = 0.15

    # --------------------------------------------------------
    # smoothing
    # --------------------------------------------------------

    smooth_sec: float = 0.06

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    debug_video: bool = True
    debug_scale: float = 0.7


# ============================================================
# Utility
# ============================================================

def resize_keep_ratio(img, target_width):

    h, w = img.shape[:2]

    if w <= target_width:
        return img.copy(), 1.0

    scale = target_width / w

    nh = max(1, round(h * scale))

    out = cv2.resize(
        img,
        (target_width, nh),
        interpolation=cv2.INTER_AREA
    )

    return out, scale


def smooth_signal(values, window):

    arr = np.asarray(values, dtype=np.float32)

    if len(arr) == 0:
        return arr

    window = max(1, int(window))

    if window <= 1:
        return arr.copy()

    kernel = np.ones(window, dtype=np.float32) / window

    return np.convolve(
        arr,
        kernel,
        mode="same"
    )


def calc_sharpness(gray):

    lap = cv2.Laplacian(
        gray,
        cv2.CV_32F
    )

    return float(lap.var())


# ============================================================
# ROI selection
# ============================================================

def select_roi(frame, title):

    h, w = frame.shape[:2]

    scale = min(
        1280 / w,
        800 / h,
        1.0
    )

    if scale < 1.0:
        preview = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale
        )
    else:
        preview = frame.copy()

    roi = cv2.selectROI(
        title,
        preview,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(title)

    x, y, rw, rh = roi

    if rw <= 0 or rh <= 0:
        raise RuntimeError(
            f"ROI not selected: {title}"
        )

    return (
        round(x / scale),
        round(y / scale),
        round(rw / scale),
        round(rh / scale)
    )


# ============================================================
# ROI preprocessing
# ============================================================

def crop_and_prepare(
    frame,
    roi,
    target_width
):

    x, y, w, h = roi

    crop = frame[
        y:y+h,
        x:x+w
    ]

    small, scale = resize_keep_ratio(
        crop,
        target_width
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return gray, scale


# ============================================================
# CST motion
# ============================================================

def phase_shift(
    prev_gray,
    curr_gray
):

    a = prev_gray.astype(
        np.float32
    )

    b = curr_gray.astype(
        np.float32
    )

    # Hanning window reduces boundary artifact
    window = cv2.createHanningWindow(
        (a.shape[1], a.shape[0]),
        cv2.CV_32F
    )

    shift, response = cv2.phaseCorrelate(
        a,
        b,
        window
    )

    dx, dy = shift

    return (
        float(dx),
        float(dy),
        float(response)
    )


# ============================================================
# Blade motion
# ============================================================

def blade_motion_score(
    prev_gray,
    curr_gray,
    pixel_threshold
):

    diff = cv2.absdiff(
        prev_gray,
        curr_gray
    )

    changed = (
        diff >= pixel_threshold
    )

    ratio = float(
        changed.mean()
    )

    mean_diff = float(
        diff.mean()
    )

    return ratio, mean_diff


# ============================================================
# Clip signal extraction
# ============================================================

def extract_signals(
    video_path,
    cst_roi,
    blade_roi,
    cfg
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frames = []
    records = []

    ok, first = cap.read()

    if not ok:
        cap.release()

        raise RuntimeError(
            f"Cannot read video: {video_path}"
        )

    frames.append(first)

    prev_cst, _ = crop_and_prepare(
        first,
        cst_roi,
        cfg.analysis_width
    )

    prev_blade, _ = crop_and_prepare(
        first,
        blade_roi,
        cfg.analysis_width
    )

    records.append({
        "frame_idx": 0,
        "time_sec": 0.0,

        "cst_dx": 0.0,
        "cst_dy": 0.0,
        "phase_response": 0.0,

        "blade_motion_ratio": 0.0,
        "blade_mean_diff": 0.0,

        "sharpness": calc_sharpness(
            prev_blade
        )
    })

    frame_idx = 1

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frames.append(frame)

        # ----------------------------------------------
        # CST
        # ----------------------------------------------

        curr_cst, _ = crop_and_prepare(
            frame,
            cst_roi,
            cfg.analysis_width
        )

        dx, dy, response = phase_shift(
            prev_cst,
            curr_cst
        )

        # unreliable correlation -> ignore
        if response < cfg.phase_response_min:
            dx = 0.0
            dy = 0.0

        # very small movement -> noise
        if abs(dy) < cfg.cst_motion_noise_px:
            dy = 0.0

        # ----------------------------------------------
        # Blade
        # ----------------------------------------------

        curr_blade, _ = crop_and_prepare(
            frame,
            blade_roi,
            cfg.analysis_width
        )

        motion_ratio, mean_diff = (
            blade_motion_score(
                prev_blade,
                curr_blade,
                cfg.blade_diff_pixel_threshold
            )
        )

        sharpness = calc_sharpness(
            curr_blade
        )

        records.append({
            "frame_idx": frame_idx,
            "time_sec": frame_idx / fps,

            "cst_dx": dx,
            "cst_dy": dy,
            "phase_response": response,

            "blade_motion_ratio": motion_ratio,
            "blade_mean_diff": mean_diff,

            "sharpness": sharpness
        })

        prev_cst = curr_cst
        prev_blade = curr_blade

        frame_idx += 1

    cap.release()

    return (
        frames,
        records,
        fps,
        frame_width,
        frame_height
    )


# ============================================================
# Temporal signal processing
# ============================================================

def prepare_temporal_signals(
    records,
    fps,
    cfg
):

    cst_dy = np.array(
        [
            r["cst_dy"]
            for r in records
        ],
        dtype=np.float32
    )

    blade_motion = np.array(
        [
            r["blade_motion_ratio"]
            for r in records
        ],
        dtype=np.float32
    )

    sharpness = np.array(
        [
            r["sharpness"]
            for r in records
        ],
        dtype=np.float32
    )

    smooth_n = max(
        1,
        round(
            cfg.smooth_sec * fps
        )
    )

    cst_dy_smooth = smooth_signal(
        cst_dy,
        smooth_n
    )

    blade_motion_smooth = smooth_signal(
        blade_motion,
        smooth_n
    )

    return {
        "cst_dy": cst_dy,
        "cst_dy_smooth": cst_dy_smooth,

        "blade_motion": blade_motion,
        "blade_motion_smooth":
            blade_motion_smooth,

        "sharpness": sharpness
    }


# ============================================================
# Segment helper
# ============================================================

def find_segments(
    mask,
    start_idx,
    end_idx,
    min_length
):

    segments = []

    in_segment = False
    seg_start = None

    for i in range(
        start_idx,
        end_idx
    ):

        if mask[i]:

            if not in_segment:
                seg_start = i
                in_segment = True

        else:

            if in_segment:

                seg_end = i - 1

                if (
                    seg_end
                    - seg_start
                    + 1
                    >= min_length
                ):
                    segments.append(
                        (
                            seg_start,
                            seg_end
                        )
                    )

                in_segment = False

    if in_segment:

        seg_end = end_idx - 1

        if (
            seg_end
            - seg_start
            + 1
            >= min_length
        ):
            segments.append(
                (
                    seg_start,
                    seg_end
                )
            )

    return segments


# ============================================================
# Sequence detection
# ============================================================

def detect_sequence(
    records,
    signals,
    fps,
    cfg
):

    n = len(records)

    cst = signals[
        "cst_dy_smooth"
    ]

    blade = signals[
        "blade_motion_smooth"
    ]

    sharpness = signals[
        "sharpness"
    ]

    search_start = max(
        1,
        round(
            n * cfg.search_start_ratio
        )
    )

    search_end = min(
        n - 1,
        round(
            n * cfg.search_end_ratio
        )
    )

    # ========================================================
    # 1. Blade motion segments
    # ========================================================

    blade_mask = (
        blade
        >= cfg.blade_motion_ratio_threshold
    )

    blade_min_frames = max(
        2,
        round(
            cfg.blade_out_min_sec
            * fps
        )
    )

    blade_segments = find_segments(
        blade_mask,
        search_start,
        search_end,
        blade_min_frames
    )

    # ========================================================
    # 2. CST vertical motion segments
    # ========================================================

    cst_mask = (
        np.abs(cst)
        >= cfg.cst_move_threshold
    )

    cst_min_frames = max(
        2,
        round(
            cfg.cst_min_move_sec
            * fps
        )
    )

    cst_segments = find_segments(
        cst_mask,
        search_start,
        search_end,
        cst_min_frames
    )

    # ========================================================
    # 3. Sequence candidates
    #
    # CST movement -> STOP -> Blade motion
    # ========================================================

    candidates = []

    max_gap_frames = max(
        1,
        round(
            cfg.max_stop_to_blade_out_sec
            * fps
        )
    )

    min_stable_frames = max(
        1,
        round(
            cfg.min_stable_sec
            * fps
        )
    )

    for blade_start, blade_end in blade_segments:

        # ----------------------------------------------------
        # Find CST segment immediately before this blade motion
        # ----------------------------------------------------

        previous_cst = [
            seg
            for seg in cst_segments
            if seg[1] < blade_start
        ]

        if not previous_cst:
            continue

        # nearest CST movement before Blade OUT
        cst_start, cst_end = max(
            previous_cst,
            key=lambda x: x[1]
        )

        gap = (
            blade_start
            - cst_end
            - 1
        )

        if gap < min_stable_frames:
            continue

        if gap > max_gap_frames:
            continue

        # ----------------------------------------------------
        # Stable interval
        # ----------------------------------------------------

        stable_start = (
            cst_end + 1
        )

        stable_end = (
            blade_start - 1
        )

        if stable_end <= stable_start:
            continue

        stable_cst = float(
            np.mean(
                np.abs(
                    cst[
                        stable_start:
                        stable_end + 1
                    ]
                )
            )
        )

        stable_blade = float(
            np.mean(
                blade[
                    stable_start:
                    stable_end + 1
                ]
            )
        )

        cst_strength = float(
            np.mean(
                np.abs(
                    cst[
                        cst_start:
                        cst_end + 1
                    ]
                )
            )
        )

        blade_strength = float(
            np.mean(
                blade[
                    blade_start:
                    blade_end + 1
                ]
            )
        )

        # ----------------------------------------------------
        # Candidate score
        # ----------------------------------------------------

        score = (
            cst_strength
            + blade_strength * 5.0
            - stable_cst * 2.0
            - stable_blade * 5.0
        )

        candidates.append({
            "score": score,

            "cst_start": cst_start,
            "cst_end": cst_end,

            "blade_out_start":
                blade_start,

            "blade_out_end":
                blade_end,

            "stable_start":
                stable_start,

            "stable_end":
                stable_end,

            "stable_cst":
                stable_cst,

            "stable_blade":
                stable_blade,

            "cst_strength":
                cst_strength,

            "blade_strength":
                blade_strength
        })

    # ========================================================
    # Fallback
    # ========================================================

    if not candidates:

        return fallback_measurement_frame(
            records,
            signals,
            fps,
            cfg
        )

    # strongest valid sequence
    best = max(
        candidates,
        key=lambda x: x["score"]
    )

    # ========================================================
    # 4. Measurement window
    # ========================================================

    blade_out_start = best[
        "blade_out_start"
    ]

    cst_end = best[
        "cst_end"
    ]

    lookback_frames = max(
        1,
        round(
            cfg.measurement_lookback_sec
            * fps
        )
    )

    guard_frames = max(
        0,
        round(
            cfg.blade_out_guard_sec
            * fps
        )
    )

    window_start = max(
        cst_end + 1,
        blade_out_start
        - lookback_frames
    )

    window_end = (
        blade_out_start
        - guard_frames
        - 1
    )

    if window_end < window_start:

        window_start = (
            cst_end + 1
        )

        window_end = (
            blade_out_start - 1
        )

    # ========================================================
    # 5. Select best frame inside measurement window
    # ========================================================

    sharp = sharpness[
        window_start:
        window_end + 1
    ]

    if len(sharp) == 0:

        measurement_frame = (
            cst_end + 1
        )

    else:

        # normalize sharpness
        sharp_min = float(
            sharp.min()
        )

        sharp_max = float(
            sharp.max()
        )

        sharp_norm = (
            sharp - sharp_min
        ) / max(
            sharp_max - sharp_min,
            1e-6
        )

        best_frame = None
        best_frame_score = None

        for k, frame_idx in enumerate(
            range(
                window_start,
                window_end + 1
            )
        ):

            cst_motion = abs(
                float(cst[frame_idx])
            )

            blade_motion = float(
                blade[frame_idx]
            )

            sharp_reward = float(
                sharp_norm[k]
            )

            frame_score = (
                cfg.cst_motion_weight
                * cst_motion
                +
                cfg.blade_motion_weight
                * blade_motion
                -
                cfg.sharpness_weight
                * sharp_reward
            )

            if (
                best_frame_score is None
                or
                frame_score
                < best_frame_score
            ):

                best_frame_score = (
                    frame_score
                )

                best_frame = (
                    frame_idx
                )

        measurement_frame = (
            best_frame
        )

    best.update({
        "status": "OK",

        "measurement_window_start":
            window_start,

        "measurement_window_end":
            window_end,

        "measurement_frame":
            measurement_frame,

        "fallback":
            False
    })

    return best


# ============================================================
# Fallback
# ============================================================

def fallback_measurement_frame(
    records,
    signals,
    fps,
    cfg
):

    """
    정상적인 CST -> STOP -> Blade sequence를 찾지 못했을 경우.

    clip 후반부에서
    CST + Blade motion이 가장 작은 frame을 선택한다.

    이 결과는 반드시 fallback=True로 기록한다.
    """

    n = len(records)

    cst = signals[
        "cst_dy_smooth"
    ]

    blade = signals[
        "blade_motion_smooth"
    ]

    sharp = signals[
        "sharpness"
    ]

    start = max(
        1,
        round(n * 0.40)
    )

    end = max(
        start + 1,
        round(n * 0.90)
    )

    end = min(
        end,
        n
    )

    local_sharp = sharp[
        start:end
    ]

    smin = float(
        local_sharp.min()
    )

    smax = float(
        local_sharp.max()
    )

    best_idx = None
    best_score = None

    for i in range(
        start,
        end
    ):

        sn = (
            sharp[i] - smin
        ) / max(
            smax - smin,
            1e-6
        )

        score = (
            abs(cst[i])
            +
            blade[i] * 2.0
            -
            sn * 0.10
        )

        if (
            best_score is None
            or score < best_score
        ):

            best_score = score
            best_idx = i

    return {
        "status":
            "FALLBACK",

        "score":
            -float(best_score),

        "cst_start":
            -1,

        "cst_end":
            -1,

        "blade_out_start":
            -1,

        "blade_out_end":
            -1,

        "stable_start":
            -1,

        "stable_end":
            -1,

        "measurement_window_start":
            start,

        "measurement_window_end":
            end - 1,

        "measurement_frame":
            best_idx,

        "fallback":
            True
    }


# ============================================================
# Save CSV
# ============================================================

def save_debug_csv(
    path,
    records,
    signals,
    result
):

    measurement_frame = result[
        "measurement_frame"
    ]

    cst_start = result.get(
        "cst_start",
        -1
    )

    cst_end = result.get(
        "cst_end",
        -1
    )

    blade_start = result.get(
        "blade_out_start",
        -1
    )

    blade_end = result.get(
        "blade_out_end",
        -1
    )

    win_start = result.get(
        "measurement_window_start",
        -1
    )

    win_end = result.get(
        "measurement_window_end",
        -1
    )

    fields = [
        "frame_idx",
        "time_sec",

        "cst_dx",
        "cst_dy",
        "cst_dy_smooth",

        "phase_response",

        "blade_motion_ratio",
        "blade_motion_smooth",
        "blade_mean_diff",

        "sharpness",

        "state",

        "is_measurement_frame"
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for i, r in enumerate(records):

            state = "SEARCH"

            if (
                cst_start
                <= i
                <= cst_end
                and
                cst_start >= 0
            ):
                state = "CST_MOVE"

            elif (
                win_start
                <= i
                <= win_end
                and
                win_start >= 0
            ):
                state = (
                    "MEASUREMENT_WINDOW"
                )

            elif (
                blade_start
                <= i
                <= blade_end
                and
                blade_start >= 0
            ):
                state = "BLADE_OUT"

            row = {
                "frame_idx":
                    r["frame_idx"],

                "time_sec":
                    r["time_sec"],

                "cst_dx":
                    r["cst_dx"],

                "cst_dy":
                    r["cst_dy"],

                "cst_dy_smooth":
                    float(
                        signals[
                            "cst_dy_smooth"
                        ][i]
                    ),

                "phase_response":
                    r["phase_response"],

                "blade_motion_ratio":
                    r[
                        "blade_motion_ratio"
                    ],

                "blade_motion_smooth":
                    float(
                        signals[
                            "blade_motion_smooth"
                        ][i]
                    ),

                "blade_mean_diff":
                    r[
                        "blade_mean_diff"
                    ],

                "sharpness":
                    r["sharpness"],

                "state":
                    state,

                "is_measurement_frame":
                    int(
                        i
                        == measurement_frame
                    )
            }

            writer.writerow(row)


# ============================================================
# Debug video
# ============================================================

def save_debug_video(
    path,
    frames,
    records,
    signals,
    result,
    fps,
    fw,
    fh,
    cst_roi,
    blade_roi,
    cfg
):

    if not cfg.debug_video:
        return

    out_w = max(
        1,
        round(
            fw
            * cfg.debug_scale
        )
    )

    out_h = max(
        1,
        round(
            fh
            * cfg.debug_scale
        )
    )

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            out_w,
            out_h
        )
    )

    measurement_frame = result[
        "measurement_frame"
    ]

    cst_start = result.get(
        "cst_start",
        -1
    )

    cst_end = result.get(
        "cst_end",
        -1
    )

    blade_start = result.get(
        "blade_out_start",
        -1
    )

    blade_end = result.get(
        "blade_out_end",
        -1
    )

    win_start = result.get(
        "measurement_window_start",
        -1
    )

    win_end = result.get(
        "measurement_window_end",
        -1
    )

    for i, frame in enumerate(frames):

        vis = frame.copy()

        # ----------------------------------------------
        # ROI
        # ----------------------------------------------

        x, y, w, h = cst_roi

        cv2.rectangle(
            vis,
            (x, y),
            (x+w, y+h),
            (255, 255, 0),
            2
        )

        cv2.putText(
            vis,
            "CST MOTION ROI",
            (x, max(20, y-8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

        x, y, w, h = blade_roi

        cv2.rectangle(
            vis,
            (x, y),
            (x+w, y+h),
            (0, 255, 255),
            2
        )

        cv2.putText(
            vis,
            "BLADE MOTION ROI",
            (x, max(20, y-8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------
        # State
        # ----------------------------------------------

        state = "SEARCH"

        if (
            cst_start
            <= i
            <= cst_end
            and
            cst_start >= 0
        ):

            state = "CST MOVE"

        elif (
            win_start
            <= i
            <= win_end
            and
            win_start >= 0
        ):

            state = (
                "MEASUREMENT WINDOW"
            )

        elif (
            blade_start
            <= i
            <= blade_end
            and
            blade_start >= 0
        ):

            state = "BLADE OUT"

        # ----------------------------------------------
        # Panel
        # ----------------------------------------------

        cv2.rectangle(
            vis,
            (15, 15),
            (620, 185),
            (0, 0, 0),
            -1
        )

        lines = [

            (
                f"Frame={i}  "
                f"Time={i/fps:.3f}s"
            ),

            (
                "CST dy="
                f"{signals['cst_dy_smooth'][i]:+.4f}"
            ),

            (
                "Phase response="
                f"{records[i]['phase_response']:.4f}"
            ),

            (
                "Blade motion="
                f"{signals['blade_motion_smooth'][i]:.5f}"
            ),

            (
                "Sharpness="
                f"{records[i]['sharpness']:.1f}"
            ),

            f"State={state}"
        ]

        for j, text in enumerate(
            lines
        ):

            cv2.putText(
                vis,
                text,
                (
                    30,
                    43 + j * 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------
        # Measurement frame
        # ----------------------------------------------

        if i == measurement_frame:

            cv2.rectangle(
                vis,
                (5, 5),
                (
                    fw - 6,
                    fh - 6
                ),
                (0, 255, 0),
                5
            )

            cv2.putText(
                vis,
                "MEASUREMENT FRAME",
                (
                    max(
                        20,
                        fw // 2 - 220
                    ),
                    70
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        vis = cv2.resize(
            vis,
            (
                out_w,
                out_h
            )
        )

        writer.write(vis)

    writer.release()


# ============================================================
# Analyze one clip
# ============================================================

def analyze_clip(
    video_path,
    output_root,
    cst_roi,
    blade_roi,
    cfg
):

    print(
        f"\nAnalyzing: {video_path.name}"
    )

    (
        frames,
        records,
        fps,
        fw,
        fh
    ) = extract_signals(
        video_path,
        cst_roi,
        blade_roi,
        cfg
    )

    signals = (
        prepare_temporal_signals(
            records,
            fps,
            cfg
        )
    )

    result = detect_sequence(
        records,
        signals,
        fps,
        cfg
    )

    clip_dir = (
        output_root
        / video_path.stem
    )

    clip_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    measurement_frame = result[
        "measurement_frame"
    ]

    measurement_path = (
        clip_dir
        / "measurement_frame.jpg"
    )

    cv2.imwrite(
        str(measurement_path),
        frames[
            measurement_frame
        ]
    )

    save_debug_csv(
        clip_dir
        / "timing_debug.csv",
        records,
        signals,
        result
    )

    save_debug_video(
        clip_dir
        / "timing_debug.mp4",
        frames,
        records,
        signals,
        result,
        fps,
        fw,
        fh,
        cst_roi,
        blade_roi,
        cfg
    )

    summary = {

        "clip":
            video_path.name,

        "status":
            result["status"],

        "fallback":
            result["fallback"],

        "fps":
            fps,

        "total_frames":
            len(frames),

        "measurement_frame":
            measurement_frame,

        "measurement_time_sec":
            measurement_frame / fps,

        "cst_start":
            result.get(
                "cst_start",
                -1
            ),

        "cst_end":
            result.get(
                "cst_end",
                -1
            ),

        "blade_out_start":
            result.get(
                "blade_out_start",
                -1
            ),

        "blade_out_end":
            result.get(
                "blade_out_end",
                -1
            ),

        "window_start":
            result.get(
                "measurement_window_start",
                -1
            ),

        "window_end":
            result.get(
                "measurement_window_end",
                -1
            ),

        "sequence_score":
            result.get(
                "score",
                0.0
            ),

        "measurement_path":
            str(
                measurement_path
            )
    }

    print(
        "  status       :",
        summary["status"]
    )

    print(
        "  CST          :",
        summary["cst_start"],
        "~",
        summary["cst_end"]
    )

    print(
        "  window       :",
        summary["window_start"],
        "~",
        summary["window_end"]
    )

    print(
        "  blade out    :",
        summary["blade_out_start"],
        "~",
        summary["blade_out_end"]
    )

    print(
        "  measurement  :",
        summary[
            "measurement_frame"
        ],
        f"({summary['measurement_time_sec']:.3f}s)"
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        default="./phase_a_result"
    )

    parser.add_argument(
        "--cst-roi",
        nargs=4,
        type=int,
        default=None,
        metavar=(
            "X",
            "Y",
            "W",
            "H"
        )
    )

    parser.add_argument(
        "--blade-roi",
        nargs=4,
        type=int,
        default=None,
        metavar=(
            "X",
            "Y",
            "W",
            "H"
        )
    )

    parser.add_argument(
        "--analysis-width",
        type=int,
        default=640
    )

    parser.add_argument(
        "--no-debug-video",
        action="store_true"
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    videos = sorted(
        input_dir.glob("*.mp4")
    )

    if not videos:

        raise FileNotFoundError(
            f"No mp4 files: {input_dir}"
        )

    cfg = Config(
        analysis_width=
            args.analysis_width,

        debug_video=
            not args.no_debug_video
    )

    # ========================================================
    # ROI
    # ========================================================

    cst_roi = (
        tuple(args.cst_roi)
        if args.cst_roi
        else None
    )

    blade_roi = (
        tuple(args.blade_roi)
        if args.blade_roi
        else None
    )

    if (
        cst_roi is None
        or
        blade_roi is None
    ):

        cap = cv2.VideoCapture(
            str(videos[0])
        )

        ok, frame = cap.read()

        cap.release()

        if not ok:

            raise RuntimeError(
                "Cannot read first clip"
            )

        if cst_roi is None:

            print(
                "\nSelect CST Motion ROI"
            )

            cst_roi = select_roi(
                frame,
                "Select CST Motion ROI"
            )

        if blade_roi is None:

            print(
                "\nSelect Blade Motion ROI"
            )

            blade_roi = select_roi(
                frame,
                "Select Blade Motion ROI"
            )

    print(
        "\n================================"
    )

    print(
        "Phase A Measurement Detector"
    )

    print(
        "================================"
    )

    print(
        "CST ROI   :",
        cst_roi
    )

    print(
        "Blade ROI :",
        blade_roi
    )

    print(
        "Clips     :",
        len(videos)
    )

    # ========================================================
    # Analyze
    # ========================================================

    summaries = []

    for idx, video in enumerate(
        videos,
        1
    ):

        print(
            f"\n[{idx}/{len(videos)}]"
        )

        try:

            summary = analyze_clip(
                video,
                output_dir,
                cst_roi,
                blade_roi,
                cfg
            )

        except Exception as e:

            print(
                "ERROR:",
                type(e).__name__,
                e
            )

            summary = {
                "clip":
                    video.name,

                "status":
                    (
                        "ERROR: "
                        f"{type(e).__name__}: "
                        f"{e}"
                    )
            }

        summaries.append(
            summary
        )

    # ========================================================
    # Summary CSV
    # ========================================================

    fields = [

        "clip",
        "status",
        "fallback",

        "fps",
        "total_frames",

        "measurement_frame",
        "measurement_time_sec",

        "cst_start",
        "cst_end",

        "window_start",
        "window_end",

        "blade_out_start",
        "blade_out_end",

        "sequence_score",

        "measurement_path"
    ]

    summary_path = (
        output_dir
        / "phase_a_results.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(
            summaries
        )

    print(
        "\n================================"
    )

    print(
        "Complete"
    )

    print(
        "================================"
    )

    print(
        "Result:",
        summary_path
    )


if __name__ == "__main__":
    main()
