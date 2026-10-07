"""
Phase E-1
Long Video Motion Signal Extractor
==================================

목적
----
긴 원본 CCTV 영상에서 판정을 수행하지 않고
후속 Type-2 sequence 분석에 필요한 motion signal만 추출한다.

추출 신호
---------
1. CST
   - cst_dx
   - cst_dy
   - cst_phase_response

2. Blade
   - blade_motion_ratio
   - blade_mean_diff
   - blade_flow_x
   - blade_flow_y
   - blade_radial_score
   - blade_radial_abs
   - blade_active_ratio

3. 기타
   - blade_sharpness

출력
----
output/
    motion_signals.csv
    motion_debug.mp4

중요
----
이 단계에서는 아래 판정을 하지 않는다.

- Blade IN / OUT
- CST UP / DOWN
- Type 1 / Type 2 / Type 3
- Measurement frame

실제 signal을 먼저 관찰한 후 Phase E-2에서 규칙을 정의한다.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:

    # ROI를 이 width 이하로 축소하여 분석
    analysis_width: int = 480

    # --------------------------------------------------------
    # CST Phase Correlation
    # --------------------------------------------------------

    # phase correlation 전에 Gaussian blur
    cst_blur_ksize: int = 5

    # --------------------------------------------------------
    # Blade Frame Difference
    # --------------------------------------------------------

    # pixel difference가 이 값 이상이면 active pixel
    blade_diff_pixel_threshold: int = 12

    # --------------------------------------------------------
    # Blade Optical Flow
    # --------------------------------------------------------

    # magnitude가 이 값 이상인 pixel만 radial 계산에 사용
    flow_pixel_motion_threshold: float = 0.35

    # Farneback parameters
    flow_pyr_scale: float = 0.5
    flow_levels: int = 2
    flow_winsize: int = 15
    flow_iterations: int = 2
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    # 1 = 모든 frame 분석
    # 2 = 2 frame마다 분석
    # 3 = 3 frame마다 분석
    #
    # E-1 첫 실행은 1 권장.
    frame_step: int = 1

    # --------------------------------------------------------
    # Debug video
    # --------------------------------------------------------

    save_debug_video: bool = True

    # 원본 대비 debug 영상 크기
    debug_scale: float = 0.5

    # 진행상황 출력 주기
    progress_sec: float = 10.0


# ============================================================
# Utility
# ============================================================

def resize_keep_ratio(
    image: np.ndarray,
    target_width: int,
) -> tuple[np.ndarray, float]:

    h, w = image.shape[:2]

    if w <= target_width:
        return image.copy(), 1.0

    scale = target_width / w

    new_h = max(
        1,
        round(h * scale),
    )

    resized = cv2.resize(
        image,
        (target_width, new_h),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def crop_roi(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:

    x, y, w, h = roi

    return frame[
        y:y + h,
        x:x + w
    ]


def prepare_gray(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    target_width: int,
    blur_ksize: int = 5,
) -> np.ndarray:

    crop = crop_roi(
        frame,
        roi,
    )

    small, _ = resize_keep_ratio(
        crop,
        target_width,
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY,
    )

    if blur_ksize > 1:

        if blur_ksize % 2 == 0:
            blur_ksize += 1

        gray = cv2.GaussianBlur(
            gray,
            (blur_ksize, blur_ksize),
            0,
        )

    return gray


def calc_sharpness(
    gray: np.ndarray,
) -> float:

    lap = cv2.Laplacian(
        gray,
        cv2.CV_32F,
    )

    return float(
        lap.var()
    )


def format_video_time(
    time_sec: float,
) -> str:

    total_ms = round(
        time_sec * 1000
    )

    ms = total_ms % 1000

    total_sec = (
        total_ms // 1000
    )

    sec = total_sec % 60

    total_min = (
        total_sec // 60
    )

    minute = total_min % 60

    hour = (
        total_min // 60
    )

    return (
        f"{hour:02d}:"
        f"{minute:02d}:"
        f"{sec:02d}."
        f"{ms:03d}"
    )


# ============================================================
# ROI Selection
# ============================================================

def select_roi(
    frame: np.ndarray,
    title: str,
) -> tuple[int, int, int, int]:

    h, w = frame.shape[:2]

    preview_scale = min(
        1280 / w,
        800 / h,
        1.0,
    )

    if preview_scale < 1.0:

        preview = cv2.resize(
            frame,
            None,
            fx=preview_scale,
            fy=preview_scale,
            interpolation=cv2.INTER_AREA,
        )

    else:
        preview = frame.copy()

    roi = cv2.selectROI(
        title,
        preview,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        title
    )

    x, y, rw, rh = roi

    if rw <= 0 or rh <= 0:

        raise RuntimeError(
            f"ROI was not selected: {title}"
        )

    return (
        round(x / preview_scale),
        round(y / preview_scale),
        round(rw / preview_scale),
        round(rh / preview_scale),
    )


# ============================================================
# CST Phase Correlation
# ============================================================

def calculate_phase_shift(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> tuple[float, float, float]:

    if prev_gray.shape != curr_gray.shape:

        raise ValueError(
            "Phase correlation images "
            "must have same shape."
        )

    prev_f = prev_gray.astype(
        np.float32
    )

    curr_f = curr_gray.astype(
        np.float32
    )

    height, width = (
        prev_f.shape
    )

    window = cv2.createHanningWindow(
        (width, height),
        cv2.CV_32F,
    )

    shift, response = (
        cv2.phaseCorrelate(
            prev_f,
            curr_f,
            window,
        )
    )

    dx, dy = shift

    return (
        float(dx),
        float(dy),
        float(response),
    )


# ============================================================
# Blade Frame Difference
# ============================================================

def calculate_frame_difference(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    pixel_threshold: int,
) -> tuple[float, float]:

    diff = cv2.absdiff(
        prev_gray,
        curr_gray,
    )

    active = (
        diff >= pixel_threshold
    )

    active_ratio = float(
        active.mean()
    )

    mean_diff = float(
        diff.mean()
    )

    return (
        active_ratio,
        mean_diff,
    )


# ============================================================
# Blade Optical Flow
# ============================================================

def calculate_blade_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    cfg: Config,
) -> dict:

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        cfg.flow_pyr_scale,
        cfg.flow_levels,
        cfg.flow_winsize,
        cfg.flow_iterations,
        cfg.flow_poly_n,
        cfg.flow_poly_sigma,
        0,
    )

    fx = flow[..., 0]
    fy = flow[..., 1]

    magnitude = np.sqrt(
        fx * fx + fy * fy
    )

    active = (
        magnitude
        >= cfg.flow_pixel_motion_threshold
    )

    active_ratio = float(
        active.mean()
    )

    if not np.any(active):

        return {
            "flow_x": 0.0,
            "flow_y": 0.0,
            "radial_score": 0.0,
            "radial_abs": 0.0,
            "active_ratio": 0.0,
            "median_magnitude": 0.0,
        }

    # --------------------------------------------------------
    # Median X / Y flow
    # --------------------------------------------------------

    flow_x = float(
        np.median(
            fx[active]
        )
    )

    flow_y = float(
        np.median(
            fy[active]
        )
    )

    median_magnitude = float(
        np.median(
            magnitude[active]
        )
    )

    # --------------------------------------------------------
    # Radial Flow
    #
    # ROI 중심으로부터 바깥 방향 = positive
    # ROI 중심으로 안쪽 방향 = negative
    # --------------------------------------------------------

    height, width = (
        prev_gray.shape
    )

    yy, xx = np.mgrid[
        0:height,
        0:width
    ]

    cx = (
        width - 1
    ) / 2.0

    cy = (
        height - 1
    ) / 2.0

    rx = xx - cx
    ry = yy - cy

    radius = np.sqrt(
        rx * rx + ry * ry
    )

    valid_radius = (
        radius > 1.0
    )

    denom = np.maximum(
        radius,
        1e-6,
    )

    radial_x = (
        rx / denom
    )

    radial_y = (
        ry / denom
    )

    radial = (
        fx * radial_x
        +
        fy * radial_y
    )

    radial_active = (
        active
        &
        valid_radius
    )

    if np.any(radial_active):

        radial_values = (
            radial[
                radial_active
            ]
        )

        radial_score = float(
            np.median(
                radial_values
            )
        )

        radial_abs = float(
            np.median(
                np.abs(
                    radial_values
                )
            )
        )

    else:

        radial_score = 0.0
        radial_abs = 0.0

    return {
        "flow_x":
            flow_x,

        "flow_y":
            flow_y,

        "radial_score":
            radial_score,

        "radial_abs":
            radial_abs,

        "active_ratio":
            active_ratio,

        "median_magnitude":
            median_magnitude,
    }


# ============================================================
# Debug Video
# ============================================================

def draw_debug_frame(
    frame: np.ndarray,
    frame_idx: int,
    time_sec: float,
    cst_roi: tuple[int, int, int, int],
    blade_roi: tuple[int, int, int, int],
    record: dict,
) -> np.ndarray:

    vis = frame.copy()

    # --------------------------------------------------------
    # CST ROI
    # --------------------------------------------------------

    x, y, w, h = cst_roi

    cv2.rectangle(
        vis,
        (x, y),
        (x + w, y + h),
        (255, 255, 0),
        2,
    )

    cv2.putText(
        vis,
        "CST MOTION ROI",
        (
            x,
            max(25, y - 8),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    # --------------------------------------------------------
    # Blade ROI
    # --------------------------------------------------------

    x, y, w, h = blade_roi

    cv2.rectangle(
        vis,
        (x, y),
        (x + w, y + h),
        (0, 255, 255),
        2,
    )

    cv2.putText(
        vis,
        "BLADE MOTION ROI",
        (
            x,
            max(25, y - 8),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # --------------------------------------------------------
    # Information panel
    # --------------------------------------------------------

    panel_x1 = 15
    panel_y1 = 15

    panel_x2 = min(
        vis.shape[1] - 15,
        650,
    )

    panel_y2 = 275

    overlay = vis.copy()

    cv2.rectangle(
        overlay,
        (
            panel_x1,
            panel_y1,
        ),
        (
            panel_x2,
            panel_y2,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.72,
        vis,
        0.28,
        0,
        vis,
    )

    lines = [
        (
            f"Frame : {frame_idx}"
        ),
        (
            "Time  : "
            f"{format_video_time(time_sec)}"
        ),
        (
            "CST dx/dy : "
            f"{record['cst_dx']:+.4f} / "
            f"{record['cst_dy']:+.4f}"
        ),
        (
            "CST response : "
            f"{record['cst_phase_response']:.4f}"
        ),
        (
            "Blade diff ratio : "
            f"{record['blade_motion_ratio']:.5f}"
        ),
        (
            "Blade flow x/y : "
            f"{record['blade_flow_x']:+.4f} / "
            f"{record['blade_flow_y']:+.4f}"
        ),
        (
            "Blade radial : "
            f"{record['blade_radial_score']:+.4f}"
        ),
        (
            "Blade active : "
            f"{record['blade_active_ratio']:.5f}"
        ),
        (
            "Blade magnitude : "
            f"{record['blade_median_magnitude']:.4f}"
        ),
    ]

    for i, text in enumerate(
        lines
    ):

        cv2.putText(
            vis,
            text,
            (
                30,
                43 + i * 26,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis


# ============================================================
# Main Extraction
# ============================================================

def extract_motion_signals(
    video_path: Path,
    output_dir: Path,
    cst_roi: tuple[int, int, int, int],
    blade_roi: tuple[int, int, int, int],
    cfg: Config,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "motion_signals.csv"
    )

    debug_video_path = (
        output_dir
        / "motion_debug.mp4"
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

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

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

    duration_sec = (
        total_frames / fps
        if total_frames > 0
        else 0.0
    )

    print()
    print(
        "=========================================="
    )
    print(
        "Phase E-1 Motion Signal Extraction"
    )
    print(
        "=========================================="
    )
    print(
        "Video       :",
        video_path,
    )
    print(
        "Resolution  :",
        f"{frame_width}x{frame_height}",
    )
    print(
        "FPS         :",
        f"{fps:.3f}",
    )
    print(
        "Frames      :",
        total_frames,
    )
    print(
        "Duration    :",
        format_video_time(
            duration_sec
        ),
    )
    print(
        "CST ROI     :",
        cst_roi,
    )
    print(
        "Blade ROI   :",
        blade_roi,
    )
    print(
        "Frame step  :",
        cfg.frame_step,
    )
    print(
        "=========================================="
    )
    print()

    # ========================================================
    # Read first frame
    # ========================================================

    ok, first_frame = cap.read()

    if not ok:

        cap.release()

        raise RuntimeError(
            "Cannot read first frame."
        )

    prev_cst = prepare_gray(
        first_frame,
        cst_roi,
        cfg.analysis_width,
        cfg.cst_blur_ksize,
    )

    prev_blade = prepare_gray(
        first_frame,
        blade_roi,
        cfg.analysis_width,
        5,
    )

    prev_processed_frame_idx = 0

    # ========================================================
    # Debug video
    # ========================================================

    debug_writer = None

    if cfg.save_debug_video:

        debug_width = max(
            1,
            round(
                frame_width
                * cfg.debug_scale
            ),
        )

        debug_height = max(
            1,
            round(
                frame_height
                * cfg.debug_scale
            ),
        )

        # frame_step을 적용하므로
        # debug 영상 FPS도 동일 비율로 감소
        debug_fps = (
            fps / cfg.frame_step
        )

        debug_writer = (
            cv2.VideoWriter(
                str(
                    debug_video_path
                ),
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                ),
                debug_fps,
                (
                    debug_width,
                    debug_height,
                ),
            )
        )

    # ========================================================
    # CSV
    # ========================================================

    fields = [
        "frame_idx",
        "time_sec",
        "video_time",

        "frame_delta",
        "delta_sec",

        # CST
        "cst_dx",
        "cst_dy",
        "cst_phase_response",

        # Blade frame difference
        "blade_motion_ratio",
        "blade_mean_diff",

        # Blade optical flow
        "blade_flow_x",
        "blade_flow_y",
        "blade_radial_score",
        "blade_radial_abs",
        "blade_active_ratio",
        "blade_median_magnitude",

        # Image quality
        "blade_sharpness",
    ]

    start_wall_time = (
        time.perf_counter()
    )

    last_progress_time = (
        start_wall_time
    )

    processed_count = 0

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
        )

        writer.writeheader()

        # ----------------------------------------------------
        # First frame
        # ----------------------------------------------------

        first_record = {
            "frame_idx": 0,
            "time_sec": 0.0,
            "video_time":
                format_video_time(0),

            "frame_delta": 0,
            "delta_sec": 0.0,

            "cst_dx": 0.0,
            "cst_dy": 0.0,
            "cst_phase_response": 0.0,

            "blade_motion_ratio": 0.0,
            "blade_mean_diff": 0.0,

            "blade_flow_x": 0.0,
            "blade_flow_y": 0.0,
            "blade_radial_score": 0.0,
            "blade_radial_abs": 0.0,
            "blade_active_ratio": 0.0,
            "blade_median_magnitude": 0.0,

            "blade_sharpness":
                calc_sharpness(
                    prev_blade
                ),
        }

        writer.writerow(
            first_record
        )

        if debug_writer is not None:

            debug_frame = (
                draw_debug_frame(
                    first_frame,
                    0,
                    0.0,
                    cst_roi,
                    blade_roi,
                    first_record,
                )
            )

            debug_frame = cv2.resize(
                debug_frame,
                (
                    debug_width,
                    debug_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

            debug_writer.write(
                debug_frame
            )

        # ====================================================
        # Main loop
        # ====================================================

        frame_idx = 0

        while True:

            # ------------------------------------------------
            # frame_step
            # ------------------------------------------------

            target_frame = (
                frame_idx
                + cfg.frame_step
            )

            frame = None

            success = True

            for _ in range(
                cfg.frame_step
            ):

                ok = cap.grab()

                if not ok:
                    success = False
                    break

            if not success:
                break

            ok, frame = cap.retrieve()

            if not ok:
                break

            frame_idx = target_frame

            if (
                total_frames > 0
                and
                frame_idx >= total_frames
            ):
                break

            # ------------------------------------------------
            # Time
            # ------------------------------------------------

            time_sec = (
                frame_idx / fps
            )

            frame_delta = (
                frame_idx
                - prev_processed_frame_idx
            )

            delta_sec = (
                frame_delta / fps
            )

            # ------------------------------------------------
            # CST
            # ------------------------------------------------

            curr_cst = prepare_gray(
                frame,
                cst_roi,
                cfg.analysis_width,
                cfg.cst_blur_ksize,
            )

            (
                cst_dx,
                cst_dy,
                cst_response,
            ) = calculate_phase_shift(
                prev_cst,
                curr_cst,
            )

            # ------------------------------------------------
            # Blade
            # ------------------------------------------------

            curr_blade = prepare_gray(
                frame,
                blade_roi,
                cfg.analysis_width,
                5,
            )

            (
                blade_motion_ratio,
                blade_mean_diff,
            ) = calculate_frame_difference(
                prev_blade,
                curr_blade,
                cfg.blade_diff_pixel_threshold,
            )

            blade_flow = (
                calculate_blade_flow(
                    prev_blade,
                    curr_blade,
                    cfg,
                )
            )

            blade_sharpness = (
                calc_sharpness(
                    curr_blade
                )
            )

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            record = {
                "frame_idx":
                    frame_idx,

                "time_sec":
                    time_sec,

                "video_time":
                    format_video_time(
                        time_sec
                    ),

                "frame_delta":
                    frame_delta,

                "delta_sec":
                    delta_sec,

                # CST
                "cst_dx":
                    cst_dx,

                "cst_dy":
                    cst_dy,

                "cst_phase_response":
                    cst_response,

                # Blade difference
                "blade_motion_ratio":
                    blade_motion_ratio,

                "blade_mean_diff":
                    blade_mean_diff,

                # Blade flow
                "blade_flow_x":
                    blade_flow[
                        "flow_x"
                    ],

                "blade_flow_y":
                    blade_flow[
                        "flow_y"
                    ],

                "blade_radial_score":
                    blade_flow[
                        "radial_score"
                    ],

                "blade_radial_abs":
                    blade_flow[
                        "radial_abs"
                    ],

                "blade_active_ratio":
                    blade_flow[
                        "active_ratio"
                    ],

                "blade_median_magnitude":
                    blade_flow[
                        "median_magnitude"
                    ],

                # quality
                "blade_sharpness":
                    blade_sharpness,
            }

            writer.writerow(
                record
            )

            processed_count += 1

            # ------------------------------------------------
            # Debug video
            # ------------------------------------------------

            if debug_writer is not None:

                debug_frame = (
                    draw_debug_frame(
                        frame,
                        frame_idx,
                        time_sec,
                        cst_roi,
                        blade_roi,
                        record,
                    )
                )

                debug_frame = cv2.resize(
                    debug_frame,
                    (
                        debug_width,
                        debug_height,
                    ),
                    interpolation=
                        cv2.INTER_AREA,
                )

                debug_writer.write(
                    debug_frame
                )

            # ------------------------------------------------
            # Previous
            # ------------------------------------------------

            prev_cst = curr_cst
            prev_blade = curr_blade

            prev_processed_frame_idx = (
                frame_idx
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            now = time.perf_counter()

            if (
                now - last_progress_time
                >= cfg.progress_sec
            ):

                elapsed = (
                    now - start_wall_time
                )

                video_progress = (
                    frame_idx / fps
                )

                if duration_sec > 0:

                    progress_pct = (
                        video_progress
                        / duration_sec
                        * 100
                    )

                else:

                    progress_pct = 0.0

                processing_fps = (
                    processed_count
                    / max(
                        elapsed,
                        1e-6,
                    )
                )

                print(
                    f"[{progress_pct:6.2f}%] "
                    f"{format_video_time(video_progress)} "
                    f"/ "
                    f"{format_video_time(duration_sec)} "
                    f"| analysis FPS="
                    f"{processing_fps:.1f}"
                )

                # 긴 영상에서 중간 결과가
                # 디스크에 실제 반영되도록 flush
                csv_file.flush()

                last_progress_time = (
                    now
                )

    # ========================================================
    # Finish
    # ========================================================

    cap.release()

    if debug_writer is not None:
        debug_writer.release()

    elapsed = (
        time.perf_counter()
        - start_wall_time
    )

    print()
    print(
        "=========================================="
    )
    print(
        "Phase E-1 Complete"
    )
    print(
        "=========================================="
    )
    print(
        "CSV        :",
        csv_path,
    )

    if cfg.save_debug_video:

        print(
            "Debug MP4  :",
            debug_video_path,
        )

    print(
        "Elapsed    :",
        f"{elapsed:.1f} sec",
    )

    print(
        "=========================================="
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Original long CCTV mp4",
    )

    parser.add_argument(
        "--output",
        default="phase_e1_result",
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
            "H",
        ),
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
            "H",
        ),
    )

    parser.add_argument(
        "--analysis-width",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--debug-scale",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--no-debug-video",
        action="store_true",
    )

    args = parser.parse_args()

    video_path = Path(
        args.video
    )

    output_dir = Path(
        args.output
    )

    if not video_path.exists():

        raise FileNotFoundError(
            video_path
        )

    if args.frame_step < 1:

        raise ValueError(
            "--frame-step must be >= 1"
        )

    cfg = Config(
        analysis_width=
            args.analysis_width,

        frame_step=
            args.frame_step,

        debug_scale=
            args.debug_scale,

        save_debug_video=
            not args.no_debug_video,
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
            str(video_path)
        )

        ok, frame = cap.read()

        cap.release()

        if not ok:

            raise RuntimeError(
                "Cannot read first frame."
            )

        if cst_roi is None:

            print()
            print(
                "Select CST Motion ROI"
            )

            cst_roi = select_roi(
                frame,
                "Select CST Motion ROI",
            )

            print(
                "CST ROI:",
                cst_roi,
            )

        if blade_roi is None:

            print()
            print(
                "Select Blade Motion ROI"
            )

            blade_roi = select_roi(
                frame,
                "Select Blade Motion ROI",
            )

            print(
                "Blade ROI:",
                blade_roi,
            )

    extract_motion_signals(
        video_path,
        output_dir,
        cst_roi,
        blade_roi,
        cfg,
    )


if __name__ == "__main__":
    main()
