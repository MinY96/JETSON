"""
Phase E-1 v2
Multi-ROI Motion Signal Extractor
=================================

Input
-----
- Long CCTV MP4
- CST Motion ROI
- Blade ROI candidates x 3

Output
------
motion_signals.csv

Purpose
-------
No classification is performed here.

Signals are extracted for later supervised calibration using:

    blade_in_out_timestamp.xlsx

Blade ROI candidates:

    blade1 = (16, 267, 621, 360)
    blade2 = (309, 379, 1593, 236)
    blade3 = (74, 300, 1109, 302)
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

    # Blade ROI resize width
    blade_analysis_width: int = 480

    # CST ROI는 원본 크기 유지
    cst_blur_ksize: int = 5

    # Frame Difference
    blade_diff_pixel_threshold: int = 12

    # Optical Flow
    flow_pixel_motion_threshold: float = 0.35

    flow_pyr_scale: float = 0.5
    flow_levels: int = 2
    flow_winsize: int = 15
    flow_iterations: int = 2
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2

    frame_step: int = 1

    progress_sec: float = 10.0


# ============================================================
# Fixed ROIs
# ============================================================

CST_ROI = (
    2039,
    110,
    149,
    654,
)

BLADE_ROIS = {
    "blade1": (
        16,
        267,
        621,
        360,
    ),

    "blade2": (
        309,
        379,
        1593,
        236,
    ),

    "blade3": (
        74,
        300,
        1109,
        302,
    ),
}


# ============================================================
# Utility
# ============================================================

def resize_keep_ratio(
    image,
    target_width,
):

    h, w = image.shape[:2]

    if w <= target_width:
        return image.copy()

    scale = target_width / w

    new_h = max(
        1,
        round(h * scale),
    )

    return cv2.resize(
        image,
        (
            target_width,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )


def crop_roi(
    frame,
    roi,
):

    x, y, w, h = roi

    return frame[
        y:y + h,
        x:x + w
    ]


def prepare_cst(
    frame,
    roi,
    blur_ksize,
):

    crop = crop_roi(
        frame,
        roi,
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    if blur_ksize > 1:

        if blur_ksize % 2 == 0:
            blur_ksize += 1

        gray = cv2.GaussianBlur(
            gray,
            (
                blur_ksize,
                blur_ksize,
            ),
            0,
        )

    return gray


def prepare_blade(
    frame,
    roi,
    target_width,
):

    crop = crop_roi(
        frame,
        roi,
    )

    crop = resize_keep_ratio(
        crop,
        target_width,
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    return gray


def format_video_time(
    time_sec,
):

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
# CST Phase Correlation
# ============================================================

def calculate_phase_shift(
    prev_gray,
    curr_gray,
):

    a = prev_gray.astype(
        np.float32
    )

    b = curr_gray.astype(
        np.float32
    )

    h, w = a.shape

    window = cv2.createHanningWindow(
        (w, h),
        cv2.CV_32F,
    )

    shift, response = cv2.phaseCorrelate(
        a,
        b,
        window,
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
    prev_gray,
    curr_gray,
    threshold,
):

    diff = cv2.absdiff(
        prev_gray,
        curr_gray,
    )

    active = (
        diff >= threshold
    )

    return (
        float(active.mean()),
        float(diff.mean()),
    )


# ============================================================
# Blade Optical Flow
# ============================================================

def calculate_blade_flow(
    prev_gray,
    curr_gray,
    cfg,
):

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
        fx * fx
        +
        fy * fy
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

    h, w = prev_gray.shape

    yy, xx = np.mgrid[
        0:h,
        0:w
    ]

    cx = (
        w - 1
    ) / 2.0

    cy = (
        h - 1
    ) / 2.0

    rx = xx - cx
    ry = yy - cy

    radius = np.sqrt(
        rx * rx
        +
        ry * ry
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
        (radius > 1.0)
    )

    if np.any(radial_active):

        values = radial[
            radial_active
        ]

        radial_score = float(
            np.median(
                values
            )
        )

        radial_abs = float(
            np.median(
                np.abs(values)
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
# CSV fields
# ============================================================

def build_fields():

    fields = [
        "frame_idx",
        "time_sec",
        "video_time",

        "cst_dx",
        "cst_dy",
        "cst_phase_response",
    ]

    blade_features = [
        "motion_ratio",
        "mean_diff",

        "flow_x",
        "flow_y",

        "radial_score",
        "radial_abs",

        "active_ratio",
        "median_magnitude",
    ]

    for name in BLADE_ROIS:

        for feature in blade_features:

            fields.append(
                f"{name}_{feature}"
            )

    return fields


# ============================================================
# Main extraction
# ============================================================

def extract(
    video_path,
    output_dir,
    cfg,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "motion_signals.csv"
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open: {video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if fps <= 0:
        fps = 15.0

    total_frames = int(
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

    duration = (
        total_frames / fps
    )

    print()
    print(
        "========================================"
    )
    print(
        "Phase E-1 v2"
    )
    print(
        "========================================"
    )
    print(
        "Video      :",
        video_path,
    )
    print(
        "Resolution :",
        f"{width}x{height}",
    )
    print(
        "FPS        :",
        fps,
    )
    print(
        "Frames     :",
        total_frames,
    )
    print(
        "Duration   :",
        format_video_time(
            duration
        ),
    )
    print(
        "CST ROI    :",
        CST_ROI,
    )

    for name, roi in BLADE_ROIS.items():

        print(
            f"{name:10}:",
            roi,
        )

    print(
        "========================================"
    )

    # ========================================================
    # First frame
    # ========================================================

    ok, frame = cap.read()

    if not ok:

        raise RuntimeError(
            "Cannot read first frame"
        )

    prev_cst = prepare_cst(
        frame,
        CST_ROI,
        cfg.cst_blur_ksize,
    )

    prev_blades = {}

    for name, roi in BLADE_ROIS.items():

        prev_blades[name] = (
            prepare_blade(
                frame,
                roi,
                cfg.blade_analysis_width,
            )
        )

    # ========================================================
    # CSV
    # ========================================================

    fields = build_fields()

    start_wall = (
        time.perf_counter()
    )

    last_progress = (
        start_wall
    )

    processed = 0

    frame_idx = 0

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

        # ====================================================
        # Main Loop
        # ====================================================

        while True:

            # ------------------------------------------------
            # Read next sampled frame
            # ------------------------------------------------

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

            frame_idx += (
                cfg.frame_step
            )

            if frame_idx >= total_frames:
                break

            time_sec = (
                frame_idx / fps
            )

            # =================================================
            # CST
            # =================================================

            curr_cst = prepare_cst(
                frame,
                CST_ROI,
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

            row = {
                "frame_idx":
                    frame_idx,

                "time_sec":
                    time_sec,

                "video_time":
                    format_video_time(
                        time_sec
                    ),

                "cst_dx":
                    cst_dx,

                "cst_dy":
                    cst_dy,

                "cst_phase_response":
                    cst_response,
            }

            # =================================================
            # Blade ROIs
            # =================================================

            for (
                name,
                roi
            ) in BLADE_ROIS.items():

                curr_blade = (
                    prepare_blade(
                        frame,
                        roi,
                        cfg.blade_analysis_width,
                    )
                )

                prev_blade = (
                    prev_blades[name]
                )

                # --------------------------------------------
                # Frame difference
                # --------------------------------------------

                (
                    motion_ratio,
                    mean_diff,
                ) = calculate_frame_difference(
                    prev_blade,
                    curr_blade,
                    cfg.blade_diff_pixel_threshold,
                )

                # --------------------------------------------
                # Optical flow
                # --------------------------------------------

                flow = (
                    calculate_blade_flow(
                        prev_blade,
                        curr_blade,
                        cfg,
                    )
                )

                # --------------------------------------------
                # Save
                # --------------------------------------------

                prefix = (
                    f"{name}_"
                )

                row[
                    prefix
                    + "motion_ratio"
                ] = motion_ratio

                row[
                    prefix
                    + "mean_diff"
                ] = mean_diff

                row[
                    prefix
                    + "flow_x"
                ] = flow[
                    "flow_x"
                ]

                row[
                    prefix
                    + "flow_y"
                ] = flow[
                    "flow_y"
                ]

                row[
                    prefix
                    + "radial_score"
                ] = flow[
                    "radial_score"
                ]

                row[
                    prefix
                    + "radial_abs"
                ] = flow[
                    "radial_abs"
                ]

                row[
                    prefix
                    + "active_ratio"
                ] = flow[
                    "active_ratio"
                ]

                row[
                    prefix
                    + "median_magnitude"
                ] = flow[
                    "median_magnitude"
                ]

                prev_blades[
                    name
                ] = curr_blade

            writer.writerow(
                row
            )

            prev_cst = curr_cst

            processed += 1

            # =================================================
            # Progress
            # =================================================

            now = (
                time.perf_counter()
            )

            if (
                now - last_progress
                >= cfg.progress_sec
            ):

                elapsed = (
                    now - start_wall
                )

                pct = (
                    frame_idx
                    / total_frames
                    * 100
                )

                analysis_fps = (
                    processed
                    / max(
                        elapsed,
                        1e-6,
                    )
                )

                print(
                    f"[{pct:6.2f}%] "
                    f"{format_video_time(time_sec)} "
                    f"| "
                    f"analysis FPS="
                    f"{analysis_fps:.2f}"
                )

                csv_file.flush()

                last_progress = now

    cap.release()

    elapsed = (
        time.perf_counter()
        - start_wall
    )

    print()
    print(
        "========================================"
    )
    print(
        "Complete"
    )
    print(
        "========================================"
    )
    print(
        "CSV     :",
        csv_path,
    )
    print(
        "Elapsed :",
        f"{elapsed:.1f} sec",
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="phase_e1_v2_result",
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

    args = parser.parse_args()

    cfg = Config(
        blade_analysis_width=
            args.analysis_width,

        frame_step=
            args.frame_step,
    )

    extract(
        Path(args.video),
        Path(args.output),
        cfg,
    )


if __name__ == "__main__":
    main()
