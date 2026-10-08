
"""
Phase B v5
==========

Dual Reference Template Matching
+ Full Measurement ROI Horizontal Wafer Detection

Detection:
    1. Normal Reference + Gray
    2. Dark Reference   + Gray
    3. Normal Reference + CLAHE
    4. Dark Reference   + CLAHE

Measurement:
    - Template matching -> Blade location
    - X ratio -> Blade measurement anchor
    - Entire Measurement ROI -> Wafer horizontal edges
    - Upper / lower wafer independently detected
    - Missing wafer -> SKIP
    - No slot-number assumptions

Output:
    geometry_results.csv
    overlay/
    detail/
    debug/
        *_matching.csv
        *_profile.csv
        *_candidates.csv
        *_sobel.png
        *_horizontal_lines.png

Requires:
    opencv-python
    numpy
    pandas
"""

from __future__ import annotations

import argparse
import traceback

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

    # Original image coordinates: x, y, width, height
    search_roi: tuple = (41, 358, 2027, 186)
    measure_roi: tuple = (49, 296, 600, 273)

    # --------------------------------------------------------
    # Matching thresholds
    # --------------------------------------------------------

    normal_gray_threshold: float = 0.65
    dark_gray_threshold: float = 0.65

    normal_clahe_threshold: float = 0.65
    dark_clahe_threshold: float = 0.65

    review_threshold: float = 0.30

    # CLAHE
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    # Optional position validation
    expected_match_x: float | None = None
    expected_match_y: float | None = None
    max_position_error: float = 30.0

    # --------------------------------------------------------
    # Normal template geometry
    # --------------------------------------------------------

    normal_x_ratio: float = 0.7

    normal_top_ratio: float = 0.0
    normal_bottom_ratio: float = 1.0

    normal_top_offset: float = 0.0
    normal_bottom_offset: float = 0.0

    # --------------------------------------------------------
    # Dark template geometry
    # --------------------------------------------------------

    dark_x_ratio: float = 0.7

    dark_top_ratio: float = 0.0
    dark_bottom_ratio: float = 1.0

    dark_top_offset: float = 0.0
    dark_bottom_offset: float = 0.0

    # --------------------------------------------------------
    # Wafer detection
    # --------------------------------------------------------

    blur_ksize: int = 3
    profile_smooth: int = 3

    # Sobel-Y strength
    wafer_edge_threshold: float = 10.0

    # Local peak prominence
    wafer_min_prominence: float = 4.0

    # Pixel-level horizontal edge threshold
    wafer_support_gradient: float = 15.0

    # Minimum valid fraction across entire ROI
    wafer_min_support: float = 0.20

    # Minimum span across X, regardless of occlusion
    wafer_min_span_ratio: float = 0.35

    # At least this many separate X strips must detect edge
    wafer_min_valid_strips: int = 3

    # Horizontal strip analysis
    strip_width: int = 24
    strip_step: int = 12

    # Same horizontal edge can shift by a few pixels
    line_y_tolerance: int = 3

    # Duplicate peaks
    edge_min_distance: int = 3

    # Gap constraints
    min_gap: float = 5.0
    max_gap: float = 100.0

    # Prevent picking Blade internal line
    blade_exclusion_px: float = 4.0

    # Output
    jpeg_quality: int = 95


# ============================================================
# Reference
# ============================================================

@dataclass
class Reference:
    name: str
    image: np.ndarray

    x_ratio: float

    top_ratio: float
    bottom_ratio: float

    top_offset: float
    bottom_offset: float


# ============================================================
# Utilities
# ============================================================

def odd(value):
    value = max(1, int(value))
    return value if value % 2 else value + 1


def crop_roi(image, roi):
    x, y, w, h = map(int, roi)
    ih, iw = image.shape[:2]

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid ROI: {roi}")

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(iw, x + w)
    y2 = min(ih, y + h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"ROI {roi} outside image {iw}x{ih}"
        )

    return (
        image[y1:y2, x1:x2],
        (x1, y1, x2, y2),
    )


def smooth_1d(values, kernel):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if kernel <= 1:
        return values.copy()

    return cv2.GaussianBlur(
        values.reshape(-1, 1),
        (1, odd(kernel)),
        0,
    ).ravel()


def fmt(value, digits=2):
    if value is None:
        return "SKIP"

    return f"{value:.{digits}f}"


# ============================================================
# Load references
# ============================================================

def load_references(normal_path, dark_path, cfg):

    normal = cv2.imread(str(normal_path))
    dark = cv2.imread(str(dark_path))

    if normal is None:
        raise FileNotFoundError(normal_path)

    if dark is None:
        raise FileNotFoundError(dark_path)

    return {
        "normal": Reference(
            name="normal",
            image=normal,

            x_ratio=cfg.normal_x_ratio,

            top_ratio=cfg.normal_top_ratio,
            bottom_ratio=cfg.normal_bottom_ratio,

            top_offset=cfg.normal_top_offset,
            bottom_offset=cfg.normal_bottom_offset,
        ),

        "dark": Reference(
            name="dark",
            image=dark,

            x_ratio=cfg.dark_x_ratio,

            top_ratio=cfg.dark_top_ratio,
            bottom_ratio=cfg.dark_bottom_ratio,

            top_offset=cfg.dark_top_offset,
            bottom_offset=cfg.dark_bottom_offset,
        ),
    }


# ============================================================
# Matching preprocessing
# ============================================================

def preprocess_matching(image, method, cfg):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    if method == "gray":

        return cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

    if method == "clahe":

        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip,
            tileGridSize=(
                cfg.clahe_grid,
                cfg.clahe_grid,
            ),
        )

        enhanced = clahe.apply(gray)

        return cv2.GaussianBlur(
            enhanced,
            (3, 3),
            0,
        )

    raise ValueError(method)


# ============================================================
# 1. Template matching
# ============================================================

def match_template(
    search_gray,
    origin,
    reference,
    method,
    cfg,
):

    target = preprocess_matching(
        reference.image,
        method,
        cfg,
    )

    th, tw = target.shape
    sh, sw = search_gray.shape

    if tw > sw or th > sh:

        return {
            "reference": reference.name,
            "method": method,
            "score": None,
            "x": None,
            "y": None,
            "w": tw,
            "h": th,
            "position_error": None,
            "position_valid": False,
        }

    if float(np.std(target)) < 1e-5:
        raise ValueError(
            f"Low contrast template: {reference.name}"
        )

    response = cv2.matchTemplate(
        search_gray,
        target,
        cv2.TM_CCOEFF_NORMED,
    )

    response = np.nan_to_num(
        response,
        nan=-1.0,
        posinf=-1.0,
        neginf=-1.0,
    )

    _, score, _, loc = cv2.minMaxLoc(response)

    x = int(origin[0] + loc[0])
    y = int(origin[1] + loc[1])

    position_error = None
    position_valid = True

    if (
        cfg.expected_match_x is not None
        and cfg.expected_match_y is not None
    ):

        position_error = float(np.hypot(
            x - cfg.expected_match_x,
            y - cfg.expected_match_y,
        ))

        position_valid = (
            position_error <= cfg.max_position_error
        )

    return {
        "reference": reference.name,
        "method": method,
        "score": float(score),

        "x": x,
        "y": y,
        "w": tw,
        "h": th,

        "position_error": position_error,
        "position_valid": position_valid,
    }


def select_blade_match(image, references, cfg):

    search, bounds = crop_roi(
        image,
        cfg.search_roi,
    )

    origin = (bounds[0], bounds[1])

    stages = [
        (
            "normal",
            "gray",
            cfg.normal_gray_threshold,
        ),
        (
            "dark",
            "gray",
            cfg.dark_gray_threshold,
        ),
        (
            "normal",
            "clahe",
            cfg.normal_clahe_threshold,
        ),
        (
            "dark",
            "clahe",
            cfg.dark_clahe_threshold,
        ),
    ]

    search_cache = {}
    attempts = []
    selected = None

    for stage, (name, method, threshold) in enumerate(
        stages,
        start=1,
    ):

        if method not in search_cache:

            search_cache[method] = preprocess_matching(
                search,
                method,
                cfg,
            )

        candidate = match_template(
            search_cache[method],
            origin,
            references[name],
            method,
            cfg,
        )

        score = candidate["score"]

        passed = (
            score is not None
            and score >= threshold
            and candidate["position_valid"]
        )

        candidate.update({
            "stage": stage,
            "threshold": threshold,
            "passed": bool(passed),
        })

        attempts.append(candidate)

        if passed:
            selected = candidate.copy()
            selected["status"] = "matched"
            break

    if selected is None:

        valid = [
            item for item in attempts
            if item["score"] is not None
            and item["position_valid"]
        ]

        if valid:

            best = max(
                valid,
                key=lambda item: item["score"],
            )

            selected = best.copy()

            selected["status"] = (
                "review"
                if best["score"] >= cfg.review_threshold
                else "match_failed"
            )

        else:

            selected = {
                "reference": None,
                "method": None,
                "stage": None,
                "score": None,

                "x": None,
                "y": None,
                "w": None,
                "h": None,

                "position_error": None,
                "status": "match_failed",
            }

    return selected, attempts


# ============================================================
# 2. Blade anchors
# ============================================================

def calculate_blade_anchors(match, references):

    ref = references[match["reference"]]

    measure_x = (
        match["x"]
        + ref.x_ratio * match["w"]
    )

    blade_top_y = (
        match["y"]
        + ref.top_ratio * match["h"]
        + ref.top_offset
    )

    blade_bottom_y = (
        match["y"]
        + ref.bottom_ratio * match["h"]
        + ref.bottom_offset
    )

    if blade_top_y >= blade_bottom_y:
        raise ValueError(
            "Invalid Blade geometry: "
            "top_y must be below bottom_y in image coordinates"
        )

    return {
        "x": float(measure_x),
        "top_y": float(blade_top_y),
        "bottom_y": float(blade_bottom_y),
        "thickness": float(
            blade_bottom_y - blade_top_y
        ),
    }


# ============================================================
# 3. Full Measurement ROI
# ============================================================

def get_measurement_roi(image, anchors, cfg):

    roi, bounds = crop_roi(
        image,
        cfg.measure_roi,
    )

    x1, y1, x2, y2 = bounds

    # Only the measurement anchor must be inside ROI.
    # Wafer edges are searched throughout entire ROI.
    if not (x1 <= anchors["x"] < x2):
        raise ValueError(
            f"Blade measure_x={anchors['x']:.1f} "
            f"outside Measurement ROI X=[{x1},{x2})"
        )

    if not (
        y1 <= anchors["top_y"]
        < anchors["bottom_y"]
        < y2
    ):
        raise ValueError(
            "Blade upper/lower anchor outside measurement ROI"
        )

    return roi, bounds


# ============================================================
# 4. Full ROI wafer Sobel profile
# ============================================================

def calculate_wafer_profile(roi, cfg):

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (odd(cfg.blur_ksize),) * 2,
        0,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0, 1,
        ksize=3,
    )

    abs_gy = np.abs(gy)

    # Clamp strong reflections to limit outlier dominance.
    clipped = np.minimum(abs_gy, 120.0)

    edge_profile = smooth_1d(
        np.mean(clipped, axis=1),
        cfg.profile_smooth,
    )

    support = smooth_1d(
        np.mean(
            abs_gy >= cfg.wafer_support_gradient,
            axis=1,
        ),
        cfg.profile_smooth,
    )

    signed_profile = smooth_1d(
        np.mean(gy, axis=1),
        cfg.profile_smooth,
    )

    return {
        "gray": gray,
        "gy": gy,
        "edge": edge_profile,
        "support": support,
        "signed": signed_profile,
    }


# ============================================================
# 5. Horizontal line support
# ============================================================

def calculate_horizontal_support(
    gy,
    y,
    cfg,
):
    """
    Measure evidence for one horizontal edge.

    X-axis is divided into strips. Each strip can support
    the line even when another strip is occluded or glared.

    Returns:
        support_ratio
        span_ratio
        valid_strips
        observed_x_min
        observed_x_max
    """

    height, width = gy.shape

    half_y = cfg.line_y_tolerance

    ya = max(0, y - half_y)
    yb = min(height, y + half_y + 1)

    abs_edge = np.abs(gy[ya:yb])

    strip_width = min(cfg.strip_width, width)
    strip_step = cfg.strip_step

    positions = list(range(
        0,
        max(1, width - strip_width + 1),
        strip_step,
    ))

    last = max(0, width - strip_width)

    if last not in positions:
        positions.append(last)

    positions = sorted(set(positions))

    valid_x = []

    valid_count = 0

    for x1 in positions:
        x2 = min(width, x1 + strip_width)

        region = abs_edge[:, x1:x2]

        # For each X pixel, check whether a gradient is
        # present near the candidate horizontal Y.
        column_hit = np.any(
            region >= cfg.wafer_support_gradient,
            axis=0,
        )

        ratio = float(np.mean(column_hit))

        if ratio >= cfg.wafer_min_support:
            valid_count += 1

            valid_x.append(
                (
                    x1 + x2 - 1
                ) / 2.0
            )

    total = len(positions)

    support_ratio = (
        valid_count / total
        if total else 0.0
    )

    if valid_x:
        xmin = float(min(valid_x))
        xmax = float(max(valid_x))

        span_ratio = (
            (xmax - xmin + strip_width)
            / max(1, width)
        )

        span_ratio = min(1.0, span_ratio)
    else:
        xmin = None
        xmax = None
        span_ratio = 0.0

    return {
        "support_ratio": support_ratio,
        "span_ratio": span_ratio,
        "valid_strips": valid_count,
        "total_strips": total,
        "observed_x_min": xmin,
        "observed_x_max": xmax,
    }


# ============================================================
# 6. Wafer line candidates
# ============================================================

def detect_wafer_lines(profile, bounds, cfg):

    edge = profile["edge"]
    support = profile["support"]
    gy = profile["gy"]

    x_offset = bounds[0]
    y_offset = bounds[1]

    height = len(edge)

    candidates = []

    for y in range(2, height - 2):

        strength = float(edge[y])

        if strength < cfg.wafer_edge_threshold:
            continue

        # Local maximum
        if not (
            strength >= edge[y - 1]
            and strength > edge[y + 1]
        ):
            continue

        local_min = float(
            np.min(edge[y - 2:y + 3])
        )

        prominence = strength - local_min

        if prominence < cfg.wafer_min_prominence:
            continue

        horizontal = calculate_horizontal_support(
            gy,
            y,
            cfg,
        )

        if (
            horizontal["valid_strips"]
            < cfg.wafer_min_valid_strips
        ):
            continue

        if (
            horizontal["support_ratio"]
            < cfg.wafer_min_support
        ):
            continue

        if (
            horizontal["span_ratio"]
            < cfg.wafer_min_span_ratio
        ):
            continue

        candidates.append({
            "y_local": float(y),
            "y_global": float(y + y_offset),

            "strength": strength,
            "prominence": prominence,

            "row_support": float(support[y]),

            "horizontal_support":
                horizontal["support_ratio"],

            "span_ratio":
                horizontal["span_ratio"],

            "valid_strips":
                horizontal["valid_strips"],

            "observed_x_min": (
                None
                if horizontal["observed_x_min"] is None
                else x_offset
                + horizontal["observed_x_min"]
            ),

            "observed_x_max": (
                None
                if horizontal["observed_x_max"] is None
                else x_offset
                + horizontal["observed_x_max"]
            ),
        })

    # Remove duplicate peaks.
    candidates.sort(
        key=lambda c: c["strength"],
        reverse=True,
    )

    filtered = []

    for candidate in candidates:

        duplicated = any(
            abs(
                candidate["y_global"]
                - prev["y_global"]
            )
            < cfg.edge_min_distance
            for prev in filtered
        )

        if not duplicated:
            filtered.append(candidate)

    filtered.sort(
        key=lambda c: c["y_global"]
    )

    return filtered


# ============================================================
# 7. Find adjacent wafer
# ============================================================

def find_nearest_wafer(
    candidates,
    blade_y,
    side,
    cfg,
):

    valid = []

    for candidate in candidates:

        wafer_y = candidate["y_global"]

        if side == "upper":
            gap = blade_y - wafer_y
        else:
            gap = wafer_y - blade_y

        minimum_gap = max(
            cfg.min_gap,
            cfg.blade_exclusion_px,
        )

        if not (
            minimum_gap <= gap <= cfg.max_gap
        ):
            continue

        valid.append({
            **candidate,
            "gap_px": float(gap),
        })

    if not valid:
        return None, "not_detected", []

    # Closest horizontal line from the Blade boundary.
    valid.sort(
        key=lambda c: c["gap_px"]
    )

    selected = valid[0]

    return selected, "detected", valid


# ============================================================
# 8. Geometry
# ============================================================

def analyze_image(image, references, cfg):

    match, attempts = select_blade_match(
        image,
        references,
        cfg,
    )

    result = {
        "status": match["status"],
        "match": match,
        "anchors": None,

        "upper": None,
        "lower": None,

        "upper_status": "not_evaluated",
        "lower_status": "not_evaluated",

        "top_gap_px": None,
        "bottom_gap_px": None,

        "top_ratio": None,
        "bottom_ratio": None,
    }

    debug = {
        "attempts": attempts,
    }

    # No automatic measurement for low-confidence matching.
    if match["status"] != "matched":
        return result, debug

    anchors = calculate_blade_anchors(
        match,
        references,
    )

    result["anchors"] = anchors

    roi, bounds = get_measurement_roi(
        image,
        anchors,
        cfg,
    )

    profile = calculate_wafer_profile(
        roi,
        cfg,
    )

    candidates = detect_wafer_lines(
        profile,
        bounds,
        cfg,
    )

    upper, upper_status, upper_candidates = (
        find_nearest_wafer(
            candidates,
            anchors["top_y"],
            "upper",
            cfg,
        )
    )

    lower, lower_status, lower_candidates = (
        find_nearest_wafer(
            candidates,
            anchors["bottom_y"],
            "lower",
            cfg,
        )
    )

    top_gap = upper["gap_px"] if upper else None
    bottom_gap = lower["gap_px"] if lower else None

    top_ratio = None
    bottom_ratio = None

    if top_gap is not None and bottom_gap is not None:

        total = top_gap + bottom_gap

        if total > 0:
            top_ratio = top_gap / total * 100.0
            bottom_ratio = bottom_gap / total * 100.0

    if upper and lower:
        status = "ok"
    elif upper or lower:
        status = "partial"
    else:
        status = "no_wafer_detected"

    result.update({
        "status": status,

        "upper": upper,
        "lower": lower,

        "upper_status": upper_status,
        "lower_status": lower_status,

        "top_gap_px": top_gap,
        "bottom_gap_px": bottom_gap,

        "top_ratio": top_ratio,
        "bottom_ratio": bottom_ratio,
    })

    debug.update({
        "bounds": bounds,
        "profile": profile,
        "candidates": candidates,

        "upper_candidates": upper_candidates,
        "lower_candidates": lower_candidates,
    })

    return result, debug


# ============================================================
# 9. Overlay
# ============================================================

def draw_overlay(image, result, debug, cfg):

    output = image.copy()

    sx, sy, sw, sh = cfg.search_roi
    mx, my, mw, mh = cfg.measure_roi

    # Search ROI
    cv2.rectangle(
        output,
        (sx, sy),
        (sx + sw, sy + sh),
        (255, 140, 0),
        2,
    )

    # Measurement ROI
    cv2.rectangle(
        output,
        (mx, my),
        (mx + mw, my + mh),
        (0, 255, 255),
        2,
    )

    match = result["match"]

    if match["x"] is not None:

        color = (
            (0, 255, 0)
            if match["status"] == "matched"
            else (0, 0, 255)
        )

        cv2.rectangle(
            output,
            (match["x"], match["y"]),
            (
                match["x"] + match["w"],
                match["y"] + match["h"],
            ),
            color,
            2,
        )

    anchors = result["anchors"]

    if anchors is None:
        return output

    x1, _, x2, _ = debug["bounds"]

    measure_x = int(round(anchors["x"]))
    blade_top = int(round(anchors["top_y"]))
    blade_bottom = int(round(anchors["bottom_y"]))

    # Blade boundary
    cv2.line(
        output,
        (x1, blade_top),
        (x2, blade_top),
        (255, 0, 0),
        2,
    )

    cv2.line(
        output,
        (x1, blade_bottom),
        (x2, blade_bottom),
        (255, 0, 0),
        2,
    )

    # Measurement X marker
    cv2.line(
        output,
        (measure_x, my),
        (measure_x, my + mh),
        (255, 255, 0),
        1,
    )

    # Upper wafer
    if result["upper"] is not None:

        yy = int(round(
            result["upper"]["y_global"]
        ))

        cv2.line(
            output,
            (x1, yy),
            (x2, yy),
            (0, 255, 0),
            2,
        )

        cv2.line(
            output,
            (measure_x, yy),
            (measure_x, blade_top),
            (0, 255, 0),
            2,
        )

    # Lower wafer
    if result["lower"] is not None:

        yy = int(round(
            result["lower"]["y_global"]
        ))

        cv2.line(
            output,
            (x1, yy),
            (x2, yy),
            (0, 165, 255),
            2,
        )

        cv2.line(
            output,
            (measure_x, blade_bottom),
            (measure_x, yy),
            (0, 165, 255),
            2,
        )

    return output


# ============================================================
# 10. Detail image
# ============================================================

def create_detail(overlay, result, cfg):

    roi, _ = crop_roi(
        overlay,
        cfg.measure_roi,
    )

    zoom = cv2.resize(
        roi,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_NEAREST,
    )

    panel = np.full(
        (zoom.shape[0], 460, 3),
        30,
        dtype=np.uint8,
    )

    match = result["match"]
    upper = result["upper"]
    lower = result["lower"]

    lines = [
        f"Status: {result['status']}",
        f"Reference: {match['reference']}",
        f"Method: {match['method']}",
        f"Stage: {match['stage']}",
        f"Score: {fmt(match['score'], 4)}",
        "",
        f"Top Gap: {fmt(result['top_gap_px'])} px",
        f"Bottom Gap: {fmt(result['bottom_gap_px'])} px",
        "",
        f"Top Ratio: {fmt(result['top_ratio'])} %",
        f"Bottom Ratio: {fmt(result['bottom_ratio'])} %",
        "",
        f"Upper: {result['upper_status']}",
        f"Lower: {result['lower_status']}",
    ]

    if upper is not None:
        lines.append(
            f"Upper support: {upper['horizontal_support']:.2f}"
        )

    if lower is not None:
        lines.append(
            f"Lower support: {lower['horizontal_support']:.2f}"
        )

    for i, line in enumerate(lines):

        y = 30 + 31 * i

        if y >= panel.shape[0]:
            break

        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return np.hstack([zoom, panel])


# ============================================================
# 11. Debug outputs
# ============================================================

def save_debug(debug_dir, stem, debug, result):

    attempts = debug.get("attempts", [])

    pd.DataFrame(attempts).to_csv(
        debug_dir / f"{stem}_matching.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if "profile" not in debug:
        return

    profile = debug["profile"]
    bounds = debug["bounds"]

    x0, y0, x2, y2 = bounds

    rows = []

    for y in range(len(profile["edge"])):

        rows.append({
            "y_global": y0 + y,
            "edge_strength": float(profile["edge"][y]),
            "signed_sobel": float(profile["signed"][y]),
            "row_support": float(profile["support"][y]),
        })

    pd.DataFrame(rows).to_csv(
        debug_dir / f"{stem}_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(debug["candidates"]).to_csv(
        debug_dir / f"{stem}_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Sobel visualization
    # --------------------------------------------------------

    gy = np.abs(profile["gy"])

    p99 = max(
        1.0,
        float(np.percentile(gy, 99)),
    )

    edge_img = np.uint8(
        np.clip(
            gy * 255.0 / p99,
            0, 255,
        )
    )

    cv2.imwrite(
        str(debug_dir / f"{stem}_sobel.png"),
        edge_img,
    )

    # --------------------------------------------------------
    # Candidate line visualization
    # --------------------------------------------------------

    line_img = cv2.cvtColor(
        profile["gray"],
        cv2.COLOR_GRAY2BGR,
    )

    for candidate in debug["candidates"]:

        yy = int(round(candidate["y_local"]))

        cv2.line(
            line_img,
            (0, yy),
            (line_img.shape[1] - 1, yy),
            (0, 255, 255),
            1,
        )

    # Chosen upper/lower line
    for key, color in [
        ("upper", (0, 255, 0)),
        ("lower", (0, 165, 255)),
    ]:

        item = result[key]

        if item is not None:

            yy = int(round(
                item["y_global"] - y0
            ))

            cv2.line(
                line_img,
                (0, yy),
                (line_img.shape[1] - 1, yy),
                color,
                2,
            )

    cv2.imwrite(
        str(
            debug_dir
            / f"{stem}_horizontal_lines.png"
        ),
        line_img,
    )


# ============================================================
# 12. CSV serialization
# ============================================================

def flatten_result(filename, result):

    match = result["match"]
    anchors = result["anchors"]
    upper = result["upper"]
    lower = result["lower"]

    def get(obj, key):
        return obj.get(key) if obj is not None else None

    return {
        "image": filename,
        "status": result["status"],

        "match_reference": match["reference"],
        "match_method": match["method"],
        "match_stage": match["stage"],
        "match_score": match["score"],

        "match_x": match["x"],
        "match_y": match["y"],
        "match_w": match["w"],
        "match_h": match["h"],

        "measure_x": get(anchors, "x"),

        "blade_top_y": get(anchors, "top_y"),
        "blade_bottom_y": get(anchors, "bottom_y"),

        "upper_wafer_y": get(upper, "y_global"),
        "lower_wafer_y": get(lower, "y_global"),

        "upper_status": result["upper_status"],
        "lower_status": result["lower_status"],

        "upper_support": get(
            upper,
            "horizontal_support",
        ),

        "lower_support": get(
            lower,
            "horizontal_support",
        ),

        "upper_span": get(upper, "span_ratio"),
        "lower_span": get(lower, "span_ratio"),

        "top_gap_px": result["top_gap_px"],
        "bottom_gap_px": result["bottom_gap_px"],

        "top_ratio": result["top_ratio"],
        "bottom_ratio": result["bottom_ratio"],
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase B v5 - Dual Reference Matching "
            "+ Full ROI Horizontal Wafer Detection"
        )
    )

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    parser.add_argument("--images", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--dark-template", required=True)

    parser.add_argument(
        "--output",
        default="output/06_phase_b_v5",
    )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    parser.add_argument(
        "--search-roi",
        nargs=4, type=int,
        default=[41, 358, 2027, 186],
    )

    parser.add_argument(
        "--measure-roi",
        nargs=4, type=int,
        default=[49, 296, 600, 273],
    )

    # --------------------------------------------------------
    # Matching
    # --------------------------------------------------------

    parser.add_argument(
        "--gray-threshold",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--dark-gray-threshold",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--clahe-threshold",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--dark-clahe-threshold",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--clahe-grid",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--expected-match-x",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--expected-match-y",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--max-position-error",
        type=float,
        default=30.0,
    )

    # --------------------------------------------------------
    # Normal template geometry
    # --------------------------------------------------------

    parser.add_argument(
        "--normal-x-ratio",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--normal-top-ratio",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--normal-bottom-ratio",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--normal-top-offset",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--normal-bottom-offset",
        type=float,
        default=0.0,
    )

    # --------------------------------------------------------
    # Dark template geometry
    # --------------------------------------------------------

    parser.add_argument(
        "--dark-x-ratio",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--dark-top-ratio",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--dark-bottom-ratio",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dark-top-offset",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--dark-bottom-offset",
        type=float,
        default=0.0,
    )

    # --------------------------------------------------------
    # Wafer detection
    # --------------------------------------------------------

    parser.add_argument(
        "--wafer-edge-threshold",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--wafer-min-prominence",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--wafer-support-gradient",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--wafer-min-support",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--wafer-min-span",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--wafer-min-strips",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--strip-width",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--strip-step",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--line-y-tolerance",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-gap",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--max-gap",
        type=float,
        default=100.0,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    cfg = Config(
        search_roi=tuple(args.search_roi),
        measure_roi=tuple(args.measure_roi),

        normal_gray_threshold=args.gray_threshold,
        dark_gray_threshold=args.dark_gray_threshold,

        normal_clahe_threshold=args.clahe_threshold,
        dark_clahe_threshold=args.dark_clahe_threshold,

        review_threshold=args.review_threshold,

        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,

        expected_match_x=args.expected_match_x,
        expected_match_y=args.expected_match_y,
        max_position_error=args.max_position_error,

        normal_x_ratio=args.normal_x_ratio,
        normal_top_ratio=args.normal_top_ratio,
        normal_bottom_ratio=args.normal_bottom_ratio,
        normal_top_offset=args.normal_top_offset,
        normal_bottom_offset=args.normal_bottom_offset,

        dark_x_ratio=args.dark_x_ratio,
        dark_top_ratio=args.dark_top_ratio,
        dark_bottom_ratio=args.dark_bottom_ratio,
        dark_top_offset=args.dark_top_offset,
        dark_bottom_offset=args.dark_bottom_offset,

        wafer_edge_threshold=args.wafer_edge_threshold,
        wafer_min_prominence=args.wafer_min_prominence,
        wafer_support_gradient=args.wafer_support_gradient,

        wafer_min_support=args.wafer_min_support,
        wafer_min_span_ratio=args.wafer_min_span,
        wafer_min_valid_strips=args.wafer_min_strips,

        strip_width=args.strip_width,
        strip_step=args.strip_step,
        line_y_tolerance=args.line_y_tolerance,

        min_gap=args.min_gap,
        max_gap=args.max_gap,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not 0 <= cfg.normal_x_ratio <= 1:
        parser.error("normal-x-ratio must be 0~1")

    if not 0 <= cfg.dark_x_ratio <= 1:
        parser.error("dark-x-ratio must be 0~1")

    if cfg.strip_width < 1 or cfg.strip_step < 1:
        parser.error("Invalid strip width/step")

    if not 0 <= cfg.wafer_min_support <= 1:
        parser.error("wafer-min-support must be 0~1")

    if not 0 <= cfg.wafer_min_span_ratio <= 1:
        parser.error("wafer-min-span must be 0~1")

    if cfg.line_y_tolerance < 0:
        parser.error("line-y-tolerance must be >= 0")

    if cfg.max_gap <= cfg.min_gap:
        parser.error("max-gap must be greater than min-gap")

    if (
        (cfg.expected_match_x is None)
        != (cfg.expected_match_y is None)
    ):
        parser.error(
            "expected-match-x and expected-match-y "
            "must be provided together"
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    image_dir = Path(args.images)
    output_dir = Path(args.output)

    overlay_dir = output_dir / "overlay"
    detail_dir = output_dir / "detail"
    debug_dir = output_dir / "debug"

    for directory in [
        output_dir,
        overlay_dir,
        detail_dir,
        debug_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    references = load_references(
        args.template,
        args.dark_template,
        cfg,
    )

    files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }
    )

    if not files:
        raise FileNotFoundError(
            f"No images found: {image_dir}"
        )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    print("=" * 72)
    print("Phase B v5 - Full ROI Wafer Detection")
    print("=" * 72)

    print(f"Images            : {len(files)}")
    print(f"Normal Reference  : {args.template}")
    print(f"Dark Reference    : {args.dark_template}")

    print(f"Search ROI        : {cfg.search_roi}")
    print(f"Measurement ROI   : {cfg.measure_roi}")

    print(f"Normal X ratio    : {cfg.normal_x_ratio}")
    print(f"Dark X ratio      : {cfg.dark_x_ratio}")

    print(f"Wafer edge thr    : {cfg.wafer_edge_threshold}")
    print(f"Wafer support     : {cfg.wafer_min_support}")
    print(f"Wafer span        : {cfg.wafer_min_span_ratio}")

    print()

    rows = []

    for i, path in enumerate(files, start=1):

        image = cv2.imread(str(path))

        if image is None:
            rows.append({
                "image": path.name,
                "status": "image_read_failed",
            })
            continue

        try:

            result, debug = analyze_image(
                image,
                references,
                cfg,
            )

            overlay = draw_overlay(
                image,
                result,
                debug,
                cfg,
            )

            detail = create_detail(
                overlay,
                result,
                cfg,
            )

            cv2.imwrite(
                str(overlay_dir / path.name),
                overlay,
            )

            cv2.imwrite(
                str(detail_dir / path.name),
                detail,
            )

            save_debug(
                debug_dir,
                path.stem,
                debug,
                result,
            )

            row = flatten_result(
                path.name,
                result,
            )

        except Exception as exc:

            row = {
                "image": path.name,
                "status": "error",
                "error": str(exc),
            }

            print(
                f"[ERROR] {path.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

        rows.append(row)

        print(
            f"[{i:02d}/{len(files):02d}] "
            f"{path.name} | "
            f"{row['status']} | "
            f"Ref={row.get('match_reference')} | "
            f"Score={row.get('match_score')} | "
            f"Top={row.get('top_gap_px')} | "
            f"Bottom={row.get('bottom_gap_px')}"
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    csv_path = (
        output_dir / "geometry_results.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print("Phase B v5 Result")
    print("=" * 72)

    print(
        df["status"]
        .value_counts()
        .to_string()
    )

    print()
    print(f"CSV     : {csv_path}")
    print(f"Overlay : {overlay_dir}")
    print(f"Detail  : {detail_dir}")
    print(f"Debug   : {debug_dir}")


if __name__ == "__main__":
    main()
