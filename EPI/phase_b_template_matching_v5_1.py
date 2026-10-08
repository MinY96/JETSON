
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
    search_roi: tuple = (41, 358, 2027, 186)
    measure_roi: tuple = (49, 296, 600, 273)

    # Template matching
    normal_gray_threshold: float = 0.65
    dark_gray_threshold: float = 0.65
    normal_clahe_threshold: float = 0.65
    dark_clahe_threshold: float = 0.65
    review_threshold: float = 0.30

    clahe_clip: float = 2.0
    clahe_grid: int = 8

    expected_match_x: float | None = None
    expected_match_y: float | None = None
    max_position_error: float = 30.0

    # Normal reference geometry
    normal_x_ratio: float = 0.7
    normal_top_ratio: float = 0.0
    normal_bottom_ratio: float = 1.0
    normal_top_offset: float = 0.0
    normal_bottom_offset: float = 0.0

    # Dark reference geometry
    dark_x_ratio: float = 0.7
    dark_top_ratio: float = 0.0
    dark_bottom_ratio: float = 1.0
    dark_top_offset: float = 0.0
    dark_bottom_offset: float = 0.0

    # Wafer detection (v5)
    blur_ksize: int = 3
    profile_smooth: int = 3
    wafer_edge_threshold: float = 10.0
    wafer_min_prominence: float = 4.0
    wafer_support_gradient: float = 15.0
    wafer_min_support: float = 0.20
    wafer_min_span_ratio: float = 0.35
    wafer_min_valid_strips: int = 3
    strip_width: int = 24
    strip_step: int = 12
    line_y_tolerance: int = 3
    edge_min_distance: int = 3

    min_gap: float = 5.0
    max_gap: float = 100.0
    blade_exclusion_px: float = 4.0

    # Overlay
    overlay_panel_width: int = 340
    detail_scale: float = 2.0
    panel_bg: tuple = (33, 37, 43)
    jpeg_quality: int = 95


@dataclass
class Reference:
    name: str
    image: np.ndarray
    x_ratio: float
    top_ratio: float
    bottom_ratio: float
    top_offset: float
    bottom_offset: float


# OpenCV uses BGR
COLOR_WAFER = (245, 130, 45)       # Blue
COLOR_BLADE = (60, 65, 245)        # Red
COLOR_GAP = (50, 50, 245)          # Red
COLOR_MEASURE = (155, 155, 155)    # Gray
COLOR_WHITE = (245, 245, 245)
COLOR_MUTED = (170, 180, 190)


# ============================================================
# Common utilities
# ============================================================

def odd(value):
    value = max(1, int(value))
    return value if value % 2 else value + 1


def safe_int(value):
    return int(round(float(value)))


def fmt(value, decimals=1):
    if value is None:
        return "SKIP"
    return f"{value:.{decimals}f}"


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

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def smooth_1d(values, kernel):
    values = np.asarray(values, dtype=np.float32)

    if kernel <= 1:
        return values.copy()

    return cv2.GaussianBlur(
        values.reshape(-1, 1),
        (1, odd(kernel)),
        0,
    ).ravel()


# ============================================================
# Template matching
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
            "normal",
            normal,
            cfg.normal_x_ratio,
            cfg.normal_top_ratio,
            cfg.normal_bottom_ratio,
            cfg.normal_top_offset,
            cfg.normal_bottom_offset,
        ),
        "dark": Reference(
            "dark",
            dark,
            cfg.dark_x_ratio,
            cfg.dark_top_ratio,
            cfg.dark_bottom_ratio,
            cfg.dark_top_offset,
            cfg.dark_bottom_offset,
        ),
    }


def preprocess_matching(image, method, cfg):
    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    if method == "gray":
        return cv2.GaussianBlur(gray, (3, 3), 0)

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
            enhanced, (3, 3), 0
        )

    raise ValueError(method)


def match_template(
    search_image,
    search_origin,
    reference,
    method,
    cfg,
):
    target = preprocess_matching(
        reference.image, method, cfg
    )

    th, tw = target.shape
    sh, sw = search_image.shape

    base = {
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

    if tw > sw or th > sh:
        return base

    if float(np.std(target)) < 1e-5:
        return base

    response = cv2.matchTemplate(
        search_image,
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

    x = search_origin[0] + loc[0]
    y = search_origin[1] + loc[1]

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

    base.update({
        "score": float(score),
        "x": int(x),
        "y": int(y),
        "position_error": position_error,
        "position_valid": position_valid,
    })

    return base


def select_blade_match(image, references, cfg):
    search, bounds = crop_roi(
        image, cfg.search_roi
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

    cache = {}
    attempts = []
    selected = None

    for stage, (name, method, threshold) in enumerate(
        stages, 1
    ):
        if method not in cache:
            cache[method] = preprocess_matching(
                search, method, cfg
            )

        candidate = match_template(
            cache[method],
            origin,
            references[name],
            method,
            cfg,
        )

        passed = (
            candidate["score"] is not None
            and candidate["score"] >= threshold
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
            c for c in attempts
            if c["score"] is not None
            and c["position_valid"]
        ]

        if valid:
            selected = max(
                valid,
                key=lambda c: c["score"],
            ).copy()
            selected["status"] = (
                "review"
                if selected["score"] >= cfg.review_threshold
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
# Blade anchors
# ============================================================

def calculate_blade_anchors(match, references):
    ref = references[match["reference"]]

    x = match["x"] + ref.x_ratio * match["w"]

    top_y = (
        match["y"]
        + ref.top_ratio * match["h"]
        + ref.top_offset
    )

    bottom_y = (
        match["y"]
        + ref.bottom_ratio * match["h"]
        + ref.bottom_offset
    )

    if top_y >= bottom_y:
        raise ValueError(
            "Invalid Blade geometry: top_y >= bottom_y"
        )

    return {
        "x": float(x),
        "top_y": float(top_y),
        "bottom_y": float(bottom_y),
        "thickness": float(bottom_y - top_y),
    }


# ============================================================
# Full ROI wafer detection (same logic as v5)
# ============================================================

def get_measurement_roi(image, anchors, cfg):
    roi, bounds = crop_roi(
        image, cfg.measure_roi
    )

    x1, y1, x2, y2 = bounds

    if not (x1 <= anchors["x"] < x2):
        raise ValueError(
            f"Measurement X={anchors['x']:.1f} "
            f"outside ROI X=[{x1},{x2})"
        )

    if not (
        y1 <= anchors["top_y"]
        < anchors["bottom_y"]
        < y2
    ):
        raise ValueError(
            "Blade anchors outside Measurement ROI"
        )

    return roi, bounds


def calculate_wafer_profile(roi, cfg):
    gray = cv2.cvtColor(
        roi, cv2.COLOR_BGR2GRAY
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
    clipped = np.minimum(abs_gy, 120.0)

    edge = smooth_1d(
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

    signed = smooth_1d(
        np.mean(gy, axis=1),
        cfg.profile_smooth,
    )

    return {
        "gray": gray,
        "gy": gy,
        "edge": edge,
        "support": support,
        "signed": signed,
    }


def calculate_horizontal_support(gy, y, cfg):
    height, width = gy.shape

    ya = max(0, y - cfg.line_y_tolerance)
    yb = min(
        height,
        y + cfg.line_y_tolerance + 1,
    )

    region_y = np.abs(gy[ya:yb])

    strip_width = min(cfg.strip_width, width)

    positions = list(range(
        0,
        max(1, width - strip_width + 1),
        cfg.strip_step,
    ))

    last = max(0, width - strip_width)

    if last not in positions:
        positions.append(last)

    positions = sorted(set(positions))

    valid_x = []

    for x1 in positions:
        x2 = min(width, x1 + strip_width)

        region = region_y[:, x1:x2]

        column_hit = np.any(
            region >= cfg.wafer_support_gradient,
            axis=0,
        )

        if float(np.mean(column_hit)) >= cfg.wafer_min_support:
            valid_x.append((x1 + x2 - 1) / 2.0)

    total = len(positions)
    valid_count = len(valid_x)

    support_ratio = (
        valid_count / total
        if total else 0.0
    )

    if valid_x:
        xmin = float(min(valid_x))
        xmax = float(max(valid_x))

        span_ratio = min(
            1.0,
            (xmax - xmin + strip_width)
            / max(1, width),
        )
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


def detect_wafer_lines(profile, bounds, cfg):
    edge = profile["edge"]
    gy = profile["gy"]

    x_offset, y_offset, _, _ = bounds
    candidates = []

    for y in range(2, len(edge) - 2):
        strength = float(edge[y])

        if strength < cfg.wafer_edge_threshold:
            continue

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
            gy, y, cfg
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

        xmin = horizontal["observed_x_min"]
        xmax = horizontal["observed_x_max"]

        candidates.append({
            "y_local": float(y),
            "y_global": float(y + y_offset),
            "strength": strength,
            "prominence": prominence,
            "row_support": float(
                profile["support"][y]
            ),
            "horizontal_support":
                horizontal["support_ratio"],
            "span_ratio":
                horizontal["span_ratio"],
            "valid_strips":
                horizontal["valid_strips"],
            "observed_x_min":
                None if xmin is None else xmin + x_offset,
            "observed_x_max":
                None if xmax is None else xmax + x_offset,
        })

    # Remove duplicate nearby peaks
    candidates.sort(
        key=lambda c: c["strength"],
        reverse=True,
    )

    filtered = []

    for candidate in candidates:
        duplicate = any(
            abs(
                candidate["y_global"]
                - previous["y_global"]
            ) < cfg.edge_min_distance
            for previous in filtered
        )

        if not duplicate:
            filtered.append(candidate)

    return sorted(
        filtered,
        key=lambda c: c["y_global"],
    )


def find_nearest_wafer(
    candidates, blade_y, side, cfg
):
    valid = []

    for candidate in candidates:
        wafer_y = candidate["y_global"]

        if side == "upper":
            gap = blade_y - wafer_y
        else:
            gap = wafer_y - blade_y

        minimum = max(
            cfg.min_gap,
            cfg.blade_exclusion_px,
        )

        if not minimum <= gap <= cfg.max_gap:
            continue

        valid.append({
            **candidate,
            "gap_px": float(gap),
        })

    if not valid:
        return None, "not_detected", []

    valid.sort(key=lambda c: c["gap_px"])

    return valid[0], "detected", valid


# ============================================================
# Geometry measurement
# ============================================================

def analyze_image(image, references, cfg):
    match, attempts = select_blade_match(
        image, references, cfg
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

    debug = {"attempts": attempts}

    if match["status"] != "matched":
        return result, debug

    anchors = calculate_blade_anchors(
        match, references
    )

    result["anchors"] = anchors

    roi, bounds = get_measurement_roi(
        image, anchors, cfg
    )

    profile = calculate_wafer_profile(
        roi, cfg
    )

    candidates = detect_wafer_lines(
        profile, bounds, cfg
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

    top_gap = (
        upper["gap_px"] if upper else None
    )

    bottom_gap = (
        lower["gap_px"] if lower else None
    )

    top_ratio = None
    bottom_ratio = None

    if top_gap is not None and bottom_gap is not None:
        total = top_gap + bottom_gap

        if total > 0:
            top_ratio = 100 * top_gap / total
            bottom_ratio = 100 * bottom_gap / total

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
# v5.1 Overlay drawing helpers
# ============================================================

def draw_dashed_line(
    image,
    pt1,
    pt2,
    color,
    thickness=1,
    dash_length=8,
    gap_length=6,
):
    x1, y1 = pt1
    x2, y2 = pt2

    length = float(
        np.hypot(x2 - x1, y2 - y1)
    )

    if length < 1:
        return

    dx = (x2 - x1) / length
    dy = (y2 - y1) / length

    distance = 0.0

    while distance < length:
        end_distance = min(
            distance + dash_length,
            length,
        )

        start = (
            safe_int(x1 + dx * distance),
            safe_int(y1 + dy * distance),
        )

        end = (
            safe_int(x1 + dx * end_distance),
            safe_int(y1 + dy * end_distance),
        )

        cv2.line(
            image,
            start,
            end,
            color,
            thickness,
            cv2.LINE_AA,
        )

        distance += dash_length + gap_length


def draw_double_arrow(
    image,
    x,
    y1,
    y2,
    color=COLOR_GAP,
    thickness=3,
):
    """
    Vertical double-ended arrow for actual Y gap.
    """
    top = min(y1, y2)
    bottom = max(y1, y2)

    if bottom - top < 2:
        return

    # Keep arrowheads proportional for short gaps
    gap = bottom - top
    tip = max(2, min(7, gap // 3))

    cv2.line(
        image,
        (x, top),
        (x, bottom),
        color,
        thickness,
        cv2.LINE_AA,
    )

    # Upper arrow head
    cv2.line(
        image,
        (x, top),
        (x - tip, top + tip),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        (x, top),
        (x + tip, top + tip),
        color,
        thickness,
        cv2.LINE_AA,
    )

    # Lower arrow head
    cv2.line(
        image,
        (x, bottom),
        (x - tip, bottom - tip),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        (x, bottom),
        (x + tip, bottom - tip),
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_pill(
    image,
    text,
    center,
    bg_color=COLOR_GAP,
    text_color=COLOR_WHITE,
    font_scale=0.42,
):
    """
    Small rounded text pill with clipped-image protection.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    padding_x = 9
    padding_y = 5

    pill_w = tw + 2 * padding_x
    pill_h = th + baseline + 2 * padding_y

    h, w = image.shape[:2]

    if pill_w >= w or pill_h >= h:
        return

    cx, cy = center

    left = safe_int(cx - pill_w / 2)
    top = safe_int(cy - pill_h / 2)

    left = max(1, min(left, w - pill_w - 1))
    top = max(1, min(top, h - pill_h - 1))

    right = left + pill_w
    bottom = top + pill_h

    radius = min(10, pill_h // 2)

    overlay = image.copy()

    # Rounded rectangle, no external libraries
    cv2.rectangle(
        overlay,
        (left + radius, top),
        (right - radius, bottom),
        bg_color,
        -1,
    )
    cv2.rectangle(
        overlay,
        (left, top + radius),
        (right, bottom - radius),
        bg_color,
        -1,
    )

    for px, py in [
        (left + radius, top + radius),
        (right - radius, top + radius),
        (left + radius, bottom - radius),
        (right - radius, bottom - radius),
    ]:
        cv2.circle(
            overlay,
            (px, py),
            radius,
            bg_color,
            -1,
        )

    # Semi-transparent background
    cv2.addWeighted(
        overlay,
        0.87,
        image,
        0.13,
        0,
        image,
    )

    text_x = left + padding_x
    text_y = top + padding_y + th

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_wafer_label(
    image,
    text,
    line_y,
    right_x,
    place_above=True,
):
    """
    Label near the right end of the Wafer horizontal line.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2

    (tw, th), _ = cv2.getTextSize(
        text, font, scale, thickness
    )

    x = max(4, right_x - tw - 12)

    if place_above:
        baseline_y = line_y - 10
    else:
        baseline_y = line_y + th + 12

    baseline_y = max(
        th + 5,
        min(
            image.shape[0] - 5,
            baseline_y,
        ),
    )

    # Thin black shadow for visibility
    cv2.putText(
        image,
        text,
        (x + 1, baseline_y + 1),
        font,
        scale,
        (20, 20, 20),
        thickness + 2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        (x, baseline_y),
        font,
        scale,
        COLOR_WAFER,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# Top:Bottom ratio 0~10
# ============================================================

def format_ratio_10(top_gap, bottom_gap):
    """
    Ratio rounded to integers whose sum is exactly 10.

    Example:
        top=19, bottom=28
        -> 4:6

    A missing wafer produces SKIP.
    """
    if top_gap is None or bottom_gap is None:
        return "SKIP"

    total = top_gap + bottom_gap

    if total <= 0:
        return "SKIP"

    top = int(np.floor(
        10.0 * top_gap / total + 0.5
    ))

    top = max(0, min(10, top))
    bottom = 10 - top

    return f"{top}:{bottom}"


# ============================================================
# v5.1 Annotation
# ============================================================

def draw_annotation(image, result, debug, cfg):
    """
    Draw ONLY measurement graphics on the original image.

    No ROI rectangles.
    No Template Matching rectangle.
    No information panel inside the image.
    """
    out = image.copy()
    anchors = result["anchors"]

    if anchors is None:
        return out

    x1, y1, x2, y2 = debug["bounds"]

    right_x = x2 - 1
    measure_x = safe_int(anchors["x"])

    blade_top = safe_int(
        anchors["top_y"]
    )

    blade_bottom = safe_int(
        anchors["bottom_y"]
    )

    # --------------------------------------------------------
    # 1. Gray dashed measurement line
    # --------------------------------------------------------

    draw_dashed_line(
        out,
        (measure_x, y1),
        (measure_x, y2 - 1),
        color=COLOR_MEASURE,
        thickness=1,
        dash_length=8,
        gap_length=6,
    )

    # --------------------------------------------------------
    # 2. Thin red dashed Blade guides
    # --------------------------------------------------------

    draw_dashed_line(
        out,
        (x1, blade_top),
        (right_x, blade_top),
        color=COLOR_BLADE,
        thickness=1,
        dash_length=7,
        gap_length=5,
    )

    draw_dashed_line(
        out,
        (x1, blade_bottom),
        (right_x, blade_bottom),
        color=COLOR_BLADE,
        thickness=1,
        dash_length=7,
        gap_length=5,
    )

    # --------------------------------------------------------
    # 3. Blue Wafer horizontal lines
    # --------------------------------------------------------

    upper = result["upper"]
    lower = result["lower"]

    if upper is not None:
        upper_y = safe_int(
            upper["y_global"]
        )

        cv2.line(
            out,
            (x1, upper_y),
            (right_x, upper_y),
            COLOR_WAFER,
            2,
            cv2.LINE_AA,
        )

        draw_wafer_label(
            out,
            "Top Wafer",
            upper_y,
            right_x,
            place_above=True,
        )

    if lower is not None:
        lower_y = safe_int(
            lower["y_global"]
        )

        cv2.line(
            out,
            (x1, lower_y),
            (right_x, lower_y),
            COLOR_WAFER,
            2,
            cv2.LINE_AA,
        )

        draw_wafer_label(
            out,
            "Lower Wafer",
            lower_y,
            right_x,
            place_above=False,
        )

    # --------------------------------------------------------
    # 4. Red gap arrows with small pill labels
    #
    # Gap arrow is offset slightly to the right
    # of the gray measurement line for readability.
    # --------------------------------------------------------

    gap_x = min(
        right_x - 30,
        measure_x + 20,
    )
    gap_x = max(x1 + 12, gap_x)

    if upper is not None:
        upper_y = safe_int(
            upper["y_global"]
        )

        draw_double_arrow(
            out,
            gap_x,
            upper_y,
            blade_top,
            color=COLOR_GAP,
            thickness=3,
        )

        mid_y = safe_int(
            (upper_y + blade_top) / 2
        )

        draw_pill(
            out,
            f"{result['top_gap_px']:.0f} px",
            (gap_x + 57, mid_y),
            bg_color=COLOR_GAP,
            font_scale=0.42,
        )

    if lower is not None:
        lower_y = safe_int(
            lower["y_global"]
        )

        draw_double_arrow(
            out,
            gap_x,
            blade_bottom,
            lower_y,
            color=COLOR_GAP,
            thickness=3,
        )

        mid_y = safe_int(
            (blade_bottom + lower_y) / 2
        )

        draw_pill(
            out,
            f"{result['bottom_gap_px']:.0f} px",
            (gap_x + 57, mid_y),
            bg_color=COLOR_GAP,
            font_scale=0.42,
        )

    return out


# ============================================================
# v5.1 Information panel
# ============================================================

def draw_info_panel(
    result,
    height,
    width=340,
    cfg=None,
):
    """
    Standalone right-side panel.

    Shows only:
      Status
      Top Gap
      Bottom Gap
      Diff
      Top:Bottom
    """
    height = max(250, int(height))
    width = max(250, int(width))

    bg = (
        cfg.panel_bg
        if cfg is not None
        else (33, 37, 43)
    )

    panel = np.full(
        (height, width, 3),
        bg,
        dtype=np.uint8,
    )

    top = result["top_gap_px"]
    bottom = result["bottom_gap_px"]

    if top is not None and bottom is not None:
        diff = abs(top - bottom)
        diff_text = f"{diff:.1f} px"
    else:
        diff_text = "SKIP"

    top_text = (
        f"{top:.1f} px"
        if top is not None
        else "SKIP"
    )

    bottom_text = (
        f"{bottom:.1f} px"
        if bottom is not None
        else "SKIP"
    )

    ratio_text = format_ratio_10(
        top, bottom
    )

    fields = [
        ("Status", str(result["status"])),
        ("Top Gap", top_text),
        ("Bottom Gap", bottom_text),
        ("Diff", diff_text),
        ("Top:Bottom", ratio_text),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Header
    cv2.putText(
        panel,
        "MEASUREMENT",
        (24, 48),
        font,
        0.73,
        COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )

    cv2.line(
        panel,
        (24, 68),
        (width - 24, 68),
        (90, 95, 105),
        1,
        cv2.LINE_AA,
    )

    # Five information rows
    row_start = 105
    row_gap = 55

    for i, (label, value) in enumerate(fields):
        y = row_start + i * row_gap

        if y + 12 >= height:
            break

        cv2.putText(
            panel,
            label,
            (24, y),
            font,
            0.54,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        # Right aligned value
        font_scale = 0.61
        font_thickness = 2

        (tw, _), _ = cv2.getTextSize(
            value,
            font,
            font_scale,
            font_thickness,
        )

        value_x = max(
            125,
            width - 24 - tw,
        )

        value_color = COLOR_WHITE

        if label == "Top:Bottom":
            value_color = (115, 220, 255)

        cv2.putText(
            panel,
            value,
            (value_x, y),
            font,
            font_scale,
            value_color,
            font_thickness,
            cv2.LINE_AA,
        )

        if i < len(fields) - 1:
            cv2.line(
                panel,
                (24, y + 18),
                (width - 24, y + 18),
                (58, 63, 70),
                1,
                cv2.LINE_AA,
            )

    return panel


# ============================================================
# v5.1 Output images
# ============================================================

def draw_overlay(image, result, debug, cfg):
    """
    Full-resolution original image + right-side info panel.
    """
    annotated = draw_annotation(
        image, result, debug, cfg
    )

    panel = draw_info_panel(
        result,
        height=annotated.shape[0],
        width=cfg.overlay_panel_width,
        cfg=cfg,
    )

    return np.hstack([annotated, panel])


def create_detail(image, result, debug, cfg):
    """
    Cropped Measurement ROI (including annotations)
    + right-side info panel.

    The image is rendered from the original frame,
    not from an already expanded overlay.
    """
    annotated = draw_annotation(
        image, result, debug, cfg
    )

    roi, _ = crop_roi(
        annotated,
        cfg.measure_roi,
    )

    scale = cfg.detail_scale

    zoom = cv2.resize(
        roi,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )

    panel = draw_info_panel(
        result,
        height=zoom.shape[0],
        width=cfg.overlay_panel_width,
        cfg=cfg,
    )

    return np.hstack([zoom, panel])


# ============================================================
# Debug
# ============================================================

def save_debug(debug_dir, stem, debug, result):
    pd.DataFrame(
        debug.get("attempts", [])
    ).to_csv(
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
            "edge_strength":
                float(profile["edge"][y]),
            "signed_sobel":
                float(profile["signed"][y]),
            "row_support":
                float(profile["support"][y]),
        })

    pd.DataFrame(rows).to_csv(
        debug_dir / f"{stem}_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        debug["candidates"],
        columns=[
            "y_local",
            "y_global",
            "strength",
            "prominence",
            "row_support",
            "horizontal_support",
            "span_ratio",
            "valid_strips",
            "observed_x_min",
            "observed_x_max",
        ],
    ).to_csv(
        debug_dir / f"{stem}_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Sobel image
    gy = np.abs(profile["gy"])

    p99 = max(
        1.0,
        float(np.percentile(gy, 99)),
    )

    sobel = np.uint8(
        np.clip(
            gy * 255.0 / p99,
            0,
            255,
        )
    )

    cv2.imwrite(
        str(debug_dir / f"{stem}_sobel.png"),
        sobel,
    )

    # All horizontal candidates
    line_img = cv2.cvtColor(
        profile["gray"],
        cv2.COLOR_GRAY2BGR,
    )

    for candidate in debug["candidates"]:
        yy = safe_int(
            candidate["y_local"]
        )

        cv2.line(
            line_img,
            (0, yy),
            (line_img.shape[1] - 1, yy),
            (0, 255, 255),
            1,
        )

    for key, color in [
        ("upper", COLOR_WAFER),
        ("lower", COLOR_WAFER),
    ]:
        wafer = result[key]

        if wafer is None:
            continue

        yy = safe_int(
            wafer["y_global"] - y0
        )

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
# CSV serialization
# ============================================================

def flatten_result(filename, result):
    match = result["match"]
    anchors = result["anchors"]
    upper = result["upper"]
    lower = result["lower"]

    def get(obj, key):
        return obj.get(key) if obj is not None else None

    top = result["top_gap_px"]
    bottom = result["bottom_gap_px"]

    diff = (
        abs(top - bottom)
        if top is not None and bottom is not None
        else None
    )

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
        "blade_thickness_px":
            get(anchors, "thickness"),

        "upper_wafer_y": get(upper, "y_global"),
        "lower_wafer_y": get(lower, "y_global"),

        "upper_status": result["upper_status"],
        "lower_status": result["lower_status"],

        "upper_support":
            get(upper, "horizontal_support"),
        "lower_support":
            get(lower, "horizontal_support"),

        "upper_span": get(upper, "span_ratio"),
        "lower_span": get(lower, "span_ratio"),

        "top_gap_px": top,
        "bottom_gap_px": bottom,
        "gap_diff_px": diff,

        "top_ratio": result["top_ratio"],
        "bottom_ratio": result["bottom_ratio"],

        "top_bottom_ratio_10":
            format_ratio_10(top, bottom),
    }


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Phase B v5.1 - Dual Reference Matching, "
            "Full ROI Wafer Detection, Enhanced Overlay"
        )
    )

    # Input / output
    parser.add_argument("--images", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--dark-template", required=True)
    parser.add_argument(
        "--output",
        default="output/06_phase_b_v5_1",
    )

    # ROI
    parser.add_argument(
        "--search-roi",
        nargs=4,
        type=int,
        default=[41, 358, 2027, 186],
    )
    parser.add_argument(
        "--measure-roi",
        nargs=4,
        type=int,
        default=[49, 296, 600, 273],
    )

    # Matching
    parser.add_argument(
        "--gray-threshold",
        type=float, default=0.65,
    )
    parser.add_argument(
        "--dark-gray-threshold",
        type=float, default=0.65,
    )
    parser.add_argument(
        "--clahe-threshold",
        type=float, default=0.65,
    )
    parser.add_argument(
        "--dark-clahe-threshold",
        type=float, default=0.65,
    )
    parser.add_argument(
        "--review-threshold",
        type=float, default=0.30,
    )
    parser.add_argument(
        "--clahe-clip",
        type=float, default=2.0,
    )
    parser.add_argument(
        "--clahe-grid",
        type=int, default=8,
    )
    parser.add_argument(
        "--expected-match-x",
        type=float, default=None,
    )
    parser.add_argument(
        "--expected-match-y",
        type=float, default=None,
    )
    parser.add_argument(
        "--max-position-error",
        type=float, default=30.0,
    )

    # Normal geometry
    parser.add_argument(
        "--normal-x-ratio",
        type=float, default=0.7,
    )
    parser.add_argument(
        "--normal-top-ratio",
        type=float, default=0.0,
    )
    parser.add_argument(
        "--normal-bottom-ratio",
        type=float, default=1.0,
    )
    parser.add_argument(
        "--normal-top-offset",
        type=float, default=0.0,
    )
    parser.add_argument(
        "--normal-bottom-offset",
        type=float, default=0.0,
    )

    # Dark geometry
    parser.add_argument(
        "--dark-x-ratio",
        type=float, default=0.7,
    )
    parser.add_argument(
        "--dark-top-ratio",
        type=float, default=0.0,
    )
    parser.add_argument(
        "--dark-bottom-ratio",
        type=float, default=1.0,
    )
    parser.add_argument(
        "--dark-top-offset",
        type=float, default=0.0,
    )
    parser.add_argument(
        "--dark-bottom-offset",
        type=float, default=0.0,
    )

    # Wafer detection
    parser.add_argument(
        "--wafer-edge-threshold",
        type=float, default=10.0,
    )
    parser.add_argument(
        "--wafer-min-prominence",
        type=float, default=4.0,
    )
    parser.add_argument(
        "--wafer-support-gradient",
        type=float, default=15.0,
    )
    parser.add_argument(
        "--wafer-min-support",
        type=float, default=0.20,
    )
    parser.add_argument(
        "--wafer-min-span",
        type=float, default=0.35,
    )
    parser.add_argument(
        "--wafer-min-strips",
        type=int, default=3,
    )
    parser.add_argument(
        "--strip-width",
        type=int, default=24,
    )
    parser.add_argument(
        "--strip-step",
        type=int, default=12,
    )
    parser.add_argument(
        "--line-y-tolerance",
        type=int, default=3,
    )
    parser.add_argument(
        "--min-gap",
        type=float, default=5.0,
    )
    parser.add_argument(
        "--max-gap",
        type=float, default=100.0,
    )

    # Visualization
    parser.add_argument(
        "--detail-scale",
        type=float, default=2.0,
    )
    parser.add_argument(
        "--panel-width",
        type=int, default=340,
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int, default=95,
    )

    return parser


def config_from_args(args):
    return Config(
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

        detail_scale=args.detail_scale,
        overlay_panel_width=args.panel_width,
        jpeg_quality=args.jpeg_quality,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = config_from_args(args)

    # Validate arguments
    for name in (
        "normal_x_ratio",
        "dark_x_ratio",
        "normal_top_ratio",
        "normal_bottom_ratio",
        "dark_top_ratio",
        "dark_bottom_ratio",
        "wafer_min_support",
        "wafer_min_span_ratio",
    ):
        value = getattr(cfg, name)

        if not 0 <= value <= 1:
            parser.error(
                f"{name} must be between 0 and 1"
            )

    if cfg.strip_width < 1 or cfg.strip_step < 1:
        parser.error("Invalid strip width/step")

    if cfg.line_y_tolerance < 0:
        parser.error("line-y-tolerance must be >= 0")

    if cfg.max_gap <= cfg.min_gap:
        parser.error(
            "max-gap must be greater than min-gap"
        )

    if cfg.detail_scale <= 0:
        parser.error("detail-scale must be > 0")

    if cfg.overlay_panel_width < 250:
        parser.error("panel-width must be >= 250")

    if (
        (cfg.expected_match_x is None)
        != (cfg.expected_match_y is None)
    ):
        parser.error(
            "expected-match-x and expected-match-y "
            "must be used together"
        )

    image_dir = Path(args.images)
    output_dir = Path(args.output)

    overlay_dir = output_dir / "overlay"
    detail_dir = output_dir / "detail"
    debug_dir = output_dir / "debug"

    for directory in (
        output_dir,
        overlay_dir,
        detail_dir,
        debug_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    references = load_references(
        args.template,
        args.dark_template,
        cfg,
    )

    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }
    )

    if not images:
        raise FileNotFoundError(
            f"No images found in: {image_dir}"
        )

    print("=" * 72)
    print("Phase B v5.1 - Enhanced Wafer Gap Overlay")
    print("=" * 72)
    print(f"Images          : {len(images)}")
    print(f"Normal template : {args.template}")
    print(f"Dark template   : {args.dark_template}")
    print(f"Search ROI      : {cfg.search_roi}")
    print(f"Measure ROI     : {cfg.measure_roi}")
    print(f"Normal X ratio  : {cfg.normal_x_ratio}")
    print(f"Dark X ratio    : {cfg.dark_x_ratio}")
    print(f"Detail scale    : {cfg.detail_scale}")
    print()

    rows = []

    for index, path in enumerate(images, start=1):
        image = cv2.imread(str(path))

        if image is None:
            rows.append({
                "image": path.name,
                "status": "image_read_failed",
            })
            print(
                f"[{index:02d}/{len(images):02d}] "
                f"{path.name}: image_read_failed"
            )
            continue

        try:
            result, debug = analyze_image(
                image, references, cfg
            )

            # Full-frame overlay + right info panel
            overlay = draw_overlay(
                image, result, debug, cfg
            )

            # Measurement ROI detail + right info panel
            detail = create_detail(
                image, result, debug, cfg
            )

            image_params = [
                cv2.IMWRITE_JPEG_QUALITY,
                cfg.jpeg_quality,
            ]

            cv2.imwrite(
                str(overlay_dir / f"{path.stem}.jpg"),
                overlay,
                image_params,
            )

            cv2.imwrite(
                str(detail_dir / f"{path.stem}.jpg"),
                detail,
                image_params,
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
            f"[{index:02d}/{len(images):02d}] "
            f"{path.name} | "
            f"Status={row['status']} | "
            f"Ref={row.get('match_reference')} | "
            f"Score={row.get('match_score')} | "
            f"Top={row.get('top_gap_px')} | "
            f"Bottom={row.get('bottom_gap_px')} | "
            f"Ratio={row.get('top_bottom_ratio_10')}"
        )

    df = pd.DataFrame(rows)

    csv_path = output_dir / "geometry_results.csv"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(df["status"].value_counts().to_string())

    print()
    print(f"CSV     : {csv_path}")
    print(f"Overlay : {overlay_dir}")
    print(f"Detail  : {detail_dir}")
    print(f"Debug   : {debug_dir}")


if __name__ == "__main__":
    main()
