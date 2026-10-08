
from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 0. Configuration
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

    # Normal template geometry
    normal_x_ratio: float = 0.7
    normal_top_ratio: float = 0.0
    normal_bottom_ratio: float = 1.0
    normal_top_offset: float = 0.0
    normal_bottom_offset: float = 0.0

    # Dark template geometry
    dark_x_ratio: float = 0.7
    dark_top_ratio: float = 0.0
    dark_bottom_ratio: float = 1.0
    dark_top_offset: float = 0.0
    dark_bottom_offset: float = 0.0

    # Wafer detection: unchanged from v5
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

    # Visualization
    detail_scale: float = 2.0
    panel_width: int = 340
    jpeg_quality: int = 98


@dataclass
class Reference:
    name: str
    image: np.ndarray

    x_ratio: float
    top_ratio: float
    bottom_ratio: float

    top_offset: float
    bottom_offset: float


# OpenCV BGR colors
COLOR_WAFER = (255, 135, 55)      # Blue
COLOR_BLADE = (45, 65, 235)       # Red
COLOR_GAP = (35, 45, 235)         # Red
COLOR_GUIDE = (155, 155, 155)     # Gray

COLOR_WHITE = (255, 255, 255)
COLOR_MUTED = (170, 180, 190)
COLOR_PANEL_BG = (33, 37, 43)


# ============================================================
# 1. Common utilities
# ============================================================

def odd(value):
    value = max(1, int(value))
    return value if value % 2 else value + 1


def safe_int(value):
    return int(round(float(value)))


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


def format_gap(value, decimals=1):
    if value is None:
        return "SKIP"
    return f"{value:.{decimals}f} px"


def format_ratio_10(top_gap, bottom_gap):
    """
    Example:
        19 : 28 -> 4 : 6

    Integer values always sum to 10.
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
# 2. References
# ============================================================

def load_references(normal_path, dark_path, cfg):
    normal_img = cv2.imread(str(normal_path))
    dark_img = cv2.imread(str(dark_path))

    if normal_img is None:
        raise FileNotFoundError(normal_path)

    if dark_img is None:
        raise FileNotFoundError(dark_path)

    return {
        "normal": Reference(
            name="normal",
            image=normal_img,
            x_ratio=cfg.normal_x_ratio,
            top_ratio=cfg.normal_top_ratio,
            bottom_ratio=cfg.normal_bottom_ratio,
            top_offset=cfg.normal_top_offset,
            bottom_offset=cfg.normal_bottom_offset,
        ),
        "dark": Reference(
            name="dark",
            image=dark_img,
            x_ratio=cfg.dark_x_ratio,
            top_ratio=cfg.dark_top_ratio,
            bottom_ratio=cfg.dark_bottom_ratio,
            top_offset=cfg.dark_top_offset,
            bottom_offset=cfg.dark_bottom_offset,
        ),
    }


# ============================================================
# 3. Template matching preprocessing
# ============================================================

def preprocess_matching(image, method, cfg):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    if method == "gray":
        return cv2.GaussianBlur(
            gray, (3, 3), 0
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
            enhanced, (3, 3), 0
        )

    raise ValueError(
        f"Unsupported matching method: {method}"
    )


# ============================================================
# 4. Template matching
# ============================================================

def match_template(
    search_gray,
    search_origin,
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

    result = {
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
        return result

    if float(np.std(target)) < 1e-5:
        return result

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

    _, score, _, location = cv2.minMaxLoc(
        response
    )

    match_x = search_origin[0] + location[0]
    match_y = search_origin[1] + location[1]

    position_error = None
    position_valid = True

    if (
        cfg.expected_match_x is not None
        and cfg.expected_match_y is not None
    ):
        position_error = float(np.hypot(
            match_x - cfg.expected_match_x,
            match_y - cfg.expected_match_y,
        ))

        position_valid = (
            position_error <= cfg.max_position_error
        )

    result.update({
        "score": float(score),

        "x": int(match_x),
        "y": int(match_y),

        "position_error": position_error,
        "position_valid": position_valid,
    })

    return result


def select_blade_match(image, references, cfg):
    search, bounds = crop_roi(
        image,
        cfg.search_roi,
    )

    origin = (bounds[0], bounds[1])

    # Ordered fallback: first passing stage wins.
    stages = [
        (
            "normal", "gray",
            cfg.normal_gray_threshold,
        ),
        (
            "dark", "gray",
            cfg.dark_gray_threshold,
        ),
        (
            "normal", "clahe",
            cfg.normal_clahe_threshold,
        ),
        (
            "dark", "clahe",
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
            search_gray=search_cache[method],
            search_origin=origin,
            reference=references[name],
            method=method,
            cfg=cfg,
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
            item
            for item in attempts
            if item["score"] is not None
            and item["position_valid"]
        ]

        if valid:
            selected = max(
                valid,
                key=lambda item: item["score"],
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
# 5. Blade anchors
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
            "top_y >= bottom_y"
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
# 6. Entire Measurement ROI
# ============================================================

def get_measurement_roi(image, anchors, cfg):
    roi, bounds = crop_roi(
        image,
        cfg.measure_roi,
    )

    x1, y1, x2, y2 = bounds

    # x_ratio defines the Blade anchor.
    # Wafer detection uses the entire ROI.
    if not (x1 <= anchors["x"] < x2):
        raise ValueError(
            f"Measurement X={anchors['x']:.1f} "
            f"outside ROI X=[{x1}, {x2})"
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


# ============================================================
# 7. Horizontal Wafer profile
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
        0,
        1,
        ksize=3,
    )

    abs_gy = np.abs(gy)

    # Reduce strong reflection dominance.
    clipped = np.minimum(
        abs_gy,
        120.0,
    )

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

    signed = smooth_1d(
        np.mean(gy, axis=1),
        cfg.profile_smooth,
    )

    return {
        "gray": gray,
        "gy": gy,

        "edge": edge_profile,
        "support": support,
        "signed": signed,
    }


# ============================================================
# 8. Horizontal support across ROI
# ============================================================

def calculate_horizontal_support(gy, y, cfg):
    height, width = gy.shape

    ya = max(
        0,
        y - cfg.line_y_tolerance,
    )

    yb = min(
        height,
        y + cfg.line_y_tolerance + 1,
    )

    region_y = np.abs(
        gy[ya:yb]
    )

    strip_width = min(
        cfg.strip_width,
        width,
    )

    positions = list(range(
        0,
        max(1, width - strip_width + 1),
        cfg.strip_step,
    ))

    last = max(
        0,
        width - strip_width,
    )

    if last not in positions:
        positions.append(last)

    positions = sorted(set(positions))

    valid_x = []

    for x1 in positions:
        x2 = min(
            width,
            x1 + strip_width,
        )

        strip = region_y[:, x1:x2]

        column_hit = np.any(
            strip >= cfg.wafer_support_gradient,
            axis=0,
        )

        ratio = float(
            np.mean(column_hit)
        )

        if ratio >= cfg.wafer_min_support:
            valid_x.append(
                (x1 + x2 - 1) / 2.0
            )

    total_strips = len(positions)
    valid_strips = len(valid_x)

    support_ratio = (
        valid_strips / total_strips
        if total_strips
        else 0.0
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

        "valid_strips": valid_strips,
        "total_strips": total_strips,

        "observed_x_min": xmin,
        "observed_x_max": xmax,
    }


# ============================================================
# 9. Wafer horizontal lines
# ============================================================

def detect_wafer_lines(profile, bounds, cfg):
    edge = profile["edge"]
    gy = profile["gy"]

    x_offset = bounds[0]
    y_offset = bounds[1]

    candidates = []

    for y in range(2, len(edge) - 2):
        strength = float(edge[y])

        if strength < cfg.wafer_edge_threshold:
            continue

        # Local maximum.
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

            "observed_x_min": (
                None
                if xmin is None
                else xmin + x_offset
            ),

            "observed_x_max": (
                None
                if xmax is None
                else xmax + x_offset
            ),
        })

    # Duplicate peak removal.
    candidates.sort(
        key=lambda item: item["strength"],
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
        key=lambda item: item["y_global"],
    )


# ============================================================
# 10. Upper / lower wafer detection
# ============================================================

def find_nearest_wafer(
    candidates,
    blade_y,
    side,
    cfg,
):
    valid = []

    minimum_gap = max(
        cfg.min_gap,
        cfg.blade_exclusion_px,
    )

    for candidate in candidates:
        wafer_y = candidate["y_global"]

        if side == "upper":
            gap = blade_y - wafer_y
        elif side == "lower":
            gap = wafer_y - blade_y
        else:
            raise ValueError(side)

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

    valid.sort(
        key=lambda item: item["gap_px"]
    )

    return valid[0], "detected", valid


# ============================================================
# 11. Measurement
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

    # Low confidence matching is not measured.
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

    top_gap = (
        upper["gap_px"]
        if upper is not None
        else None
    )

    bottom_gap = (
        lower["gap_px"]
        if lower is not None
        else None
    )

    top_ratio = None
    bottom_ratio = None

    if (
        top_gap is not None
        and bottom_gap is not None
    ):
        total = top_gap + bottom_gap

        if total > 0:
            top_ratio = 100.0 * top_gap / total
            bottom_ratio = 100.0 * bottom_gap / total

    if upper is not None and lower is not None:
        status = "ok"
    elif upper is not None or lower is not None:
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
# 12. Visualization helpers
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

        p1 = (
            safe_int(x1 + dx * distance),
            safe_int(y1 + dy * distance),
        )

        p2 = (
            safe_int(x1 + dx * end_distance),
            safe_int(y1 + dy * end_distance),
        )

        cv2.line(
            image,
            p1,
            p2,
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
    scale=1.0,
):
    """
    Thin vertical double-headed arrow.
    """
    x = safe_int(x)
    top = safe_int(min(y1, y2))
    bottom = safe_int(max(y1, y2))

    distance = bottom - top

    if distance < 3:
        return

    thickness = max(
        1,
        int(round(1.0 * scale)),
    )

    tip = max(
        2,
        min(
            safe_int(5 * scale),
            distance // 4,
        ),
    )

    cv2.line(
        image,
        (x, top),
        (x, bottom),
        COLOR_GAP,
        thickness,
        cv2.LINE_AA,
    )

    # Top arrowhead
    cv2.line(
        image,
        (x, top),
        (x - tip, top + tip),
        COLOR_GAP,
        thickness,
        cv2.LINE_AA,
    )

    cv2.line(
        image,
        (x, top),
        (x + tip, top + tip),
        COLOR_GAP,
        thickness,
        cv2.LINE_AA,
    )

    # Bottom arrowhead
    cv2.line(
        image,
        (x, bottom),
        (x - tip, bottom - tip),
        COLOR_GAP,
        thickness,
        cv2.LINE_AA,
    )

    cv2.line(
        image,
        (x, bottom),
        (x + tip, bottom - tip),
        COLOR_GAP,
        thickness,
        cv2.LINE_AA,
    )


def draw_pill(
    image,
    text,
    center_x,
    center_y,
    scale=1.0,
):
    """
    Red rounded pill / white text.

    Rendering occurs after detail enlargement,
    so text remains sharp.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.40 * scale
    thickness = max(
        1,
        int(round(0.65 * scale)),
    )

    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    pad_x = max(3, safe_int(8 * scale))
    pad_y = max(2, safe_int(5 * scale))

    pill_w = text_w + pad_x * 2
    pill_h = text_h + baseline + pad_y * 2

    image_h, image_w = image.shape[:2]

    if pill_w >= image_w or pill_h >= image_h:
        return

    left = safe_int(
        center_x - pill_w / 2
    )
    top = safe_int(
        center_y - pill_h / 2
    )

    left = max(
        1,
        min(left, image_w - pill_w - 1),
    )

    top = max(
        1,
        min(top, image_h - pill_h - 1),
    )

    patch = image[
        top:top + pill_h,
        left:left + pill_w,
    ]

    mask = np.zeros(
        (pill_h, pill_w),
        dtype=np.uint8,
    )

    radius = min(
        safe_int(9 * scale),
        pill_h // 2,
        pill_w // 2,
    )

    # Rounded rectangle mask.
    cv2.rectangle(
        mask,
        (radius, 0),
        (pill_w - radius - 1, pill_h - 1),
        255,
        -1,
    )

    cv2.rectangle(
        mask,
        (0, radius),
        (pill_w - 1, pill_h - radius - 1),
        255,
        -1,
    )

    for cx, cy in (
        (radius, radius),
        (pill_w - radius - 1, radius),
        (radius, pill_h - radius - 1),
        (pill_w - radius - 1, pill_h - radius - 1),
    ):
        cv2.circle(
            mask,
            (cx, cy),
            radius,
            255,
            -1,
            cv2.LINE_AA,
        )

    # Blend only masked pixels.
    alpha = (
        mask.astype(np.float32) / 255.0
        * 0.94
    )[:, :, None]

    bg = np.empty_like(patch)
    bg[:] = COLOR_GAP

    blended = (
        bg.astype(np.float32) * alpha
        + patch.astype(np.float32) * (1.0 - alpha)
    )

    patch[:] = np.uint8(
        np.clip(blended, 0, 255)
    )

    cv2.putText(
        image,
        text,
        (
            left + pad_x,
            top + pad_y + text_h,
        ),
        font,
        font_scale,
        COLOR_WHITE,
        thickness,
        cv2.LINE_AA,
    )


def draw_wafer_label(
    image,
    text,
    right_x,
    wafer_y,
    above,
    scale=1.0,
):
    """
    Normal thin font:
    - no shadow
    - no thick outline
    - no background box
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.45 * scale
    thickness = max(
        1,
        int(round(0.55 * scale)),
    )

    (text_w, text_h), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    margin_x = safe_int(8 * scale)
    margin_y = safe_int(7 * scale)

    x = safe_int(
        right_x - text_w - margin_x
    )

    if above:
        y = safe_int(
            wafer_y - margin_y
        )
    else:
        y = safe_int(
            wafer_y + text_h + margin_y
        )

    x = max(
        2,
        min(x, image.shape[1] - text_w - 2),
    )

    y = max(
        text_h + 2,
        min(y, image.shape[0] - 2),
    )

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        COLOR_WAFER,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# 13. Shared annotation renderer
# ============================================================

def draw_annotation(
    image,
    result,
    debug,
    cfg,
    origin=(0, 0),
    scale=1.0,
):
    """
    Draw all annotations in the current image space.

    Global measurement coordinates are converted by:

        local_x = (global_x - origin_x) * scale
        local_y = (global_y - origin_y) * scale

    Overlay:
        origin=(0,0), scale=1

    Detail:
        origin=(ROI_x, ROI_y), scale=detail_scale

    This allows annotations to be rendered AFTER
    resizing the original measurement ROI.
    """

    out = image.copy()

    anchors = result["anchors"]

    if (
        anchors is None
        or "bounds" not in debug
    ):
        return out

    origin_x, origin_y = origin

    def X(global_x):
        return safe_int(
            (global_x - origin_x) * scale
        )

    def Y(global_y):
        return safe_int(
            (global_y - origin_y) * scale
        )

    x1, y1, x2, y2 = debug["bounds"]

    left = X(x1)
    right = X(x2 - 1)

    top = Y(y1)
    bottom = Y(y2 - 1)

    measure_x = X(
        anchors["x"]
    )

    blade_top = Y(
        anchors["top_y"]
    )

    blade_bottom = Y(
        anchors["bottom_y"]
    )

    # --------------------------------------------------------
    # 1) Gray dashed measurement line
    # --------------------------------------------------------

    draw_dashed_line(
        out,
        (measure_x, top),
        (measure_x, bottom),
        COLOR_GUIDE,
        thickness=1,
        dash_length=max(4, safe_int(8 * scale)),
        gap_length=max(3, safe_int(6 * scale)),
    )

    # --------------------------------------------------------
    # 2) Thin red dashed Blade top / bottom
    # --------------------------------------------------------

    for yy in (blade_top, blade_bottom):
        draw_dashed_line(
            out,
            (left, yy),
            (right, yy),
            COLOR_BLADE,
            thickness=1,
            dash_length=max(4, safe_int(7 * scale)),
            gap_length=max(3, safe_int(5 * scale)),
        )

    # --------------------------------------------------------
    # 3) Blue wafer lines and thin labels
    # --------------------------------------------------------

    upper = result["upper"]
    lower = result["lower"]

    wafer_thickness = max(
        1,
        safe_int(1.3 * scale),
    )

    if upper is not None:
        upper_y = Y(
            upper["y_global"]
        )

        cv2.line(
            out,
            (left, upper_y),
            (right, upper_y),
            COLOR_WAFER,
            wafer_thickness,
            cv2.LINE_AA,
        )

        draw_wafer_label(
            out,
            text="Top Wafer",
            right_x=right,
            wafer_y=upper_y,
            above=True,
            scale=scale,
        )

    if lower is not None:
        lower_y = Y(
            lower["y_global"]
        )

        cv2.line(
            out,
            (left, lower_y),
            (right, lower_y),
            COLOR_WAFER,
            wafer_thickness,
            cv2.LINE_AA,
        )

        draw_wafer_label(
            out,
            text="Lower Wafer",
            right_x=right,
            wafer_y=lower_y,
            above=False,
            scale=scale,
        )

    # --------------------------------------------------------
    # 4) Red thin arrows and red pill values
    # --------------------------------------------------------

    arrow_x = (
        measure_x
        + safe_int(16 * scale)
    )

    arrow_x = min(
        arrow_x,
        right - safe_int(65 * scale),
    )

    arrow_x = max(
        left + safe_int(12 * scale),
        arrow_x,
    )

    pill_offset = safe_int(
        46 * scale
    )

    if upper is not None:
        upper_y = Y(
            upper["y_global"]
        )

        draw_double_arrow(
            out,
            x=arrow_x,
            y1=upper_y,
            y2=blade_top,
            scale=scale,
        )

        draw_pill(
            out,
            text=f"{result['top_gap_px']:.0f} px",
            center_x=arrow_x + pill_offset,
            center_y=(upper_y + blade_top) / 2,
            scale=scale,
        )

    if lower is not None:
        lower_y = Y(
            lower["y_global"]
        )

        draw_double_arrow(
            out,
            x=arrow_x,
            y1=blade_bottom,
            y2=lower_y,
            scale=scale,
        )

        draw_pill(
            out,
            text=f"{result['bottom_gap_px']:.0f} px",
            center_x=arrow_x + pill_offset,
            center_y=(blade_bottom + lower_y) / 2,
            scale=scale,
        )

    return out


# ============================================================
# 14. Detail information panel
# ============================================================

def draw_info_panel(result, height, width=340):
    """
    Only five fields:
        Status
        Top Gap
        Bottom Gap
        Diff
        Top:Bottom
    """
    height = max(
        250,
        int(height),
    )

    width = max(
        250,
        int(width),
    )

    panel = np.full(
        (height, width, 3),
        COLOR_PANEL_BG,
        dtype=np.uint8,
    )

    top_gap = result["top_gap_px"]
    bottom_gap = result["bottom_gap_px"]

    if (
        top_gap is not None
        and bottom_gap is not None
    ):
        diff_text = (
            f"{abs(top_gap - bottom_gap):.1f} px"
        )
    else:
        diff_text = "SKIP"

    rows = [
        ("Status", str(result["status"])),
        ("Top Gap", format_gap(top_gap)),
        ("Bottom Gap", format_gap(bottom_gap)),
        ("Diff", diff_text),
        (
            "Top:Bottom",
            format_ratio_10(top_gap, bottom_gap),
        ),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        panel,
        "MEASUREMENT",
        (24, 46),
        font,
        0.72,
        COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )

    cv2.line(
        panel,
        (24, 66),
        (width - 24, 66),
        (85, 95, 105),
        1,
        cv2.LINE_AA,
    )

    start_y = 105
    row_gap = 55

    for i, (label, value) in enumerate(rows):
        y = start_y + i * row_gap

        if y + 10 >= height:
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

        font_scale = 0.61
        value_thickness = 2

        (text_w, _), _ = cv2.getTextSize(
            value,
            font,
            font_scale,
            value_thickness,
        )

        value_x = max(
            125,
            width - text_w - 24,
        )

        value_color = (
            (115, 220, 255)
            if label == "Top:Bottom"
            else COLOR_WHITE
        )

        cv2.putText(
            panel,
            value,
            (value_x, y),
            font,
            font_scale,
            value_color,
            value_thickness,
            cv2.LINE_AA,
        )

        if i < len(rows) - 1:
            cv2.line(
                panel,
                (24, y + 19),
                (width - 24, y + 19),
                (58, 63, 70),
                1,
                cv2.LINE_AA,
            )

    return panel


# ============================================================
# 15. Full-frame overlay
# ============================================================

def draw_overlay(image, result, debug, cfg):
    """
    No information panel.

    Output dimensions are identical to original image.
    """
    return draw_annotation(
        image,
        result,
        debug,
        cfg,
        origin=(0, 0),
        scale=1.0,
    )


# ============================================================
# 16. High-quality detail
# ============================================================

def create_detail(image, result, debug, cfg):
    """
    Critical rendering order:

    1) Crop the ORIGINAL frame.
    2) Resize the raw ROI.
    3) Render lines, arrows and text at enlarged resolution.
    4) Append right information panel.

    Never crop from an already annotated overlay.
    """

    raw_roi, bounds = crop_roi(
        image,
        cfg.measure_roi,
    )

    roi_x, roi_y = bounds[:2]

    scale = cfg.detail_scale

    interpolation = (
        cv2.INTER_CUBIC
        if scale > 1
        else cv2.INTER_AREA
    )

    enlarged = cv2.resize(
        raw_roi,
        None,
        fx=scale,
        fy=scale,
        interpolation=interpolation,
    )

    annotated = draw_annotation(
        enlarged,
        result,
        debug,
        cfg,
        origin=(roi_x, roi_y),
        scale=scale,
    )

    panel = draw_info_panel(
        result,
        height=annotated.shape[0],
        width=cfg.panel_width,
    )

    return np.hstack([
        annotated,
        panel,
    ])


# ============================================================
# 17. Debug saving
# ============================================================

def save_debug(debug_dir, stem, debug, result):
    attempts = debug.get(
        "attempts", []
    )

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

    candidate_columns = [
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
    ]

    pd.DataFrame(
        debug["candidates"],
        columns=candidate_columns,
    ).to_csv(
        debug_dir / f"{stem}_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Sobel visualization.
    gy = np.abs(
        profile["gy"]
    )

    p99 = max(
        1.0,
        float(np.percentile(gy, 99)),
    )

    sobel_image = np.uint8(
        np.clip(
            gy * 255.0 / p99,
            0,
            255,
        )
    )

    cv2.imwrite(
        str(debug_dir / f"{stem}_sobel.png"),
        sobel_image,
    )

    # Horizontal line candidates.
    line_image = cv2.cvtColor(
        profile["gray"],
        cv2.COLOR_GRAY2BGR,
    )

    for candidate in debug["candidates"]:
        yy = safe_int(
            candidate["y_local"]
        )

        cv2.line(
            line_image,
            (0, yy),
            (line_image.shape[1] - 1, yy),
            (0, 255, 255),
            1,
        )

    for key in ("upper", "lower"):
        wafer = result[key]

        if wafer is None:
            continue

        yy = safe_int(
            wafer["y_global"] - y0
        )

        cv2.line(
            line_image,
            (0, yy),
            (line_image.shape[1] - 1, yy),
            COLOR_WAFER,
            2,
        )

    cv2.imwrite(
        str(
            debug_dir
            / f"{stem}_horizontal_lines.png"
        ),
        line_image,
    )


# ============================================================
# 18. Results serialization
# ============================================================

def flatten_result(filename, result):
    match = result["match"]

    anchors = result["anchors"]
    upper = result["upper"]
    lower = result["lower"]

    def get(obj, key):
        if obj is None:
            return None
        return obj.get(key)

    top_gap = result["top_gap_px"]
    bottom_gap = result["bottom_gap_px"]

    gap_diff = (
        abs(top_gap - bottom_gap)
        if top_gap is not None
        and bottom_gap is not None
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

        "blade_top_y": get(
            anchors, "top_y"
        ),
        "blade_bottom_y": get(
            anchors, "bottom_y"
        ),
        "blade_thickness_px": get(
            anchors, "thickness"
        ),

        "upper_wafer_y": get(
            upper, "y_global"
        ),
        "lower_wafer_y": get(
            lower, "y_global"
        ),

        "upper_status": result["upper_status"],
        "lower_status": result["lower_status"],

        "upper_support": get(
            upper, "horizontal_support"
        ),
        "lower_support": get(
            lower, "horizontal_support"
        ),

        "upper_span": get(
            upper, "span_ratio"
        ),
        "lower_span": get(
            lower, "span_ratio"
        ),

        "top_gap_px": top_gap,
        "bottom_gap_px": bottom_gap,
        "gap_diff_px": gap_diff,

        "top_ratio": result["top_ratio"],
        "bottom_ratio": result["bottom_ratio"],

        "top_bottom_ratio_10": format_ratio_10(
            top_gap,
            bottom_gap,
        ),
    }


# ============================================================
# 19. CLI arguments
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Phase B v5.2: Dual Template Matching + "
            "Full ROI Wafer Detection + Refined Overlay"
        )
    )

    # Input / output
    parser.add_argument(
        "--images",
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--dark-template",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="output/06_phase_b_v5_2",
    )

    # ROI
    parser.add_argument(
        "--search-roi",
        type=int,
        nargs=4,
        default=[41, 358, 2027, 186],
    )

    parser.add_argument(
        "--measure-roi",
        type=int,
        nargs=4,
        default=[49, 296, 600, 273],
    )

    # Matching
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

    # Normal template geometry
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

    # Dark template geometry
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

    # Wafer detection
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

    # Visualization
    parser.add_argument(
        "--detail-scale",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--panel-width",
        type=int,
        default=340,
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=98,
    )

    return parser


# ============================================================
# 20. Config construction
# ============================================================

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
        panel_width=args.panel_width,
        jpeg_quality=args.jpeg_quality,
    )


# ============================================================
# 21. Validation
# ============================================================

def validate_config(cfg, parser):
    fraction_fields = (
        "normal_x_ratio",
        "dark_x_ratio",
        "normal_top_ratio",
        "normal_bottom_ratio",
        "dark_top_ratio",
        "dark_bottom_ratio",
        "wafer_min_support",
        "wafer_min_span_ratio",
    )

    for name in fraction_fields:
        value = getattr(cfg, name)

        if not 0 <= value <= 1:
            parser.error(
                f"{name} must be between 0 and 1"
            )

    if cfg.strip_width < 1 or cfg.strip_step < 1:
        parser.error(
            "strip-width and strip-step must be >= 1"
        )

    if cfg.wafer_min_valid_strips < 1:
        parser.error(
            "wafer-min-strips must be >= 1"
        )

    if cfg.line_y_tolerance < 0:
        parser.error(
            "line-y-tolerance must be >= 0"
        )

    if cfg.max_gap <= cfg.min_gap:
        parser.error(
            "max-gap must be greater than min-gap"
        )

    if cfg.detail_scale <= 0:
        parser.error(
            "detail-scale must be > 0"
        )

    if cfg.panel_width < 250:
        parser.error(
            "panel-width must be >= 250"
        )

    if not 1 <= cfg.jpeg_quality <= 100:
        parser.error(
            "jpeg-quality must be 1~100"
        )

    if (
        (cfg.expected_match_x is None)
        != (cfg.expected_match_y is None)
    ):
        parser.error(
            "expected-match-x and expected-match-y "
            "must be supplied together"
        )


# ============================================================
# 22. Main
# ============================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = config_from_args(args)
    validate_config(cfg, parser)

    image_dir = Path(args.images)
    output_dir = Path(args.output)

    overlay_dir = output_dir / "overlay"
    detail_dir = output_dir / "detail"
    debug_dir = output_dir / "debug"

    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)

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

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }
    )

    if not images:
        raise FileNotFoundError(
            f"No images found: {image_dir}"
        )

    print("=" * 72)
    print("Phase B v5.2 - Refined Wafer Gap Visualization")
    print("=" * 72)
    print(f"Images          : {len(images)}")
    print(f"Normal template : {args.template}")
    print(f"Dark template   : {args.dark_template}")
    print(f"Search ROI      : {cfg.search_roi}")
    print(f"Measure ROI     : {cfg.measure_roi}")
    print(f"Normal X ratio  : {cfg.normal_x_ratio}")
    print(f"Dark X ratio    : {cfg.dark_x_ratio}")
    print(f"Detail scale    : {cfg.detail_scale}")
    print(f"JPEG quality    : {cfg.jpeg_quality}")
    print()

    rows = []

    for index, path in enumerate(
        images,
        start=1,
    ):
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
                image,
                references,
                cfg,
            )

            # Full original resolution.
            # No info panel.
            overlay = draw_overlay(
                image,
                result,
                debug,
                cfg,
            )

            # Crop original ROI first, enlarge,
            # then draw annotations and append panel.
            detail = create_detail(
                image,
                result,
                debug,
                cfg,
            )

            save_params = [
                cv2.IMWRITE_JPEG_QUALITY,
                cfg.jpeg_quality,
            ]

            overlay_path = (
                overlay_dir / f"{path.stem}.jpg"
            )

            detail_path = (
                detail_dir / f"{path.stem}.jpg"
            )

            if not cv2.imwrite(
                str(overlay_path),
                overlay,
                save_params,
            ):
                raise IOError(
                    f"Failed to save: {overlay_path}"
                )

            if not cv2.imwrite(
                str(detail_path),
                detail,
                save_params,
            ):
                raise IOError(
                    f"Failed to save: {detail_path}"
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
            f"Method={row.get('match_method')} | "
            f"Score={row.get('match_score')} | "
            f"Top={row.get('top_gap_px')} | "
            f"Bottom={row.get('bottom_gap_px')} | "
            f"Ratio={row.get('top_bottom_ratio_10')}"
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
    print("Result Summary")
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
