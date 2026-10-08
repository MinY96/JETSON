"""
python phase_b_template_matching_v4.py ^
  --images output\05_phase_a_v3\measurement_frames ^
  --template references\blade_left.png ^
  --dark-template references\blade_left_dark.png ^
  --output output\06_phase_b_v4 ^
  --search-roi 41 358 2027 186 ^
  --measure-roi 49 296 381 273
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
    search_roi: tuple = (41, 358, 2027, 186)
    measure_roi: tuple = (49, 296, 381, 273)

    # 각 매칭 단계의 통과 기준
    gray_threshold: float = 0.65
    dark_gray_threshold: float = 0.65
    clahe_threshold: float = 0.65
    dark_clahe_threshold: float = 0.65

    # 통과 실패 시 REVIEW 후보를 남기는 기준
    review_threshold: float = 0.30

    # CLAHE
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    # 예상 매칭 좌표: 미설정이면 위치 제약 없음
    expected_match_x: float | None = None
    expected_match_y: float | None = None
    max_position_error: float = 30.0

    # Normal reference의 측정 기준
    normal_x_ratio: float = 0.5
    normal_top_ratio: float = 0.0
    normal_bottom_ratio: float = 1.0
    normal_top_offset: float = 0.0
    normal_bottom_offset: float = 0.0

    # Dark reference의 측정 기준
    dark_x_ratio: float = 0.5
    dark_top_ratio: float = 0.0
    dark_bottom_ratio: float = 1.0
    dark_top_offset: float = 0.0
    dark_bottom_offset: float = 0.0

    # 실제 간격 측정 영역
    measure_band_width: int = 60

    # Wafer 에지 검출
    min_gap: float = 5.0
    max_gap: float = 100.0
    exclusion_px: float = 4.0

    blur_ksize: int = 3
    profile_smooth: int = 3
    wafer_edge_threshold: float = 18.0
    wafer_min_support: float = 0.35
    wafer_support_gradient: float = 15.0
    wafer_min_prominence: float = 5.0
    wafer_max_y_std: float = 4.0
    edge_min_distance: int = 3

    jpeg_quality: int = 95


# ============================================================
# Common utilities
# ============================================================

def odd(n):
    n = max(1, int(n))
    return n if n % 2 else n + 1


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
            f"ROI outside image: {roi}, image={iw}x{ih}"
        )

    return (
        image[y1:y2, x1:x2],
        (x1, y1, x2, y2),
    )


def smooth_1d(values, kernel):
    values = np.asarray(values, dtype=np.float32)

    if kernel <= 1:
        return values

    return cv2.GaussianBlur(
        values.reshape(-1, 1),
        (1, odd(kernel)),
        0,
    ).ravel()


# ============================================================
# Reference configuration
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


def load_references(normal_path, dark_path, cfg):
    normal_image = cv2.imread(str(normal_path))
    dark_image = cv2.imread(str(dark_path))

    if normal_image is None:
        raise FileNotFoundError(normal_path)

    if dark_image is None:
        raise FileNotFoundError(dark_path)

    return {
        "normal": Reference(
            name="normal",
            image=normal_image,
            x_ratio=cfg.normal_x_ratio,
            top_ratio=cfg.normal_top_ratio,
            bottom_ratio=cfg.normal_bottom_ratio,
            top_offset=cfg.normal_top_offset,
            bottom_offset=cfg.normal_bottom_offset,
        ),
        "dark": Reference(
            name="dark",
            image=dark_image,
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

def preprocess_match(image, method, cfg):
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

    raise ValueError(f"Unknown matching method: {method}")


# ============================================================
# Template matching
# ============================================================

def match_one(
    search_gray,
    search_origin,
    reference,
    method,
    cfg,
):
    target = preprocess_match(
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
            "status": "template_too_large",
            "score": None,
            "x": None,
            "y": None,
            "w": tw,
            "h": th,
            "position_error": None,
            "position_valid": False,
        }

    if float(target.std()) < 1e-5:
        return {
            "reference": reference.name,
            "method": method,
            "status": "low_template_variance",
            "score": None,
            "x": None,
            "y": None,
            "w": tw,
            "h": th,
            "position_error": None,
            "position_valid": False,
        }

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

    _, score, _, location = cv2.minMaxLoc(response)

    x = search_origin[0] + location[0]
    y = search_origin[1] + location[1]

    # 예상 좌표가 지정되어 있을 때만 위치 검증
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
    else:
        position_error = None
        position_valid = True

    return {
        "reference": reference.name,
        "method": method,
        "status": "candidate",
        "score": float(score),
        "x": int(x),
        "y": int(y),
        "w": int(tw),
        "h": int(th),
        "position_error": position_error,
        "position_valid": bool(position_valid),
    }


def select_blade_match(image, references, cfg):
    """
    순차 매칭:

      1. normal + gray
      2. dark   + gray
      3. normal + clahe
      4. dark   + clahe

    매 단계에서 score와 위치 조건을 모두 충족하면
    즉시 채택한다.

    전부 실패하면 최고 점수 후보를 REVIEW로 보관.
    REVIEW는 자동 간격 측정하지 않는다.
    """
    search, bounds = crop_roi(
        image,
        cfg.search_roi,
    )

    x0, y0, _, _ = bounds
    origin = (x0, y0)

    stages = [
        ("normal", "gray", cfg.gray_threshold),
        ("dark", "gray", cfg.dark_gray_threshold),
        ("normal", "clahe", cfg.clahe_threshold),
        ("dark", "clahe", cfg.dark_clahe_threshold),
    ]

    # Search ROI는 동일하므로 전처리 캐싱
    search_cache = {}

    attempts = []
    selected = None

    for stage_num, (ref_name, method, threshold) in enumerate(
        stages, start=1
    ):
        if method not in search_cache:
            search_cache[method] = preprocess_match(
                search,
                method,
                cfg,
            )

        candidate = match_one(
            search_gray=search_cache[method],
            search_origin=origin,
            reference=references[ref_name],
            method=method,
            cfg=cfg,
        )

        candidate["stage"] = stage_num
        candidate["threshold"] = threshold

        score = candidate["score"]

        passed = (
            score is not None
            and score >= threshold
            and candidate["position_valid"]
        )

        candidate["passed"] = bool(passed)

        attempts.append(candidate)

        if passed:
            selected = candidate.copy()
            selected["status"] = "matched"
            break

    if selected is None:
        valid_candidates = [
            a for a in attempts
            if a["score"] is not None
            and a["position_valid"]
        ]

        if valid_candidates:
            best = max(
                valid_candidates,
                key=lambda a: a["score"],
            )

            selected = best.copy()
            selected["status"] = (
                "review"
                if best["score"] >= cfg.review_threshold
                else "match_failed"
            )
        else:
            selected = {
                "status": "match_failed",
                "reference": None,
                "method": None,
                "stage": None,
                "score": None,
                "x": None,
                "y": None,
                "w": None,
                "h": None,
                "position_error": None,
            }

    return selected, attempts


# ============================================================
# Blade anchors
# ============================================================

def calculate_blade_anchors(match, references):
    reference = references[match["reference"]]

    x = (
        match["x"]
        + reference.x_ratio * match["w"]
    )

    top_y = (
        match["y"]
        + reference.top_ratio * match["h"]
        + reference.top_offset
    )

    bottom_y = (
        match["y"]
        + reference.bottom_ratio * match["h"]
        + reference.bottom_offset
    )

    if top_y >= bottom_y:
        raise ValueError(
            "Blade top must be above blade bottom. "
            "Check reference-specific offsets."
        )

    return {
        "x": float(x),
        "top_y": float(top_y),
        "bottom_y": float(bottom_y),
        "thickness": float(bottom_y - top_y),
    }


# ============================================================
# Measurement band
# ============================================================

def get_measurement_band(image, anchors, cfg):
    _, bounds = crop_roi(
        image,
        cfg.measure_roi,
    )

    mx1, my1, mx2, my2 = bounds

    center_x = anchors["x"]
    half = cfg.measure_band_width / 2.0

    x1 = max(
        mx1,
        int(round(center_x - half)),
    )

    x2 = min(
        mx2,
        int(round(center_x + half)),
    )

    if x2 - x1 < 8:
        raise ValueError(
            "Measurement band does not overlap measurement ROI. "
            "Adjust reference X ratio, band width or ROI."
        )

    return (
        image[my1:my2, x1:x2],
        (x1, my1, x2, my2),
    )


# ============================================================
# Wafer edge extraction (Phase B v3 기반)
# ============================================================

def calculate_wafer_profile(band, cfg):
    gray = cv2.cvtColor(
        band,
        cv2.COLOR_BGR2GRAY,
    )

    if cfg.blur_ksize > 1:
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

    edge = smooth_1d(
        np.median(np.abs(gy), axis=1),
        cfg.profile_smooth,
    )

    signed = smooth_1d(
        np.median(gy, axis=1),
        cfg.profile_smooth,
    )

    support = smooth_1d(
        np.mean(
            np.abs(gy) >= cfg.wafer_support_gradient,
            axis=1,
        ),
        cfg.profile_smooth,
    )

    return {
        "gray": gray,
        "gy": gy,
        "edge": edge,
        "signed": signed,
        "support": support,
    }


def detect_edge_candidates(profile, bounds, cfg):
    y0 = bounds[1]

    edge = profile["edge"]
    support = profile["support"]
    gy = profile["gy"]

    candidates = []

    for y in range(1, len(edge) - 1):
        strength = float(edge[y])

        if strength < cfg.wafer_edge_threshold:
            continue

        if not (
            strength >= edge[y - 1]
            and strength > edge[y + 1]
        ):
            continue

        left = max(0, y - 3)
        right = min(len(edge), y + 4)

        prominence = strength - float(
            np.min(edge[left:right])
        )

        if prominence < cfg.wafer_min_prominence:
            continue

        line_support = float(support[y])

        if line_support < cfg.wafer_min_support:
            continue

        # X별 edge Y 위치 분산 확인
        y1 = max(0, y - 3)
        y2 = min(gy.shape[0], y + 4)

        local_grad = np.abs(
            gy[y1:y2, :]
        )

        column_max = np.max(
            local_grad,
            axis=0,
        )

        valid_cols = (
            column_max >= cfg.wafer_support_gradient
        )

        if np.count_nonzero(valid_cols) < 3:
            continue

        column_ys = (
            np.argmax(local_grad, axis=0)[valid_cols]
            + y1
        )

        y_std = float(np.std(column_ys))

        if y_std > cfg.wafer_max_y_std:
            continue

        candidates.append({
            "y_local": int(y),
            "y_global": int(y + y0),
            "strength": strength,
            "prominence": prominence,
            "support": line_support,
            "y_std": y_std,
        })

    # 가까운 중복 peak 제거
    candidates.sort(
        key=lambda c: c["strength"],
        reverse=True,
    )

    selected = []

    for candidate in candidates:
        if any(
            abs(candidate["y_local"] - prev["y_local"])
            < cfg.edge_min_distance
            for prev in selected
        ):
            continue

        selected.append(candidate)

    return sorted(
        selected,
        key=lambda c: c["y_global"],
    )


def find_nearest_wafer(candidates, blade_y, side, cfg):
    valid = []

    for candidate in candidates:
        wafer_y = candidate["y_global"]

        if side == "upper":
            gap = blade_y - wafer_y
        else:
            gap = wafer_y - blade_y

        if not (
            max(cfg.min_gap, cfg.exclusion_px)
            <= gap <= cfg.max_gap
        ):
            continue

        valid.append({
            **candidate,
            "gap_px": float(gap),
        })

    if not valid:
        return None, "not_detected", []

    valid.sort(
        key=lambda c: c["gap_px"]
    )

    return valid[0], "detected", valid


# ============================================================
# Geometry measurement
# ============================================================

def analyze_image(image, references, cfg):
    match, attempts = select_blade_match(
        image,
        references,
        cfg,
    )

    result = {
        "status": "",
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

    # REVIEW는 위치 표시만 하고 자동 측정하지 않음
    if match["status"] != "matched":
        result["status"] = match["status"]
        return result, debug

    anchors = calculate_blade_anchors(
        match,
        references,
    )

    result["anchors"] = anchors

    _, measure_bounds = crop_roi(
        image,
        cfg.measure_roi,
    )

    _, my1, _, my2 = measure_bounds

    if not (
        my1 <= anchors["top_y"]
        < anchors["bottom_y"]
        < my2
    ):
        result["status"] = "blade_outside_measure_roi"
        return result, debug

    band, bounds = get_measurement_band(
        image,
        anchors,
        cfg,
    )

    profile = calculate_wafer_profile(
        band,
        cfg,
    )

    candidates = detect_edge_candidates(
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
            top_ratio = 100.0 * top_gap / total
            bottom_ratio = 100.0 * bottom_gap / total

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
        "band": band,
        "bounds": bounds,
        "profile": profile,
        "candidates": candidates,
        "upper_candidates": upper_candidates,
        "lower_candidates": lower_candidates,
    })

    return result, debug


# ============================================================
# Overlay
# ============================================================

def draw_overlay(image, result, debug, cfg):
    out = image.copy()

    # Search ROI
    sx, sy, sw, sh = cfg.search_roi
    cv2.rectangle(
        out,
        (sx, sy),
        (sx + sw, sy + sh),
        (255, 140, 0),
        2,
    )

    # Measurement ROI
    mx, my, mw, mh = cfg.measure_roi
    cv2.rectangle(
        out,
        (mx, my),
        (mx + mw, my + mh),
        (0, 255, 255),
        2,
    )

    match = result["match"]

    if match["x"] is not None:
        bx = match["x"]
        by = match["y"]
        bw = match["w"]
        bh = match["h"]

        color = (
            (0, 255, 0)
            if match["status"] == "matched"
            else (0, 0, 255)
        )

        cv2.rectangle(
            out,
            (bx, by),
            (bx + bw, by + bh),
            color,
            2,
        )

    anchors = result["anchors"]

    if anchors is None:
        return out

    center_x = int(round(anchors["x"]))
    top_y = int(round(anchors["top_y"]))
    bottom_y = int(round(anchors["bottom_y"]))

    if "bounds" in debug:
        x1, _, x2, _ = debug["bounds"]
    else:
        x1 = center_x - 20
        x2 = center_x + 20

    # Blade 상하단
    for yy in [top_y, bottom_y]:
        cv2.line(
            out,
            (x1, yy),
            (x2, yy),
            (255, 0, 0),
            2,
        )

    # Upper wafer
    if result["upper"] is not None:
        yy = result["upper"]["y_global"]

        cv2.line(
            out, (x1, yy), (x2, yy),
            (0, 255, 0), 2
        )

        cv2.line(
            out, (center_x, yy),
            (center_x, top_y),
            (0, 255, 0), 2
        )

    # Lower wafer
    if result["lower"] is not None:
        yy = result["lower"]["y_global"]

        cv2.line(
            out, (x1, yy), (x2, yy),
            (0, 165, 255), 2
        )

        cv2.line(
            out, (center_x, bottom_y),
            (center_x, yy),
            (0, 165, 255), 2
        )

    return out


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
        (zoom.shape[0], 470, 3),
        30,
        dtype=np.uint8,
    )

    match = result["match"]

    def fmt(value, digits=2):
        if value is None:
            return "SKIP"
        return f"{value:.{digits}f}"

    lines = [
        f"Status: {result['status']}",
        f"Reference: {match['reference']}",
        f"Method: {match['method']}",
        f"Stage: {match['stage']}",
        f"Match Score: {fmt(match['score'], 4)}",
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

    for i, line in enumerate(lines):
        y = 30 + i * 33

        if y >= panel.shape[0]:
            break

        cv2.putText(
            panel,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return np.hstack([zoom, panel])


# ============================================================
# Debug outputs
# ============================================================

def save_debug(debug_dir, stem, debug):
    attempts = debug.get("attempts", [])

    # 각 Reference / 전처리 매칭 기록
    pd.DataFrame(attempts).to_csv(
        debug_dir / f"{stem}_matching.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if "profile" not in debug:
        return

    profile = debug["profile"]
    y0 = debug["bounds"][1]

    rows = []

    for y in range(len(profile["edge"])):
        rows.append({
            "y_local": y,
            "y_global": y + y0,
            "edge_strength": float(profile["edge"][y]),
            "signed_sobel": float(profile["signed"][y]),
            "edge_support": float(profile["support"][y]),
        })

    pd.DataFrame(rows).to_csv(
        debug_dir / f"{stem}_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        debug["candidates"],
        columns=[
            "y_local", "y_global",
            "strength", "prominence",
            "support", "y_std",
        ],
    ).to_csv(
        debug_dir / f"{stem}_edge_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gy = np.abs(profile["gy"])

    p99 = max(1.0, float(np.percentile(gy, 99)))

    sobel = np.uint8(
        np.clip(gy * 255.0 / p99, 0, 255)
    )

    cv2.imwrite(
        str(debug_dir / f"{stem}_sobel.png"),
        sobel,
    )


# ============================================================
# CSV result
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
        "match_position_error": match["position_error"],

        "measure_x": get(anchors, "x"),
        "blade_top_y": get(anchors, "top_y"),
        "blade_bottom_y": get(anchors, "bottom_y"),
        "blade_thickness_px": get(anchors, "thickness"),

        "upper_wafer_y": get(upper, "y_global"),
        "lower_wafer_y": get(lower, "y_global"),

        "upper_status": result["upper_status"],
        "lower_status": result["lower_status"],

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
        description="Phase B v4 - Dual Reference Template Matching"
    )

    # Input / output
    parser.add_argument("--images", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--dark-template", required=True)

    parser.add_argument(
        "--output",
        default="output/06_phase_b_v4",
    )

    # ROI
    parser.add_argument(
        "--search-roi",
        nargs=4, type=int,
        default=[41, 358, 2027, 186],
    )

    parser.add_argument(
        "--measure-roi",
        nargs=4, type=int,
        default=[49, 296, 381, 273],
    )

    # Matching thresholds
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

    # CLAHE
    parser.add_argument(
        "--clahe-clip",
        type=float, default=2.0,
    )
    parser.add_argument(
        "--clahe-grid",
        type=int, default=8,
    )

    # Optional position validation
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

    # Normal template geometry
    parser.add_argument(
        "--normal-x-ratio",
        type=float, default=0.5,
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

    # Dark template geometry
    parser.add_argument(
        "--dark-x-ratio",
        type=float, default=0.5,
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

    # Measurement
    parser.add_argument(
        "--measure-band-width",
        type=int, default=60,
    )
    parser.add_argument(
        "--min-gap",
        type=float, default=5.0,
    )
    parser.add_argument(
        "--max-gap",
        type=float, default=100.0,
    )
    parser.add_argument(
        "--wafer-edge-threshold",
        type=float, default=18.0,
    )
    parser.add_argument(
        "--wafer-min-support",
        type=float, default=0.35,
    )
    parser.add_argument(
        "--wafer-min-prominence",
        type=float, default=5.0,
    )
    parser.add_argument(
        "--wafer-max-y-std",
        type=float, default=4.0,
    )

    args = parser.parse_args()

    cfg = Config(
        search_roi=tuple(args.search_roi),
        measure_roi=tuple(args.measure_roi),

        gray_threshold=args.gray_threshold,
        dark_gray_threshold=args.dark_gray_threshold,
        clahe_threshold=args.clahe_threshold,
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

        measure_band_width=args.measure_band_width,
        min_gap=args.min_gap,
        max_gap=args.max_gap,

        wafer_edge_threshold=args.wafer_edge_threshold,
        wafer_min_support=args.wafer_min_support,
        wafer_min_prominence=args.wafer_min_prominence,
        wafer_max_y_std=args.wafer_max_y_std,
    )

    # Validation
    if cfg.measure_band_width < 8:
        parser.error("measure-band-width must be >= 8")

    if cfg.min_gap < 0 or cfg.max_gap <= cfg.min_gap:
        parser.error("Invalid gap range")

    for name in [
        "normal_x_ratio",
        "dark_x_ratio",
        "normal_top_ratio",
        "normal_bottom_ratio",
        "dark_top_ratio",
        "dark_bottom_ratio",
    ]:
        if not 0 <= getattr(cfg, name) <= 1:
            parser.error(f"{name} must be in [0, 1]")

    for name in [
        "gray_threshold",
        "dark_gray_threshold",
        "clahe_threshold",
        "dark_clahe_threshold",
    ]:
        if not 0 <= getattr(cfg, name) <= 1:
            parser.error(f"{name} must be in [0, 1]")

    if (
        (cfg.expected_match_x is None)
        != (cfg.expected_match_y is None)
    ):
        parser.error(
            "expected-match-x and expected-match-y "
            "must be provided together"
        )

    image_dir = Path(args.images)
    output_dir = Path(args.output)

    references = load_references(
        args.template,
        args.dark_template,
        cfg,
    )

    overlay_dir = output_dir / "overlay"
    detail_dir = output_dir / "detail"
    debug_dir = output_dir / "debug"

    for directory in [
        overlay_dir,
        detail_dir,
        debug_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }
    )

    if not files:
        raise FileNotFoundError(image_dir)

    print("=" * 70)
    print("Phase B v4 - Dual Reference Template Matching")
    print("=" * 70)
    print(f"Images          : {len(files)}")
    print(f"Normal template : {args.template}")
    print(f"Dark template   : {args.dark_template}")
    print(f"Search ROI      : {cfg.search_roi}")
    print(f"Measure ROI     : {cfg.measure_roi}")
    print()

    results = []

    for i, path in enumerate(files, start=1):
        image = cv2.imread(str(path))

        if image is None:
            results.append({
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
                image, result, debug, cfg
            )

            detail = create_detail(
                overlay, result, cfg
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

        results.append(row)

        score = row.get("match_score")

        score_text = (
            f"{score:.4f}"
            if score is not None
            else "N/A"
        )

        print(
            f"[{i:02d}/{len(files):02d}] "
            f"{path.name} | "
            f"{row['status']} | "
            f"Ref={row.get('match_reference')} | "
            f"Method={row.get('match_method')} | "
            f"Stage={row.get('match_stage')} | "
            f"Score={score_text} | "
            f"Top={row.get('top_gap_px')} | "
            f"Bottom={row.get('bottom_gap_px')}"
        )

    df = pd.DataFrame(results)

    result_path = output_dir / "geometry_results.csv"

    df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("Result Summary")
    print("=" * 70)

    print(df["status"].value_counts().to_string())

    if "match_stage" in df.columns:
        print()
        print("Matching Stage Distribution")
        print("-" * 70)
        print(
            df["match_stage"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    print()
    print(f"CSV     : {result_path}")
    print(f"Overlay : {overlay_dir}")
    print(f"Detail  : {detail_dir}")
    print(f"Debug   : {debug_dir}")


if __name__ == "__main__":
    main()
