
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
    search_roi: tuple[int, int, int, int] = (
        41, 358, 2027, 186
    )
    measure_roi: tuple[int, int, int, int] = (
        49, 296, 381, 273
    )

    match_threshold: float = 0.65

    # Template 내 측정 기준 위치
    measure_x_ratio: float = 0.5

    # Template 상하 경계 위치
    blade_top_ratio: float = 0.0
    blade_bottom_ratio: float = 1.0
    blade_top_offset: float = 0.0
    blade_bottom_offset: float = 0.0

    # 측정 기준 X 중심 주변 분석 폭
    measure_band_width: int = 60

    # Wafer 탐색 범위
    min_gap: int = 5
    max_gap: int = 100
    exclusion_px: int = 4

    # Wafer edge 검출
    blur_ksize: int = 3
    profile_smooth: int = 3
    wafer_edge_threshold: float = 18.0
    wafer_min_support: float = 0.35
    wafer_support_gradient: float = 15.0
    wafer_min_prominence: float = 5.0

    # 수평 edge의 위치가 X 방향으로 일관적인지 검사
    wafer_max_y_std: float = 4.0

    # 경계 후보 사이의 최소 거리
    edge_min_distance: int = 3

    jpeg_quality: int = 95


# ============================================================
# Utilities
# ============================================================

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
            f"ROI outside image: {roi}, {iw}x{ih}"
        )

    return (
        image[y1:y2, x1:x2],
        (x1, y1, x2, y2),
    )


def odd(value):
    value = max(1, int(value))
    return value if value % 2 else value + 1


def smooth_1d(values, kernel):
    values = np.asarray(values, dtype=np.float32)

    if kernel <= 1:
        return values

    return cv2.GaussianBlur(
        values.reshape(-1, 1),
        (1, odd(kernel)),
        0,
    ).ravel()


def preprocess_gray(image, blur_ksize=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur_ksize > 1:
        k = odd(blur_ksize)
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    return gray


# ============================================================
# 1. Template Matching
# ============================================================

def match_blade(image, template, cfg):
    search, bounds = crop_roi(
        image,
        cfg.search_roi,
    )

    sx1, sy1, sx2, sy2 = bounds

    search_gray = preprocess_gray(search)
    template_gray = preprocess_gray(template)

    th, tw = template_gray.shape
    sh, sw = search_gray.shape

    if tw > sw or th > sh:
        raise ValueError(
            f"Template {tw}x{th} is larger than "
            f"search ROI {sw}x{sh}"
        )

    if template_gray.std() < 2.0:
        raise ValueError(
            "Template has insufficient contrast"
        )

    response = cv2.matchTemplate(
        search_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED,
    )

    # 보호: 수치적으로 유효하지 않은 결과 제외
    response = np.nan_to_num(
        response,
        nan=-1.0,
        posinf=-1.0,
        neginf=-1.0,
    )

    _, max_score, _, max_loc = cv2.minMaxLoc(
        response
    )

    local_x, local_y = max_loc

    match_x = sx1 + local_x
    match_y = sy1 + local_y

    # 최고점 주변을 가리고 2순위 매칭 확인
    second_response = response.copy()

    exclusion_x = max(4, tw // 2)
    exclusion_y = max(4, th // 2)

    x1 = max(0, local_x - exclusion_x)
    y1 = max(0, local_y - exclusion_y)
    x2 = min(
        second_response.shape[1],
        local_x + exclusion_x + 1,
    )
    y2 = min(
        second_response.shape[0],
        local_y + exclusion_y + 1,
    )

    second_response[y1:y2, x1:x2] = -1.0

    second_score = float(
        np.max(second_response)
    )

    return {
        "status": (
            "matched"
            if max_score >= cfg.match_threshold
            else "low_score"
        ),
        "score": float(max_score),
        "second_score": second_score,
        "score_margin": float(
            max_score - second_score
        ),
        "x": int(match_x),
        "y": int(match_y),
        "w": int(tw),
        "h": int(th),
    }


# ============================================================
# 2. Blade measurement anchors
# ============================================================

def calculate_blade_anchors(match, cfg):
    x = (
        match["x"]
        + cfg.measure_x_ratio * match["w"]
    )

    top_y = (
        match["y"]
        + cfg.blade_top_ratio * match["h"]
        + cfg.blade_top_offset
    )

    bottom_y = (
        match["y"]
        + cfg.blade_bottom_ratio * match["h"]
        + cfg.blade_bottom_offset
    )

    if top_y >= bottom_y:
        raise ValueError(
            "blade_top_y must be smaller than blade_bottom_y"
        )

    return {
        "x": float(x),
        "top_y": float(top_y),
        "bottom_y": float(bottom_y),
        "thickness": float(bottom_y - top_y),
    }


# ============================================================
# 3. Measurement band
# ============================================================

def get_measurement_band(image, anchors, cfg):
    mx, my, mw, mh = cfg.measure_roi

    ih, iw = image.shape[:2]

    roi_x1 = max(0, mx)
    roi_y1 = max(0, my)
    roi_x2 = min(iw, mx + mw)
    roi_y2 = min(ih, my + mh)

    center_x = anchors["x"]
    half = cfg.measure_band_width / 2.0

    x1 = max(
        roi_x1,
        int(round(center_x - half)),
    )

    x2 = min(
        roi_x2,
        int(round(center_x + half)),
    )

    if x2 - x1 < 8:
        raise ValueError(
            "Measurement X band does not sufficiently "
            "overlap measure ROI. Adjust --measure-x-ratio, "
            "--measure-band-width or --measure-roi."
        )

    if roi_y2 <= roi_y1:
        raise ValueError("Invalid measurement ROI")

    return (
        image[roi_y1:roi_y2, x1:x2],
        (x1, roi_y1, x2, roi_y2),
    )


# ============================================================
# 4. Wafer edge profile
# ============================================================

def calculate_wafer_profile(band, cfg):
    gray = preprocess_gray(
        band,
        cfg.blur_ksize,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    # X 방향 중앙값을 사용해 반사 잡음 완화
    edge_profile = np.median(
        np.abs(gy),
        axis=1,
    )

    edge_profile = smooth_1d(
        edge_profile,
        cfg.profile_smooth,
    )

    signed_profile = smooth_1d(
        np.median(gy, axis=1),
        cfg.profile_smooth,
    )

    # 해당 row의 어느 정도 폭에서 edge가 나타나는가
    support = np.mean(
        np.abs(gy) >= cfg.wafer_support_gradient,
        axis=1,
    )

    support = smooth_1d(
        support,
        cfg.profile_smooth,
    )

    return {
        "gray": gray,
        "gy": gy,
        "edge": edge_profile,
        "signed": signed_profile,
        "support": support,
    }


# ============================================================
# 5. Candidate edge extraction
# ============================================================

def detect_edge_candidates(profile, bounds, cfg):
    y_offset = bounds[1]

    edge = profile["edge"]
    support = profile["support"]
    gy = profile["gy"]

    candidates = []

    for y in range(1, len(edge) - 1):
        strength = float(edge[y])

        if strength < cfg.wafer_edge_threshold:
            continue

        # Local peak
        if not (
            strength >= edge[y - 1]
            and strength > edge[y + 1]
        ):
            continue

        prominence = strength - min(
            float(edge[y - 1]),
            float(edge[y + 1]),
        )

        # 근방 변화량: smoothing으로 peak가 넓게
        # 형성되는 경우를 고려해 +/- 3px도 비교
        left = max(0, y - 3)
        right = min(len(edge), y + 4)

        local_min = float(
            np.min(edge[left:right])
        )

        prominence = max(
            prominence,
            strength - local_min,
        )

        if prominence < cfg.wafer_min_prominence:
            continue

        line_support = float(support[y])

        if line_support < cfg.wafer_min_support:
            continue

        # 각 X column에서 현재 Y 근처 최대 gradient 위치
        local_y1 = max(0, y - 3)
        local_y2 = min(gy.shape[0], y + 4)

        local_grad = np.abs(
            gy[local_y1:local_y2, :]
        )

        column_max = np.max(
            local_grad,
            axis=0,
        )

        valid_columns = (
            column_max >= cfg.wafer_support_gradient
        )

        if np.count_nonzero(valid_columns) < 3:
            continue

        local_argmax = np.argmax(
            local_grad,
            axis=0,
        )

        column_ys = (
            local_argmax[valid_columns] + local_y1
        )

        y_std = float(
            np.std(column_ys)
        )

        if y_std > cfg.wafer_max_y_std:
            continue

        candidates.append({
            "y_local": int(y),
            "y_global": int(y + y_offset),
            "strength": strength,
            "prominence": float(prominence),
            "support": line_support,
            "y_std": y_std,
        })

    # 서로 가까운 중복 peak는 높은 강도만 남김
    candidates.sort(
        key=lambda c: c["strength"],
        reverse=True,
    )

    selected = []

    for item in candidates:
        if any(
            abs(item["y_local"] - prev["y_local"])
            < cfg.edge_min_distance
            for prev in selected
        ):
            continue

        selected.append(item)

    return sorted(
        selected,
        key=lambda c: c["y_global"],
    )


# ============================================================
# 6. Nearest wafer edge
# ============================================================

def find_nearest_wafer(
    candidates,
    blade_y,
    side,
    cfg,
):
    """
    upper:
        blade_top 위쪽에서 아래로 접근하는 첫 경계

    lower:
        blade_bottom 아래쪽에서 아래로 접근하는 첫 경계

    검출된 단일 경계가 실제 wafer의 맞은편 표면인지
    여부는 추후 edge polarity / edge-pair 검증 대상.
    """

    valid = []

    for candidate in candidates:
        wafer_y = candidate["y_global"]

        if side == "upper":
            gap = blade_y - wafer_y
        else:
            gap = wafer_y - blade_y

        minimum = max(
            cfg.min_gap,
            cfg.exclusion_px,
        )

        if not (
            minimum <= gap <= cfg.max_gap
        ):
            continue

        valid.append({
            **candidate,
            "gap_px": float(gap),
        })

    if not valid:
        return None, "not_detected", []

    # 가장 가까운 경계 선택
    valid.sort(
        key=lambda item: item["gap_px"]
    )

    return valid[0], "detected", valid


# ============================================================
# 7. Geometry measurement
# ============================================================

def analyze_image(image, template, cfg):
    match = match_blade(
        image,
        template,
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

    debug = {}

    if match["status"] != "matched":
        result["status"] = "blade_match_failed"
        return result, debug

    anchors = calculate_blade_anchors(
        match,
        cfg,
    )

    result["anchors"] = anchors

    # 템플릿 측정 기준점이 ROI 안에 있는지 검증
    my = cfg.measure_roi[1]
    mh = cfg.measure_roi[3]

    if not (
        my <= anchors["top_y"] < anchors["bottom_y"]
        < my + mh
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

    result.update({
        "upper": upper,
        "lower": lower,
        "upper_status": upper_status,
        "lower_status": lower_status,
        "top_gap_px": (
            upper["gap_px"] if upper else None
        ),
        "bottom_gap_px": (
            lower["gap_px"] if lower else None
        ),
    })

    top = result["top_gap_px"]
    bottom = result["bottom_gap_px"]

    if top is not None and bottom is not None:
        total = top + bottom

        if total > 0:
            result["top_ratio"] = 100.0 * top / total
            result["bottom_ratio"] = 100.0 * bottom / total

    if upper and lower:
        result["status"] = "ok"
    elif upper or lower:
        result["status"] = "partial"
    else:
        result["status"] = "no_wafer_detected"

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
# 8. Overlay
# ============================================================

def draw_overlay(image, result, debug, cfg):
    out = image.copy()

    sx, sy, sw, sh = cfg.search_roi
    mx, my, mw, mh = cfg.measure_roi

    cv2.rectangle(
        out,
        (sx, sy),
        (sx + sw, sy + sh),
        (255, 140, 0),
        2,
    )

    cv2.rectangle(
        out,
        (mx, my),
        (mx + mw, my + mh),
        (0, 255, 255),
        2,
    )

    match = result["match"]

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

    if anchors is not None:
        center_x = int(round(anchors["x"]))
        blade_top = int(round(anchors["top_y"]))
        blade_bottom = int(round(anchors["bottom_y"]))

        if "bounds" in debug:
            x1, _, x2, _ = debug["bounds"]
        else:
            x1 = center_x - 20
            x2 = center_x + 20

        # Blade 기준 경계
        cv2.line(
            out,
            (x1, blade_top),
            (x2, blade_top),
            (255, 0, 0),
            2,
        )
        cv2.line(
            out,
            (x1, blade_bottom),
            (x2, blade_bottom),
            (255, 0, 0),
            2,
        )

        # Upper wafer
        if result["upper"] is not None:
            wafer_y = result["upper"]["y_global"]

            cv2.line(
                out,
                (x1, wafer_y),
                (x2, wafer_y),
                (0, 255, 0),
                2,
            )

            cv2.line(
                out,
                (center_x, wafer_y),
                (center_x, blade_top),
                (0, 255, 0),
                2,
            )

        # Lower wafer
        if result["lower"] is not None:
            wafer_y = result["lower"]["y_global"]

            cv2.line(
                out,
                (x1, wafer_y),
                (x2, wafer_y),
                (0, 165, 255),
                2,
            )

            cv2.line(
                out,
                (center_x, blade_bottom),
                (center_x, wafer_y),
                (0, 165, 255),
                2,
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
        (zoom.shape[0], 450, 3),
        30,
        dtype=np.uint8,
    )

    def fmt(v):
        return "SKIP" if v is None else f"{v:.2f}"

    match = result["match"]
    anchors = result["anchors"]

    lines = [
        f"Status: {result['status']}",
        f"Match: {match['score']:.4f}",
        f"Match margin: {match['score_margin']:.4f}",
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

    if anchors is not None:
        lines.extend([
            "",
            f"Blade Top Y: {anchors['top_y']:.1f}",
            f"Blade Bottom Y: {anchors['bottom_y']:.1f}",
        ])

    for i, line in enumerate(lines):
        y = 30 + i * 31

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
# 9. Result serialization
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

        "match_score": match["score"],
        "second_match_score": match["second_score"],
        "match_margin": match["score_margin"],

        "match_x": match["x"],
        "match_y": match["y"],
        "match_w": match["w"],
        "match_h": match["h"],

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


def save_debug(debug_dir, stem, result, debug):
    if not debug:
        return

    bounds = debug["bounds"]
    y_offset = bounds[1]

    profile = debug["profile"]

    rows = []

    for y in range(len(profile["edge"])):
        rows.append({
            "y_local": y,
            "y_global": y + y_offset,
            "edge_strength": float(profile["edge"][y]),
            "signed_sobel": float(profile["signed"][y]),
            "edge_support": float(profile["support"][y]),
        })

    pd.DataFrame(rows).to_csv(
        debug_dir / f"{stem}_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidates = debug["candidates"]

    pd.DataFrame(
        candidates,
        columns=[
            "y_local",
            "y_global",
            "strength",
            "prominence",
            "support",
            "y_std",
        ],
    ).to_csv(
        debug_dir / f"{stem}_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 측정 구간 Sobel 시각화
    gy = np.abs(profile["gy"])

    scale = 255.0 / max(
        1.0,
        float(np.percentile(gy, 99)),
    )

    sobel = np.uint8(
        np.clip(gy * scale, 0, 255)
    )

    cv2.imwrite(
        str(debug_dir / f"{stem}_sobel.png"),
        sobel,
    )


# ============================================================
# 10. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase B v3 Template Matching Gap Measurement"
    )

    parser.add_argument("--images", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument(
        "--output",
        default="output/06_phase_b_v3",
    )

    parser.add_argument(
        "--search-roi",
        nargs=4, type=int,
        default=[41, 358, 2027, 186],
        metavar=("X", "Y", "W", "H"),
    )

    parser.add_argument(
        "--measure-roi",
        nargs=4, type=int,
        default=[49, 296, 381, 273],
        metavar=("X", "Y", "W", "H"),
    )

    parser.add_argument(
        "--match-threshold",
        type=float, default=0.65,
    )

    parser.add_argument(
        "--measure-x-ratio",
        type=float, default=0.5,
    )

    parser.add_argument(
        "--blade-top-ratio",
        type=float, default=0.0,
    )

    parser.add_argument(
        "--blade-bottom-ratio",
        type=float, default=1.0,
    )

    parser.add_argument(
        "--blade-top-offset",
        type=float, default=0.0,
    )

    parser.add_argument(
        "--blade-bottom-offset",
        type=float, default=0.0,
    )

    parser.add_argument(
        "--measure-band-width",
        type=int, default=60,
    )

    parser.add_argument(
        "--min-gap",
        type=int, default=5,
    )

    parser.add_argument(
        "--max-gap",
        type=int, default=100,
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
        match_threshold=args.match_threshold,
        measure_x_ratio=args.measure_x_ratio,
        blade_top_ratio=args.blade_top_ratio,
        blade_bottom_ratio=args.blade_bottom_ratio,
        blade_top_offset=args.blade_top_offset,
        blade_bottom_offset=args.blade_bottom_offset,
        measure_band_width=args.measure_band_width,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        wafer_edge_threshold=args.wafer_edge_threshold,
        wafer_min_support=args.wafer_min_support,
        wafer_min_prominence=args.wafer_min_prominence,
        wafer_max_y_std=args.wafer_max_y_std,
    )

    if not (0 <= cfg.measure_x_ratio <= 1):
        parser.error("--measure-x-ratio must be 0~1")
    if cfg.measure_band_width < 8:
        parser.error("--measure-band-width must be >= 8")
    if cfg.min_gap < 0 or cfg.max_gap <= cfg.min_gap:
        parser.error("Invalid gap range")
    if not (0 <= cfg.match_threshold <= 1):
        parser.error("--match-threshold must be 0~1")

    image_dir = Path(args.images)
    template_path = Path(args.template)
    output_dir = Path(args.output)

    template = cv2.imread(str(template_path))

    if template is None:
        raise FileNotFoundError(
            f"Cannot load template: {template_path}"
        )

    overlay_dir = output_dir / "overlay"
    detail_dir = output_dir / "detail"
    debug_dir = output_dir / "debug"

    for directory in [
        overlay_dir,
        detail_dir,
        debug_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

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

    print("=" * 65)
    print("Phase B v3 - Template Matching")
    print("=" * 65)
    print(f"Images           : {len(files)}")
    print(f"Template         : {template_path}")
    print(f"Template size    : {template.shape[1]}x{template.shape[0]}")
    print(f"Search ROI       : {cfg.search_roi}")
    print(f"Measure ROI      : {cfg.measure_roi}")
    print(f"Match threshold  : {cfg.match_threshold}")
    print(f"Measure X ratio  : {cfg.measure_x_ratio}")
    print()

    rows = []

    for i, path in enumerate(files, 1):
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
                template,
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
                result,
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

        rows.append(row)

        print(
            f"[{i:02d}/{len(files):02d}] "
            f"{path.name} | "
            f"{row['status']} | "
            f"Match={row.get('match_score', None)} | "
            f"Top={row.get('top_gap_px', None)} | "
            f"Bottom={row.get('bottom_gap_px', None)}"
        )

    result_df = pd.DataFrame(rows)

    result_path = output_dir / "geometry_results.csv"

    result_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 65)
    print("Phase B v3 Result")
    print("=" * 65)
    print(result_df["status"].value_counts().to_string())
    print()
    print(f"CSV     : {result_path}")
    print(f"Overlay : {overlay_dir}")
    print(f"Detail  : {detail_dir}")
    print(f"Debug   : {debug_dir}")


if __name__ == "__main__":
    main()
