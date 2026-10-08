
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class Config:
    # x, y, width, height (원본 이미지 좌표)
    blade_roi: tuple[int, int, int, int] = (41, 358, 2027, 186)
    measure_roi: tuple[int, int, int, int] = (49, 296, 381, 273)

    # Sobel / profile
    blur_ksize: int = 3
    sobel_ksize: int = 3
    profile_smooth: int = 3
    min_peak_distance: int = 2
    min_peak_prominence: float = 10.0
    peak_mad_scale: float = 2.0

    # Gradient가 일정 X 범위 이상 유지되는지 확인
    min_support_ratio: float = 0.18

    # 경계 쌍의 수직 거리
    blade_min_thickness: int = 8
    blade_max_thickness: int = 65
    wafer_min_thickness: int = 1
    wafer_max_thickness: int = 10

    # Blade 중심 예상 위치
    # None이면 blade_roi의 수직 중심 사용
    blade_center_y: float | None = None

    # Blade 위치 prior (낮을수록 중심 위치 우선)
    blade_position_weight: float = 1.5

    # Blade 경계와 Wafer 사이 허용 거리
    min_gap: float = 1.0
    max_gap: float = 110.0

    # 기준 두께/위치와 비교할 때 사용
    min_pair_strength: float = 12.0

    # 한쪽 Wafer 후보가 애매하면 검출 실패 처리
    min_wafer_score: float = 0.0

    # 후보가 2개 이상일 때 모호성 기록
    ambiguity_ratio: float = 0.90

    # Edge가 작은 영역의 표면 반사에만 집중되는 것 방지
    support_gradient_threshold: float = 20.0

    jpeg_quality: int = 95


def crop_with_bounds(image, roi):
    x, y, w, h = roi
    ih, iw = image.shape[:2]

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(iw, int(x + w))
    y2 = min(ih, int(y + h))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid ROI {roi} for image {iw}x{ih}"
        )

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def smooth_1d(arr, k):
    if k <= 1:
        return arr.astype(float)

    if k % 2 == 0:
        k += 1

    return cv2.GaussianBlur(
        np.asarray(arr, dtype=np.float32).reshape(-1, 1),
        (1, k),
        0,
    ).ravel().astype(float)


def find_signed_peaks(signal, min_height, min_distance):
    """
    양수/음수 Sobel profile의 local extrema 추출.

    peak:
       index, sign, magnitude
    """
    n = len(signal)
    candidates = []

    for i in range(1, n - 1):
        v = float(signal[i])

        if v >= min_height:
            if v >= signal[i - 1] and v > signal[i + 1]:
                candidates.append((i, +1, abs(v)))

        elif -v >= min_height:
            if v <= signal[i - 1] and v < signal[i + 1]:
                candidates.append((i, -1, abs(v)))

    # 큰 peak부터 선택해 인접 peak 중복 제거
    candidates.sort(key=lambda p: p[2], reverse=True)

    selected = []

    for item in candidates:
        if all(
            abs(item[0] - prev[0]) >= min_distance
            for prev in selected
        ):
            selected.append(item)

    return sorted(selected, key=lambda p: p[0])


def calculate_edge_profile(image, cfg):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    k = cfg.blur_ksize
    if k > 1:
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=cfg.sobel_ksize,
    )

    # 수평 방향으로 집계하여 Y 위치별 edge profile 생성
    signed = np.median(gy, axis=1).astype(float)
    absolute = np.median(np.abs(gy), axis=1).astype(float)

    signed = smooth_1d(signed, cfg.profile_smooth)
    absolute = smooth_1d(absolute, cfg.profile_smooth)

    # 각 row에서 edge가 얼마나 넓게 존재하는지
    support = np.mean(
        np.abs(gy) >= cfg.support_gradient_threshold,
        axis=1,
    ).astype(float)

    support = smooth_1d(support, cfg.profile_smooth)

    return gray, gy, signed, absolute, support


def extract_edge_pairs(
    peaks,
    support,
    min_thickness,
    max_thickness,
    cfg,
):
    """
    서로 반대 부호인 수평 경계 2개를 하나의 객체 후보로 묶음.
    두께가 얇으면 wafer, 두꺼우면 blade 후보가 됨.

    밝은 물체: + -> -
    어두운 물체: - -> +
    양쪽 모두 허용.
    """
    pairs = []

    for i, top in enumerate(peaks):
        for bottom in peaks[i + 1:]:
            y1, s1, v1 = top
            y2, s2, v2 = bottom

            thickness = y2 - y1

            if thickness > max_thickness:
                break

            if thickness < min_thickness:
                continue

            if s1 == s2:
                continue

            p1 = float(support[y1])
            p2 = float(support[y2])
            pair_support = min(p1, p2)

            if pair_support < cfg.min_support_ratio:
                continue

            strength = min(v1, v2)

            if strength < cfg.min_pair_strength:
                continue

            score = strength * (0.5 + pair_support)

            pairs.append({
                "top_y": int(y1),
                "bottom_y": int(y2),
                "center_y": (y1 + y2) / 2.0,
                "thickness": int(thickness),
                "strength": float(strength),
                "support": float(pair_support),
                "score": float(score),
                "polarity": "bright" if s1 > 0 else "dark",
            })

    return pairs


def detect_blade(pairs, measure_bounds, cfg):
    """
    Blade의 예상 Y 범위 안에서 두꺼운 경계 쌍 선택.

    좌측 Measurement ROI에서 실제로 추출된 edge pair만 사용.
    """
    _, my1, _, _ = measure_bounds

    bx, by, bw, bh = cfg.blade_roi

    expected_center = (
        cfg.blade_center_y
        if cfg.blade_center_y is not None
        else by + bh / 2
    )

    candidates = []

    for pair in pairs:
        center_global = my1 + pair["center_y"]
        top_global = my1 + pair["top_y"]
        bottom_global = my1 + pair["bottom_y"]

        if top_global < by or bottom_global > by + bh:
            continue

        distance = abs(center_global - expected_center)

        # 강한 수평 경계를 선호하되 Blade 위치 prior 사용
        score = (
            np.log1p(pair["score"]) * 10.0
            - cfg.blade_position_weight * distance
        )

        candidates.append({
            **pair,
            "selection_score": float(score),
        })

    candidates.sort(
        key=lambda x: x["selection_score"],
        reverse=True,
    )

    return (
        candidates[0] if candidates else None,
        candidates,
    )


def detect_adjacent_wafer(
    wafer_pairs,
    blade,
    side,
    cfg,
):
    """
    Blade와 마주 보는 wafer 표면 기준.

    upper:
      wafer.bottom_y < blade.top_y

    lower:
      wafer.top_y > blade.bottom_y
    """
    candidates = []

    for pair in wafer_pairs:
        if side == "upper":
            gap = blade["top_y"] - pair["bottom_y"]
            facing_y = pair["bottom_y"]
        else:
            gap = pair["top_y"] - blade["bottom_y"]
            facing_y = pair["top_y"]

        if not (cfg.min_gap <= gap <= cfg.max_gap):
            continue

        # 가까운 wafer를 우선하되 선의 강도/연속성 고려
        strength_bonus = np.log1p(pair["score"]) * 2.0

        score = 100.0 / (1.0 + gap) + strength_bonus

        if score < cfg.min_wafer_score:
            continue

        candidates.append({
            **pair,
            "gap_px": float(gap),
            "facing_y": int(facing_y),
            "selection_score": float(score),
        })

    candidates.sort(
        key=lambda x: x["selection_score"],
        reverse=True,
    )

    selected = candidates[0] if candidates else None

    if selected is None:
        status = "not_detected"
    elif (
        len(candidates) >= 2
        and candidates[1]["selection_score"]
        >= selected["selection_score"] * cfg.ambiguity_ratio
    ):
        status = "ambiguous"
    else:
        status = "detected"

    return selected, status, candidates


def analyze_geometry(image, cfg):
    roi_img, bounds = crop_with_bounds(
        image,
        cfg.measure_roi,
    )

    _, y_offset, _, _ = bounds

    gray, gy, signed, absolute, support = (
        calculate_edge_profile(roi_img, cfg)
    )

    # MAD 기반 threshold와 절대 threshold 조합
    median = float(np.median(signed))
    mad = float(np.median(np.abs(signed - median)))

    threshold = max(
        cfg.min_peak_prominence,
        cfg.peak_mad_scale * 1.4826 * mad,
    )

    peaks = find_signed_peaks(
        signed,
        threshold,
        cfg.min_peak_distance,
    )

    blade_pairs = extract_edge_pairs(
        peaks,
        support,
        cfg.blade_min_thickness,
        cfg.blade_max_thickness,
        cfg,
    )

    wafer_pairs = extract_edge_pairs(
        peaks,
        support,
        cfg.wafer_min_thickness,
        cfg.wafer_max_thickness,
        cfg,
    )

    blade, blade_candidates = detect_blade(
        blade_pairs,
        bounds,
        cfg,
    )

    debug = {
        "bounds": bounds,
        "gray": gray,
        "gy": gy,
        "signed": signed,
        "absolute": absolute,
        "support": support,
        "peaks": peaks,
        "threshold": threshold,
        "blade_candidates": blade_candidates,
        "wafer_pairs": wafer_pairs,
    }

    if blade is None:
        return {
            "status": "blade_not_detected",
            "blade": None,
            "upper": None,
            "lower": None,
            "upper_status": "not_evaluated",
            "lower_status": "not_evaluated",
            "top_gap_px": None,
            "bottom_gap_px": None,
            "top_ratio": None,
            "bottom_ratio": None,
        }, debug

    upper, upper_status, upper_candidates = (
        detect_adjacent_wafer(
            wafer_pairs,
            blade,
            "upper",
            cfg,
        )
    )

    lower, lower_status, lower_candidates = (
        detect_adjacent_wafer(
            wafer_pairs,
            blade,
            "lower",
            cfg,
        )
    )

    debug["upper_candidates"] = upper_candidates
    debug["lower_candidates"] = lower_candidates

    top_gap = upper["gap_px"] if upper else None
    bottom_gap = lower["gap_px"] if lower else None

    top_ratio = None
    bottom_ratio = None

    if top_gap is not None and bottom_gap is not None:
        total = top_gap + bottom_gap

        if total > 0:
            top_ratio = 100.0 * top_gap / total
            bottom_ratio = 100.0 * bottom_gap / total

    if upper is None and lower is None:
        status = "no_adjacent_wafer"
    elif upper_status == "ambiguous" or lower_status == "ambiguous":
        status = "review"
    elif upper is None or lower is None:
        status = "partial"
    else:
        status = "ok"

    return {
        "status": status,
        "blade": blade,
        "upper": upper,
        "lower": lower,
        "upper_status": upper_status,
        "lower_status": lower_status,
        "top_gap_px": top_gap,
        "bottom_gap_px": bottom_gap,
        "top_ratio": top_ratio,
        "bottom_ratio": bottom_ratio,
    }, debug


def draw_overlay(image, result, cfg):
    out = image.copy()

    mx, my, mw, mh = cfg.measure_roi
    bx, by, bw, bh = cfg.blade_roi

    # Blade ROI
    cv2.rectangle(
        out, (bx, by), (bx + bw, by + bh),
        (255, 160, 0), 2,
    )

    # Measurement ROI
    cv2.rectangle(
        out, (mx, my), (mx + mw, my + mh),
        (0, 255, 255), 2,
    )

    blade = result["blade"]
    upper = result["upper"]
    lower = result["lower"]

    if blade is not None:
        _, y_offset, _, _ = crop_with_bounds(
            image, cfg.measure_roi
        )[1]

        def line(local_y, color, thickness=2):
            gy = int(round(y_offset + local_y))
            cv2.line(
                out,
                (mx, gy),
                (mx + mw, gy),
                color,
                thickness,
            )
            return gy

        bt = line(blade["top_y"], (255, 0, 0), 3)
        bb = line(blade["bottom_y"], (255, 0, 0), 3)

        if upper is not None:
            uy = line(
                upper["bottom_y"],
                (0, 255, 0),
            )
            cv2.line(
                out,
                (mx + mw // 2, uy),
                (mx + mw // 2, bt),
                (0, 255, 0),
                2,
            )

        if lower is not None:
            ly = line(
                lower["top_y"],
                (0, 165, 255),
            )
            cv2.line(
                out,
                (mx + mw // 2, bb),
                (mx + mw // 2, ly),
                (0, 165, 255),
                2,
            )

    # 측정 영역을 별도 패널로 확대해 출력
    roi_img, _ = crop_with_bounds(out, cfg.measure_roi)

    scale = 2
    zoom = cv2.resize(
        roi_img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    # 우측 정보 패널
    panel = np.zeros((zoom.shape[0], 430, 3), np.uint8)
    panel[:] = (30, 30, 30)

    def fmt(value):
        return "SKIP" if value is None else f"{value:.2f}"

    lines = [
        f"STATUS: {result['status']}",
        "",
        f"TOP GAP: {fmt(result['top_gap_px'])} px",
        f"BOTTOM GAP: {fmt(result['bottom_gap_px'])} px",
        "",
        f"TOP RATIO: {fmt(result['top_ratio'])} %",
        f"BOTTOM RATIO: {fmt(result['bottom_ratio'])} %",
        "",
        f"UPPER: {result['upper_status']}",
        f"LOWER: {result['lower_status']}",
    ]

    for i, text in enumerate(lines):
        cv2.putText(
            panel,
            text,
            (15, 38 + i * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    detail = np.hstack([zoom, panel])
    return out, detail


def save_debug_csv(path, debug):
    bounds = debug["bounds"]
    y_offset = bounds[1]

    rows = []

    for i, value in enumerate(debug["signed"]):
        rows.append({
            "y_local": i,
            "y_global": i + y_offset,
            "signed_sobel": float(value),
            "abs_sobel": float(debug["absolute"][i]),
            "edge_support": float(debug["support"][i]),
            "peak_threshold": debug["threshold"],
        })

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def save_candidates_csv(path, debug):
    rows = []

    for kind, key in [
        ("blade", "blade_candidates"),
        ("upper", "upper_candidates"),
        ("lower", "lower_candidates"),
    ]:
        for rank, candidate in enumerate(
            debug.get(key, []), start=1
        ):
            rows.append({
                "kind": kind,
                "rank": rank,
                **candidate,
            })

    pd.DataFrame(
        rows,
        columns=[
            "kind", "rank", "top_y", "bottom_y",
            "center_y", "thickness", "strength",
            "support", "score", "polarity",
            "selection_score", "gap_px", "facing_y",
        ],
    ).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def to_row(image_name, result):
    blade = result["blade"]
    upper = result["upper"]
    lower = result["lower"]

    def value(obj, key):
        return obj[key] if obj is not None else None

    return {
        "image": image_name,
        "status": result["status"],

        "blade_top_y_local": value(blade, "top_y"),
        "blade_bottom_y_local": value(blade, "bottom_y"),
        "blade_thickness_px": value(blade, "thickness"),

        "upper_wafer_bottom_y_local": (
            value(upper, "bottom_y")
        ),
        "lower_wafer_top_y_local": (
            value(lower, "top_y")
        ),

        "upper_status": result["upper_status"],
        "lower_status": result["lower_status"],

        "top_gap_px": result["top_gap_px"],
        "bottom_gap_px": result["bottom_gap_px"],

        "top_ratio": result["top_ratio"],
        "bottom_ratio": result["bottom_ratio"],

        "blade_score": value(blade, "selection_score"),
        "upper_score": value(upper, "selection_score"),
        "lower_score": value(lower, "selection_score"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase B - Wafer Blade Gap Measurement"
    )

    parser.add_argument("--images", required=True)
    parser.add_argument(
        "--output", default="output/06_phase_b"
    )

    parser.add_argument(
        "--blade-roi", nargs=4, type=int,
        default=[41, 358, 2027, 186],
        metavar=("X", "Y", "W", "H"),
    )

    parser.add_argument(
        "--measure-roi", nargs=4, type=int,
        default=[49, 296, 381, 273],
        metavar=("X", "Y", "W", "H"),
    )

    parser.add_argument(
        "--blade-center-y", type=float, default=None
    )

    parser.add_argument(
        "--blade-thickness", nargs=2, type=int,
        default=[8, 65],
        metavar=("MIN", "MAX"),
    )

    parser.add_argument(
        "--wafer-thickness", nargs=2, type=int,
        default=[1, 10],
        metavar=("MIN", "MAX"),
    )

    parser.add_argument(
        "--max-gap", type=float, default=110.0
    )

    parser.add_argument(
        "--min-gap", type=float, default=1.0
    )

    parser.add_argument(
        "--peak-prominence", type=float, default=10.0
    )

    parser.add_argument(
        "--peak-mad-scale", type=float, default=2.0
    )

    parser.add_argument(
        "--min-support", type=float, default=0.18
    )

    parser.add_argument(
        "--min-pair-strength", type=float, default=12.0
    )

    parser.add_argument(
        "--blade-position-weight", type=float, default=1.5
    )

    args = parser.parse_args()

    cfg = Config(
        blade_roi=tuple(args.blade_roi),
        measure_roi=tuple(args.measure_roi),
        blade_center_y=args.blade_center_y,
        blade_min_thickness=args.blade_thickness[0],
        blade_max_thickness=args.blade_thickness[1],
        wafer_min_thickness=args.wafer_thickness[0],
        wafer_max_thickness=args.wafer_thickness[1],
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        min_peak_prominence=args.peak_prominence,
        peak_mad_scale=args.peak_mad_scale,
        min_support_ratio=args.min_support,
        min_pair_strength=args.min_pair_strength,
        blade_position_weight=args.blade_position_weight,
    )

    images_dir = Path(args.images)
    output_dir = Path(args.output)

    overlay_dir = output_dir / "overlay"
    detail_dir = output_dir / "detail"
    debug_dir = output_dir / "debug"
    candidate_dir = output_dir / "candidates"

    for directory in [
        output_dir, overlay_dir, detail_dir,
        debug_dir, candidate_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )

    if not images:
        raise FileNotFoundError(
            f"No images in {images_dir}"
        )

    print("=" * 65)
    print("Phase B v1 - Wafer / Blade Gap Measurement")
    print("=" * 65)
    print(f"Images       : {len(images)}")
    print(f"Blade ROI    : {cfg.blade_roi}")
    print(f"Measure ROI  : {cfg.measure_roi}")
    print(f"Blade thick  : {cfg.blade_min_thickness}~{cfg.blade_max_thickness}")
    print(f"Wafer thick  : {cfg.wafer_min_thickness}~{cfg.wafer_max_thickness}")
    print()

    results = []

    for i, path in enumerate(images, 1):
        image = cv2.imread(str(path))

        if image is None:
            results.append({
                "image": path.name,
                "status": "image_read_failed",
            })
            continue

        try:
            result, debug = analyze_geometry(image, cfg)

            overlay, detail = draw_overlay(image, result, cfg)

            cv2.imwrite(
                str(overlay_dir / path.name),
                overlay,
            )

            cv2.imwrite(
                str(detail_dir / path.name),
                detail,
            )

            save_debug_csv(
                debug_dir / f"{path.stem}_profile.csv",
                debug,
            )

            save_candidates_csv(
                candidate_dir / f"{path.stem}_candidates.csv",
                debug,
            )

            row = to_row(path.name, result)

        except Exception as exc:
            row = {
                "image": path.name,
                "status": "error",
                "error": str(exc),
            }

        results.append(row)

        print(
            f"[{i:02d}/{len(images):02d}] "
            f"{path.name} | "
            f"{row['status']} | "
            f"top={row.get('top_gap_px')} | "
            f"bottom={row.get('bottom_gap_px')}"
        )

    result_df = pd.DataFrame(results)

    csv_path = output_dir / "geometry_results.csv"

    result_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Result summary")
    print("-" * 65)
    print(result_df["status"].value_counts().to_string())
    print()
    print(f"CSV       : {csv_path}")
    print(f"Overlay   : {overlay_dir}")
    print(f"Detail    : {detail_dir}")
    print(f"Profile   : {debug_dir}")
    print(f"Candidates: {candidate_dir}")


if __name__ == "__main__":
    main()
