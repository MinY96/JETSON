
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class Config:
    # 이미지 원본 좌표계의 x, y, w, h
    blade_roi: tuple = (41, 358, 2027, 186)
    measure_roi: tuple = (49, 296, 381, 273)

    # Blade 하단의 예상 전역 Y 좌표
    # None이면 Blade ROI의 아래쪽 65% 지점을 사용
    blade_bottom_y: float | None = None

    # 수평 Blade 하단 위치 탐색
    bottom_search_radius: int = 55
    bottom_min_strength: float = 10.0
    bottom_position_weight: float = 0.25

    # Blade 상단의 예상 두께
    blade_min_thickness: int = 10
    blade_max_thickness: int = 75

    # Blade 상단 외곽 추적
    strip_width: int = 12
    strip_step: int = 6
    top_edge_min_strength: float = 7.0
    top_upper_preference: float = 0.20
    top_smooth_penalty: float = 2.0
    top_max_step: int = 5
    top_anchor_y: float | None = None
    top_anchor_weight: float = 0.0

    # 상단 경계가 기울어진 경우 허용
    top_curve_smooth: int = 5

    # Wafer 검출
    wafer_min_thickness: int = 1
    wafer_max_thickness: int = 9
    wafer_min_strength: float = 7.0
    wafer_min_support: float = 0.25

    min_gap: float = 3.0
    max_gap: float = 110.0

    # Blade 외곽과 떨어진 wafer만 선택
    wafer_exclusion_px: int = 4

    # 측정 위치의 신뢰도
    min_valid_strips: int = 5
    min_valid_fraction: float = 0.50

    # 후보 간 점수 차이가 작으면 검토 대상
    ambiguity_ratio: float = 0.93

    # 전처리
    gaussian_ksize: int = 3
    profile_smooth: int = 3

    jpeg_quality: int = 95


def crop_roi(img, roi):
    x, y, w, h = map(int, roi)
    ih, iw = img.shape[:2]

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid ROI: {roi}")

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"ROI outside image: {roi}, size={iw}x{ih}"
        )

    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def odd(value):
    value = max(1, int(value))
    return value if value % 2 else value + 1


def smooth_1d(a, k):
    a = np.asarray(a, dtype=np.float32)
    if k <= 1:
        return a.copy()

    return cv2.GaussianBlur(
        a.reshape(-1, 1),
        (1, odd(k)),
        0
    ).ravel()


def local_peaks(a, threshold=0.0):
    """
    부호와 관계없이 절댓값 기준 국소 peak.
    반환: (y, signed_value, abs_strength)
    """
    a = np.asarray(a)
    result = []

    for i in range(1, len(a) - 1):
        v = float(a[i])
        m = abs(v)

        if (
            m >= threshold
            and m >= abs(float(a[i - 1]))
            and m > abs(float(a[i + 1]))
        ):
            result.append((i, v, m))

    return result


def prepare_edges(roi, cfg):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if cfg.gaussian_ksize > 1:
        gray = cv2.GaussianBlur(
            gray,
            (odd(cfg.gaussian_ksize),) * 2,
            0
        )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0, 1,
        ksize=3
    )

    return gray, gy


def detect_blade_bottom(gy, bounds, cfg):
    """
    Blade 하단은 비교적 수평이라고 가정.
    ROI 전체의 수평 에지와 Y 위치 prior로 선정.
    """
    _, y0, _, _ = bounds
    h = gy.shape[0]

    profile = smooth_1d(
        np.median(np.abs(gy), axis=1),
        cfg.profile_smooth
    )

    if cfg.blade_bottom_y is None:
        by = cfg.blade_roi[1]
        bh = cfg.blade_roi[3]
        expected_global = by + 0.65 * bh
    else:
        expected_global = cfg.blade_bottom_y

    expected_local = expected_global - y0

    candidates = []

    for y, _, strength in local_peaks(
        profile,
        cfg.bottom_min_strength
    ):
        distance = abs(y - expected_local)

        if distance > cfg.bottom_search_radius:
            continue

        score = (
            strength
            - cfg.bottom_position_weight * distance
        )

        candidates.append({
            "y_local": int(y),
            "y_global": float(y + y0),
            "strength": float(strength),
            "distance": float(distance),
            "score": float(score),
        })

    candidates.sort(
        key=lambda d: d["score"],
        reverse=True
    )

    return (
        candidates[0] if candidates else None,
        candidates,
        profile
    )


def create_x_strips(width, cfg):
    strip_w = min(cfg.strip_width, width)
    step = max(1, cfg.strip_step)

    starts = list(range(
        0,
        max(1, width - strip_w + 1),
        step
    ))

    last = max(0, width - strip_w)

    if not starts or starts[-1] != last:
        starts.append(last)

    starts = sorted(set(starts))

    return [
        (s, min(width, s + strip_w))
        for s in starts
    ]


def detect_blade_top_curve(gy, bottom_y, bounds, cfg):
    """
    각 X strip에서 상단 외곽 후보 탐색 후
    Dynamic Programming으로 연속 곡선을 선택한다.

    Blade 내부 검은 선을 억제하기 위해:
      1. 하단으로부터 두께 범위 제한
      2. 더 위쪽 후보를 선호
      3. 인접 strip 간 급격한 Y 변화 제한
      4. 전체적으로 매끄러운 경계 선택

    중요한 한계:
      실제 상단/내부선의 에지 강도와 위치가
      크게 겹치면 추적이 실패할 수 있음.
    """
    y_offset = bounds[1]
    h, w = gy.shape

    y_min = max(
        1,
        int(round(bottom_y - cfg.blade_max_thickness))
    )
    y_max = min(
        h - 2,
        int(round(bottom_y - cfg.blade_min_thickness))
    )

    if y_max <= y_min:
        return None, [], []

    strips = create_x_strips(w, cfg)

    candidates_by_strip = []

    # 후보 에지 추출
    for x1, x2 in strips:
        profile = smooth_1d(
            np.median(gy[:, x1:x2], axis=1),
            cfg.profile_smooth
        )

        peaks = local_peaks(
            profile,
            cfg.top_edge_min_strength
        )

        candidates = []

        for y, signed, strength in peaks:
            if not (y_min <= y <= y_max):
                continue

            # 상단 외곽은 두께가 큰 쪽(위쪽)에 위치.
            thickness = bottom_y - y

            upper_bonus = (
                cfg.top_upper_preference * thickness
            )

            anchor_penalty = 0.0
            if cfg.top_anchor_y is not None:
                anchor_local = cfg.top_anchor_y - y_offset
                anchor_penalty = (
                    cfg.top_anchor_weight
                    * abs(y - anchor_local)
                )

            score = (
                strength
                + upper_bonus
                - anchor_penalty
            )

            candidates.append({
                "y": int(y),
                "signed": float(signed),
                "strength": float(strength),
                "score": float(score),
            })

        candidates.sort(
            key=lambda p: p["score"],
            reverse=True
        )

        # strip별 상위 12개 후보 보관
        candidates_by_strip.append(candidates[:12])

    # --------------------------------------------------------
    # DP: X 방향으로 연속적인 경계 추적
    # --------------------------------------------------------

    n = len(strips)
    dp = []
    back = []

    for i in range(n):
        curr = candidates_by_strip[i]

        scores = np.full(
            len(curr),
            -np.inf,
            dtype=float
        )
        parents = np.full(
            len(curr),
            -1,
            dtype=int
        )

        for j, candidate in enumerate(curr):
            local_score = candidate["score"]

            if i == 0:
                scores[j] = local_score
                continue

            prev = candidates_by_strip[i - 1]

            for k, prev_candidate in enumerate(prev):
                if not np.isfinite(dp[-1][k]):
                    continue

                delta = abs(
                    candidate["y"] - prev_candidate["y"]
                )

                if delta > cfg.top_max_step:
                    continue

                score = (
                    dp[-1][k]
                    + local_score
                    - cfg.top_smooth_penalty * delta
                )

                if score > scores[j]:
                    scores[j] = score
                    parents[j] = k

        dp.append(scores)
        back.append(parents)

    # 가장 많은 strip을 연속으로 연결한 경로 우선
    best_path = []
    best_avg = -np.inf

    for end_i in range(n):
        if not len(dp[end_i]):
            continue

        for j in range(len(dp[end_i])):
            if not np.isfinite(dp[end_i][j]):
                continue

            path = []
            i, k = end_i, j

            while i >= 0 and k >= 0:
                path.append((i, k))
                k = back[i][k]
                i -= 1

            path.reverse()

            # 너무 짧은 경로 제외
            if len(path) < cfg.min_valid_strips:
                continue

            avg = dp[end_i][j] / len(path)

            # 연결 길이와 점수 모두 고려
            combined = (
                avg + 0.5 * len(path)
            )

            if combined > best_avg:
                best_avg = combined
                best_path = path

    if not best_path:
        return None, strips, candidates_by_strip

    x_points = []
    y_points = []

    for strip_idx, candidate_idx in best_path:
        x1, x2 = strips[strip_idx]
        candidate = candidates_by_strip[
            strip_idx
        ][candidate_idx]

        x_points.append((x1 + x2 - 1) / 2.0)
        y_points.append(float(candidate["y"]))

    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)

    if len(y_points) >= 3:
        y_points = smooth_1d(
            y_points,
            min(cfg.top_curve_smooth, len(y_points))
        ).astype(float)

    # 전체 ROI의 X 좌표에 대해 보간.
    # 단, 관측 범위 바깥쪽은 측정에 사용하지 않음.
    xs = np.arange(w, dtype=float)

    curve = np.interp(
        xs,
        x_points,
        y_points
    )

    valid = (
        (xs >= x_points[0])
        & (xs <= x_points[-1])
    )

    curve[~valid] = np.nan

    coverage = len(best_path) / max(1, n)

    result = {
        "curve": curve,
        "x_points": x_points,
        "y_points": y_points,
        "coverage": float(coverage),
        "mean_top_y": float(np.nanmedian(curve)),
        "mean_thickness": float(
            bottom_y - np.nanmedian(curve)
        ),
    }

    return result, strips, candidates_by_strip


def find_wafer_pairs(
    gy,
    blade_top_curve,
    blade_bottom,
    side,
    cfg,
):
    """
    Blade 외부에서 wafer 상·하단 edge pair 추출.

    각 wafer 후보는:
      - 얇은 수직 두께
      - 서로 반대 부호의 edge
      - 여러 X strip에서 존재
    를 만족해야 함.
    """
    h, w = gy.shape
    strips = create_x_strips(w, cfg)

    # Blade 상단은 X에 따라 다르므로 중앙값은
    # 후보 탐색 영역을 대략 제한하는 데만 사용.
    median_top = float(
        np.nanmedian(blade_top_curve)
    )

    detections = []

    for strip_idx, (x1, x2) in enumerate(strips):
        profile = smooth_1d(
            np.median(gy[:, x1:x2], axis=1),
            cfg.profile_smooth
        )

        peaks = local_peaks(
            profile,
            cfg.wafer_min_strength
        )

        candidates = []

        for i, p1 in enumerate(peaks):
            y1, s1, v1 = p1

            for p2 in peaks[i + 1:]:
                y2, s2, v2 = p2
                thickness = y2 - y1

                if thickness > cfg.wafer_max_thickness:
                    break

                if thickness < cfg.wafer_min_thickness:
                    continue

                if s1 * s2 >= 0:
                    continue

                strip_x = (x1 + x2 - 1) // 2
                local_top = blade_top_curve[strip_x]

                if not np.isfinite(local_top):
                    local_top = median_top

                if side == "upper":
                    # Blade 상단보다 위에 완전히 존재
                    gap = local_top - y2
                    facing_y = y2
                else:
                    # Blade 하단보다 아래에 완전히 존재
                    gap = y1 - blade_bottom
                    facing_y = y1

                if gap < max(
                    cfg.min_gap,
                    cfg.wafer_exclusion_px
                ):
                    continue

                if gap > cfg.max_gap:
                    continue

                strength = min(v1, v2)

                candidates.append({
                    "strip": strip_idx,
                    "x": float(strip_x),
                    "top_y": float(y1),
                    "bottom_y": float(y2),
                    "facing_y": float(facing_y),
                    "gap": float(gap),
                    "strength": float(strength),
                    "thickness": float(thickness),
                })

        detections.extend(candidates)

    if not detections:
        return None, [], strips

    # --------------------------------------------------------
    # 같은 수평 wafer를 하나의 물체로 묶기
    # --------------------------------------------------------

    # wafer를 대표하는 facing_y를 기준으로 그룹화
    detections.sort(
        key=lambda d: d["facing_y"]
    )

    groups = []

    for item in detections:
        matched = None

        for group in groups:
            median_y = np.median([
                x["facing_y"] for x in group
            ])

            if abs(item["facing_y"] - median_y) <= 4:
                matched = group
                break

        if matched is None:
            groups.append([item])
        else:
            matched.append(item)

    wafer_candidates = []

    for group in groups:
        unique_strips = len(set(
            x["strip"] for x in group
        ))

        support = unique_strips / max(1, len(strips))

        if support < cfg.wafer_min_support:
            continue

        median_gap = float(np.median([
            x["gap"] for x in group
        ]))

        median_strength = float(np.median([
            x["strength"] for x in group
        ]))

        # 가까운 wafer 우선 + 수평 연속성
        score = (
            120.0 / (1.0 + median_gap)
            + 20.0 * support
            + 0.3 * median_strength
        )

        wafer_candidates.append({
            "y": float(np.median([
                x["facing_y"] for x in group
            ])),
            "gap": median_gap,
            "support": float(support),
            "strength": median_strength,
            "score": float(score),
            "detections": group,
        })

    wafer_candidates.sort(
        key=lambda d: d["score"],
        reverse=True
    )

    selected = (
        wafer_candidates[0]
        if wafer_candidates
        else None
    )

    return selected, wafer_candidates, strips


def calculate_gap_statistics(
    wafer,
    blade_top_curve,
    blade_bottom,
    side,
    roi_width,
):
    if wafer is None:
        return None

    # 같은 X 위치의 wafer facing edge와
    # Blade 외곽 경계 간 차이 계산
    gaps = []

    for item in wafer["detections"]:
        x = int(round(item["x"]))

        if not (0 <= x < roi_width):
            continue

        if side == "upper":
            bt = blade_top_curve[x]
            if not np.isfinite(bt):
                continue
            gap = bt - item["facing_y"]
        else:
            gap = item["facing_y"] - blade_bottom

        if gap >= 0:
            gaps.append(float(gap))

    if not gaps:
        return None

    values = np.asarray(gaps, dtype=float)

    return {
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "count": int(len(values)),
    }


def analyze_image(image, cfg):
    roi, bounds = crop_roi(image, cfg.measure_roi)
    _, gy = prepare_edges(roi, cfg)

    bottom, bottom_candidates, bottom_profile = (
        detect_blade_bottom(gy, bounds, cfg)
    )

    debug = {
        "gy": gy,
        "bounds": bounds,
        "bottom_profile": bottom_profile,
        "bottom_candidates": bottom_candidates,
    }

    if bottom is None:
        return {
            "status": "blade_bottom_not_found",
        }, debug

    bottom_y = bottom["y_local"]

    top, strips, top_candidates = detect_blade_top_curve(
        gy,
        bottom_y,
        bounds,
        cfg
    )

    debug["top_strips"] = strips
    debug["top_candidates"] = top_candidates

    if top is None:
        return {
            "status": "blade_top_not_found",
            "blade_bottom_y": bottom_y,
        }, debug

    debug["top_curve"] = top["curve"]

    if top["coverage"] < cfg.min_valid_fraction:
        return {
            "status": "blade_top_low_coverage",
            "blade_bottom_y": bottom_y,
            "blade_top_coverage": top["coverage"],
        }, debug

    upper, upper_candidates, _ = find_wafer_pairs(
        gy,
        top["curve"],
        bottom_y,
        "upper",
        cfg
    )

    lower, lower_candidates, _ = find_wafer_pairs(
        gy,
        top["curve"],
        bottom_y,
        "lower",
        cfg
    )

    debug["upper_candidates"] = upper_candidates
    debug["lower_candidates"] = lower_candidates

    upper_gap = calculate_gap_statistics(
        upper, top["curve"], bottom_y,
        "upper", gy.shape[1]
    )

    lower_gap = calculate_gap_statistics(
        lower, top["curve"], bottom_y,
        "lower", gy.shape[1]
    )

    top_gap = upper_gap["median"] if upper_gap else None
    bottom_gap = lower_gap["median"] if lower_gap else None

    top_ratio = bottom_ratio = None

    if top_gap is not None and bottom_gap is not None:
        total = top_gap + bottom_gap
        if total > 0:
            top_ratio = top_gap / total * 100.0
            bottom_ratio = bottom_gap / total * 100.0

    if upper_gap is None and lower_gap is None:
        status = "no_wafer_detected"
    elif upper_gap is None or lower_gap is None:
        status = "partial"
    else:
        status = "ok"

    if (
        len(upper_candidates) > 1
        and upper_candidates[1]["score"]
        >= upper_candidates[0]["score"] * cfg.ambiguity_ratio
    ):
        status = "review"

    if (
        len(lower_candidates) > 1
        and lower_candidates[1]["score"]
        >= lower_candidates[0]["score"] * cfg.ambiguity_ratio
    ):
        status = "review"

    return {
        "status": status,
        "blade_bottom_y": float(bottom_y),
        "blade_top_y_median": top["mean_top_y"],
        "blade_thickness_median": top["mean_thickness"],
        "blade_top_coverage": top["coverage"],
        "blade_top_curve": top["curve"],

        "upper_wafer_y": upper["y"] if upper else None,
        "lower_wafer_y": lower["y"] if lower else None,

        "upper_status": (
            "detected" if upper else "not_detected"
        ),
        "lower_status": (
            "detected" if lower else "not_detected"
        ),

        "top_gap": upper_gap,
        "bottom_gap": lower_gap,

        "top_gap_px": top_gap,
        "bottom_gap_px": bottom_gap,
        "top_ratio": top_ratio,
        "bottom_ratio": bottom_ratio,
    }, debug


def draw_overlay(image, result, cfg, debug):
    out = image.copy()
    x, y, w, h = cfg.measure_roi

    cv2.rectangle(
        out, (x, y), (x + w, y + h),
        (0, 255, 255), 2
    )

    bx, by, bw, bh = cfg.blade_roi
    cv2.rectangle(
        out,
        (bx, by),
        (bx + bw, by + bh),
        (255, 150, 0), 2
    )

    if "blade_top_curve" not in result:
        return out

    curve = result["blade_top_curve"]

    # Blade 상단 곡선: 빨간색
    pts = []

    for lx, local_y in enumerate(curve):
        if np.isfinite(local_y):
            pts.append([
                int(x + lx),
                int(y + local_y),
            ])

    if len(pts) >= 2:
        cv2.polylines(
            out,
            [np.asarray(pts, dtype=np.int32)],
            False,
            (0, 0, 255), 2,
            cv2.LINE_AA
        )

    # Blade 하단: 파란색
    bottom_y = int(
        round(y + result["blade_bottom_y"])
    )

    cv2.line(
        out,
        (x, bottom_y),
        (x + w, bottom_y),
        (255, 0, 0), 2
    )

    # 상단 Wafer facing edge: 초록색
    if result["upper_wafer_y"] is not None:
        wy = int(round(
            y + result["upper_wafer_y"]
        ))
        cv2.line(
            out, (x, wy), (x + w, wy),
            (0, 255, 0), 2
        )

    # 하단 Wafer facing edge: 주황색
    if result["lower_wafer_y"] is not None:
        wy = int(round(
            y + result["lower_wafer_y"]
        ))
        cv2.line(
            out, (x, wy), (x + w, wy),
            (0, 165, 255), 2
        )

    return out


def create_detail(overlay, result, cfg):
    roi, _ = crop_roi(
        overlay,
        cfg.measure_roi
    )

    zoom = cv2.resize(
        roi,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_NEAREST
    )

    panel = np.full(
        (zoom.shape[0], 460, 3),
        30,
        dtype=np.uint8
    )

    def fmt(v):
        return "SKIP" if v is None else f"{v:.2f}"

    lines = [
        f"Status: {result['status']}",
        "",
        f"Top Gap: {fmt(result.get('top_gap_px'))} px",
        f"Bottom Gap: {fmt(result.get('bottom_gap_px'))} px",
        "",
        f"Top Ratio: {fmt(result.get('top_ratio'))} %",
        f"Bottom Ratio: {fmt(result.get('bottom_ratio'))} %",
        "",
        f"Top Coverage: {fmt(result.get('blade_top_coverage'))}",
        f"Upper: {result.get('upper_status', 'N/A')}",
        f"Lower: {result.get('lower_status', 'N/A')}",
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (15, 35 + 38 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return np.hstack([zoom, panel])


def save_debug(output_dir, stem, result, debug):
    gy = debug["gy"]
    y0 = debug["bounds"][1]

    # Y 방향 edge profile
    rows = []

    profile = debug["bottom_profile"]

    for i in range(len(profile)):
        rows.append({
            "y_local": i,
            "y_global": i + y0,
            "bottom_edge_profile": float(profile[i]),
        })

    pd.DataFrame(rows).to_csv(
        output_dir / "debug" / f"{stem}_profile.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Blade 상단 곡선
    if "top_curve" in debug:
        curve = debug["top_curve"]

        pd.DataFrame({
            "x_local": np.arange(len(curve)),
            "x_global": (
                np.arange(len(curve))
                + debug["bounds"][0]
            ),
            "blade_top_y_local": curve,
            "blade_top_y_global": curve + y0,
        }).to_csv(
            output_dir / "debug" / f"{stem}_top_curve.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # Wafer 후보
    candidate_rows = []

    for side in ("upper", "lower"):
        for rank, c in enumerate(
            debug.get(f"{side}_candidates", []),
            1
        ):
            candidate_rows.append({
                "side": side,
                "rank": rank,
                "y_local": c["y"],
                "gap_px": c["gap"],
                "support": c["support"],
                "strength": c["strength"],
                "score": c["score"],
            })

    pd.DataFrame(
        candidate_rows,
        columns=[
            "side", "rank", "y_local", "gap_px",
            "support", "strength", "score"
        ]
    ).to_csv(
        output_dir / "debug" / f"{stem}_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Sobel visualization
    edge = cv2.convertScaleAbs(
        gy,
        alpha=0.25
    )
    cv2.imwrite(
        str(output_dir / "debug" / f"{stem}_sobel.png"),
        edge
    )


def flatten_result(name, result):
    top = result.get("top_gap")
    bottom = result.get("bottom_gap")

    def get(d, key):
        return d.get(key) if isinstance(d, dict) else None

    return {
        "image": name,
        "status": result["status"],

        "blade_top_y_local": result.get("blade_top_y_median"),
        "blade_bottom_y_local": result.get("blade_bottom_y"),
        "blade_thickness_px": result.get(
            "blade_thickness_median"
        ),
        "blade_top_coverage": result.get(
            "blade_top_coverage"
        ),

        "upper_wafer_y_local": result.get("upper_wafer_y"),
        "lower_wafer_y_local": result.get("lower_wafer_y"),

        "upper_status": result.get("upper_status"),
        "lower_status": result.get("lower_status"),

        "top_gap_px": result.get("top_gap_px"),
        "top_gap_min_px": get(top, "min"),
        "top_gap_max_px": get(top, "max"),
        "top_gap_std_px": get(top, "std"),

        "bottom_gap_px": result.get("bottom_gap_px"),
        "bottom_gap_min_px": get(bottom, "min"),
        "bottom_gap_max_px": get(bottom, "max"),
        "bottom_gap_std_px": get(bottom, "std"),

        "top_ratio": result.get("top_ratio"),
        "bottom_ratio": result.get("bottom_ratio"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase B v2: Wafer-Blade Gap Measurement"
    )

    parser.add_argument("--images", required=True)
    parser.add_argument(
        "--output",
        default="output/06_phase_b_v2"
    )

    parser.add_argument(
        "--blade-roi", nargs=4, type=int,
        default=[41, 358, 2027, 186]
    )
    parser.add_argument(
        "--measure-roi", nargs=4, type=int,
        default=[49, 296, 381, 273]
    )

    parser.add_argument(
        "--blade-bottom-y", type=float,
        default=None
    )

    parser.add_argument(
        "--blade-thickness", nargs=2, type=int,
        default=[10, 75]
    )

    parser.add_argument(
        "--strip-width", type=int, default=12
    )
    parser.add_argument(
        "--strip-step", type=int, default=6
    )
    parser.add_argument(
        "--top-edge-strength", type=float,
        default=7.0
    )
    parser.add_argument(
        "--top-upper-preference", type=float,
        default=0.20
    )
    parser.add_argument(
        "--top-smooth-penalty", type=float,
        default=2.0
    )
    parser.add_argument(
        "--top-max-step", type=int,
        default=5
    )
    parser.add_argument(
        "--top-anchor-y", type=float,
        default=None
    )
    parser.add_argument(
        "--top-anchor-weight", type=float,
        default=0.0
    )

    parser.add_argument(
        "--wafer-thickness", nargs=2, type=int,
        default=[1, 9]
    )
    parser.add_argument(
        "--wafer-min-support", type=float,
        default=0.25
    )
    parser.add_argument(
        "--wafer-min-strength", type=float,
        default=7.0
    )
    parser.add_argument(
        "--max-gap", type=float, default=110.0
    )
    parser.add_argument(
        "--min-gap", type=float, default=3.0
    )

    args = parser.parse_args()

    cfg = Config(
        blade_roi=tuple(args.blade_roi),
        measure_roi=tuple(args.measure_roi),
        blade_bottom_y=args.blade_bottom_y,
        blade_min_thickness=args.blade_thickness[0],
        blade_max_thickness=args.blade_thickness[1],
        strip_width=args.strip_width,
        strip_step=args.strip_step,
        top_edge_min_strength=args.top_edge_strength,
        top_upper_preference=args.top_upper_preference,
        top_smooth_penalty=args.top_smooth_penalty,
        top_max_step=args.top_max_step,
        top_anchor_y=args.top_anchor_y,
        top_anchor_weight=args.top_anchor_weight,
        wafer_min_thickness=args.wafer_thickness[0],
        wafer_max_thickness=args.wafer_thickness[1],
        wafer_min_support=args.wafer_min_support,
        wafer_min_strength=args.wafer_min_strength,
        max_gap=args.max_gap,
        min_gap=args.min_gap,
    )

    if cfg.blade_min_thickness >= cfg.blade_max_thickness:
        parser.error("blade-thickness MIN must be < MAX")
    if cfg.wafer_min_thickness > cfg.wafer_max_thickness:
        parser.error("wafer-thickness MIN must be <= MAX")
    if cfg.strip_width < 1 or cfg.strip_step < 1:
        parser.error("strip-width and strip-step must be positive")

    image_dir = Path(args.images)
    output_dir = Path(args.output)

    for sub in ["overlay", "detail", "debug"]:
        (output_dir / sub).mkdir(
            parents=True,
            exist_ok=True
        )

    files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }
    )

    if not files:
        raise FileNotFoundError(
            f"No image files: {image_dir}"
        )

    print("=" * 65)
    print("Phase B v2 - Blade Top Contour / Wafer Gap")
    print("=" * 65)
    print(f"Images          : {len(files)}")
    print(f"Blade ROI       : {cfg.blade_roi}")
    print(f"Measure ROI     : {cfg.measure_roi}")
    print(f"Blade thickness : "
          f"{cfg.blade_min_thickness}~"
          f"{cfg.blade_max_thickness}")
    print()

    rows = []

    for i, path in enumerate(files, 1):
        img = cv2.imread(str(path))

        if img is None:
            rows.append({
                "image": path.name,
                "status": "image_read_failed"
            })
            continue

        try:
            result, debug = analyze_image(
                img, cfg
            )

            overlay = draw_overlay(
                img, result, cfg, debug
            )

            detail = create_detail(
                overlay, result, cfg
            )

            cv2.imwrite(
                str(output_dir / "overlay" / path.name),
                overlay
            )

            cv2.imwrite(
                str(output_dir / "detail" / path.name),
                detail
            )

            save_debug(
                output_dir, path.stem,
                result, debug
            )

            row = flatten_result(
                path.name, result
            )

        except Exception as exc:
            row = {
                "image": path.name,
                "status": "error",
                "error": str(exc)
            }

        rows.append(row)

        print(
            f"[{i:02d}/{len(files):02d}] "
            f"{path.name} | "
            f"{row['status']} | "
            f"Top={row.get('top_gap_px')} | "
            f"Bottom={row.get('bottom_gap_px')}"
        )

    result_df = pd.DataFrame(rows)

    result_df.to_csv(
        output_dir / "geometry_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 65)
    print("Phase B v2 Result")
    print("=" * 65)
    print(result_df["status"].value_counts().to_string())
    print()
    print(f"Result: {output_dir / 'geometry_results.csv'}")


if __name__ == "__main__":
    main()
