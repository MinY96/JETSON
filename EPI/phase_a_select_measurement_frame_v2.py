
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
    # E-1과 동일한 CST ROI
    cst_roi: tuple[int, int, int, int] = (2039, 110, 149, 654)

    # E-2에서 가장 유효했던 Blade ROI
    blade_roi: tuple[int, int, int, int] = (309, 379, 1593, 236)

    cst_blur_ksize: int = 5
    cst_up_threshold: float = -2.0

    # Blade 분석용 축소 너비
    blade_analysis_width: int = 640

    # 기준 프레임과 비교하는 픽셀 변화 임계값
    blade_diff_threshold: int = 15

    # 프레임별 변화량의 노이즈 보정
    baseline_frames: int = 2
    onset_sigma: float = 3.0
    min_change_ratio: float = 0.008

    # Blade OUT 지속성 확인
    confirm_frames: int = 2
    min_growth_ratio: float = 0.003

    # CST peak 이후 탐색
    search_before_peak_frames: int = 0
    search_after_peak_sec: float = 1.2

    # 측정 시점: OUT 시작 직전
    measurement_lead_frames: int = 1

    # 영상에서 불필요한 테두리 제외
    inner_margin_ratio: float = 0.03

    # 디버그
    save_debug_csv: bool = True
    save_comparison: bool = True
    save_debug_video: bool = False


# ============================================================
# Image / ROI utilities
# ============================================================

def crop_roi(frame, roi):
    x, y, w, h = roi
    fh, fw = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI {roi}, image={fw}x{fh}")

    return frame[y1:y2, x1:x2]


def preprocess_cst(frame, cfg):
    roi = crop_roi(frame, cfg.cst_roi)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    k = cfg.cst_blur_ksize
    if k > 1:
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    return gray.astype(np.float32)


def preprocess_blade(frame, cfg):
    roi = crop_roi(frame, cfg.blade_roi)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    scale = cfg.blade_analysis_width / w
    target_h = max(1, round(h * scale))

    gray = cv2.resize(
        gray,
        (cfg.blade_analysis_width, target_h),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def read_video_frames(clip_path):
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {clip_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS: {clip_path}")
    if len(frames) < 8:
        raise RuntimeError(f"Too few frames: {clip_path}")

    return frames, fps


# ============================================================
# 1. CST phase correlation
# ============================================================

def calculate_cst_signal(frames, fps, cfg):
    cst_images = [preprocess_cst(f, cfg) for f in frames]

    h, w = cst_images[0].shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)

    rows = []

    for i, cur in enumerate(cst_images):
        if i == 0:
            dx, dy, response = 0.0, 0.0, 1.0
        else:
            (dx, dy), response = cv2.phaseCorrelate(
                cst_images[i - 1],
                cur,
                hann,
            )

        rows.append({
            "frame_idx": i,
            "clip_sec": i / fps,
            "cst_dx": float(dx),
            "cst_dy": float(dy),
            "cst_response": float(response),
        })

    return pd.DataFrame(rows)


def detect_cst_peak(signal_df, cfg):
    candidates = signal_df[
        signal_df["cst_dy"] <= cfg.cst_up_threshold
    ]

    if candidates.empty:
        return None

    pos = int(candidates["cst_dy"].idxmin())

    return {
        "frame_idx": pos,
        "time_sec": float(signal_df.iloc[pos]["clip_sec"]),
        "dy": float(signal_df.iloc[pos]["cst_dy"]),
    }


# ============================================================
# 2. Blade feature extraction
# ============================================================

def masked_image_difference(reference, current, threshold, margin):
    diff = cv2.absdiff(reference, current)

    h, w = diff.shape
    mx = int(w * margin)
    my = int(h * margin)

    valid = np.zeros_like(diff, dtype=np.uint8)
    valid[my:h - my if my else h,
          mx:w - mx if mx else w] = 255

    mask = np.uint8((diff >= threshold) & (valid > 0)) * 255

    # 작은 점 노이즈 제거
    kernel = np.ones((2, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    ratio = float(np.count_nonzero(mask)) / max(
        1, np.count_nonzero(valid)
    )

    energy = float(cv2.mean(diff, mask=valid)[0])

    # 수평 방향으로 변화가 발생한 영역의 범위
    column_count = np.count_nonzero(mask, axis=0)

    # 최소한 몇 행에서 변화가 발생해야 해당 column 활성화
    min_column_pixels = max(2, int(h * 0.04))

    active_x = np.flatnonzero(
        column_count >= min_column_pixels
    )

    if len(active_x):
        change_x_min = int(active_x.min())
        change_x_max = int(active_x.max())
        change_x_width = change_x_max - change_x_min + 1
    else:
        change_x_min = np.nan
        change_x_max = np.nan
        change_x_width = 0

    return {
        "change_ratio": ratio,
        "change_energy": energy,
        "change_x_min": change_x_min,
        "change_x_max": change_x_max,
        "change_x_width": change_x_width,
    }


def calculate_blade_signal(frames, fps, cst_peak_frame, cfg):
    blade_images = [preprocess_blade(f, cfg) for f in frames]

    # CST peak 직전 프레임을 'Blade가 들어온 상태' 기준으로 가정.
    # 첫 분석 이후 필요하면 별도 기준 프레임 선택 로직으로 개선.
    reference_idx = max(0, cst_peak_frame - 1)
    reference = blade_images[reference_idx]

    rows = []

    previous = None

    for i, image in enumerate(blade_images):
        features = masked_image_difference(
            reference,
            image,
            cfg.blade_diff_threshold,
            cfg.inner_margin_ratio,
        )

        if previous is None:
            temporal_ratio = 0.0
            temporal_energy = 0.0
        else:
            temporal = masked_image_difference(
                previous,
                image,
                cfg.blade_diff_threshold,
                cfg.inner_margin_ratio,
            )
            temporal_ratio = temporal["change_ratio"]
            temporal_energy = temporal["change_energy"]

        # 포커스 변화 참고 신호
        laplacian = cv2.Laplacian(image, cv2.CV_32F)
        sharpness = float(laplacian.var())

        rows.append({
            "frame_idx": i,
            "clip_sec": i / fps,
            **features,
            "temporal_ratio": temporal_ratio,
            "temporal_energy": temporal_energy,
            "sharpness": sharpness,
        })

        previous = image

    return pd.DataFrame(rows), reference_idx


# ============================================================
# 3. Blade OUT start detection
# ============================================================

def detect_blade_out(
    blade_df,
    cst_peak_frame,
    fps,
    cfg,
):
    """
    CST peak 이후 기준 영상 대비 변화가
    일정 수준 이상 지속되는 최초 시점을 찾는다.

    반환값은 실제 Blade OUT의 확정 위치가 아니라
    Blade OUT 시작 후보 위치다.
    """

    n = len(blade_df)

    start = max(
        1,
        cst_peak_frame - cfg.search_before_peak_frames,
    )

    end = min(
        n,
        cst_peak_frame
        + int(round(cfg.search_after_peak_sec * fps))
        + 1,
    )

    # peak 이전 2프레임으로 변화 신호 바닥값 추정
    base_end = max(1, cst_peak_frame)
    base_start = max(0, base_end - cfg.baseline_frames)

    baseline = blade_df.iloc[base_start:base_end][
        "change_ratio"
    ].to_numpy(dtype=float)

    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))

    threshold = max(
        cfg.min_change_ratio,
        baseline_mean + cfg.onset_sigma * baseline_std,
    )

    ratios = blade_df["change_ratio"].to_numpy(dtype=float)
    temporal = blade_df["temporal_ratio"].to_numpy(dtype=float)

    candidates = []

    # 연속적인 변화 증가 + 실제 움직임 존재 여부
    for i in range(start, end):
        last = i + cfg.confirm_frames

        if last > n or last > end:
            break

        window = ratios[i:last]
        motion_window = temporal[i:last]

        sufficiently_changed = np.all(window >= threshold)

        # 변화 누적이 진행되는지
        prev_ratio = ratios[max(0, i - 1)]
        growth = float(np.max(window) - prev_ratio)

        growing = growth >= cfg.min_growth_ratio

        moving = bool(
            np.max(motion_window) >= cfg.min_change_ratio
        )

        if sufficiently_changed and growing and moving:
            candidates.append({
                "frame_idx": i,
                "time_sec": i / fps,
                "threshold": threshold,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "growth": growth,
                "change_ratio": float(ratios[i]),
                "temporal_ratio": float(temporal[i]),
            })

    if not candidates:
        return None, threshold

    # 가장 먼저 나타난 유효 변화 구간
    return candidates[0], threshold


# ============================================================
# 4. Measurement selection
# ============================================================

def choose_measurement_frame(out_start, peak, cfg):
    idx = (
        out_start["frame_idx"]
        - cfg.measurement_lead_frames
    )

    # CST peak 이전이면 시퀀스와 맞지 않으므로 실패
    if idx < peak["frame_idx"]:
        return None

    return idx


# ============================================================
# 5. Image / Debug Output
# ============================================================

def put_lines(frame, lines, color=(0, 255, 0)):
    out = frame.copy()

    y = 45
    for line in lines:
        cv2.putText(
            out,
            line,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 37

    return out


def draw_frame_overlay(
    frame,
    cfg,
    label,
    clip_time,
    frame_idx,
):
    out = frame.copy()

    x, y, w, h = cfg.blade_roi

    cv2.rectangle(
        out,
        (x, y),
        (x + w, y + h),
        (0, 255, 255),
        2,
    )

    return put_lines(out, [
        label,
        f"Frame: {frame_idx}",
        f"Clip time: {clip_time:.3f}s",
    ])


def save_comparison_image(
    measurement_frame,
    out_frame,
    measurement_idx,
    out_idx,
    fps,
    path,
    cfg,
):
    left = draw_frame_overlay(
        measurement_frame,
        cfg,
        "MEASUREMENT - BEFORE OUT",
        measurement_idx / fps,
        measurement_idx,
    )

    right = draw_frame_overlay(
        out_frame,
        cfg,
        "BLADE OUT START",
        out_idx / fps,
        out_idx,
    )

    # 비교 영상은 축소하여 저장
    target_width = 960

    def resize(frame):
        h, w = frame.shape[:2]
        new_h = max(1, int(h * target_width / w))
        return cv2.resize(
            frame,
            (target_width, new_h),
            interpolation=cv2.INTER_AREA,
        )

    comparison = np.hstack([
        resize(left),
        resize(right),
    ])

    cv2.imwrite(str(path), comparison)


def save_debug_video(frames, signal_df, measurement_idx, out_idx,
                     fps, path, cfg):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot write video: {path}")

    for i, frame in enumerate(frames):
        row = signal_df.iloc[i]
        label = "BEFORE OUT"

        if i == measurement_idx:
            label = "MEASUREMENT"
        elif i >= out_idx:
            label = "BLADE OUT"

        out = draw_frame_overlay(frame, cfg, label, i / fps, i)

        cv2.putText(
            out,
            f"CST dy={row['cst_dy']:.3f} "
            f"Change={row['change_ratio']:.4f} "
            f"Temporal={row['temporal_ratio']:.4f}",
            (25, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        writer.write(out)

    writer.release()


# ============================================================
# 6. Analyze clip
# ============================================================

def analyze_clip(clip_path, output_dir, cfg):
    frames, fps = read_video_frames(clip_path)

    cst_df = calculate_cst_signal(frames, fps, cfg)
    peak = detect_cst_peak(cst_df, cfg)

    result = {
        "clip": clip_path.name,
        "fps": fps,
        "frame_count": len(frames),
        "status": "",
    }

    if peak is None:
        result["status"] = "cst_peak_not_found"
        return result

    result.update({
        "cst_peak_frame": peak["frame_idx"],
        "cst_peak_sec": peak["time_sec"],
        "cst_peak_dy": peak["dy"],
    })

    blade_df, reference_idx = calculate_blade_signal(
        frames,
        fps,
        peak["frame_idx"],
        cfg,
    )

    signal_df = cst_df.merge(
        blade_df.drop(columns=["clip_sec"]),
        on="frame_idx",
        how="left",
    )

    result["blade_reference_frame"] = reference_idx

    out_start, threshold = detect_blade_out(
        blade_df,
        peak["frame_idx"],
        fps,
        cfg,
    )

    result["out_threshold"] = threshold

    debug_dir = output_dir / "debug"
    measurement_dir = output_dir / "measurement_frames"
    comparison_dir = output_dir / "comparison"
    video_dir = output_dir / "debug_video"

    for directory in [
        debug_dir,
        measurement_dir,
        comparison_dir,
        video_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if out_start is None:
        result["status"] = "blade_out_not_found"
        measurement_idx = None
        out_idx = None
    else:
        out_idx = out_start["frame_idx"]

        measurement_idx = choose_measurement_frame(
            out_start,
            peak,
            cfg,
        )

        result.update({
            "blade_out_start_frame": out_idx,
            "blade_out_start_sec": out_idx / fps,
            "blade_out_change_ratio": out_start["change_ratio"],
            "blade_out_temporal_ratio": out_start["temporal_ratio"],
            "blade_out_growth": out_start["growth"],
        })

        if measurement_idx is None:
            result["status"] = "invalid_sequence"
        else:
            result["status"] = "ok"

    signal_df["is_cst_peak"] = (
        signal_df["frame_idx"] == peak["frame_idx"]
    )

    signal_df["is_blade_out_start"] = (
        signal_df["frame_idx"] == out_idx
        if out_idx is not None
        else False
    )

    signal_df["is_measurement"] = (
        signal_df["frame_idx"] == measurement_idx
        if measurement_idx is not None
        else False
    )

    signal_df["out_threshold"] = threshold

    if cfg.save_debug_csv:
        signal_df.to_csv(
            debug_dir / f"{clip_path.stem}_debug.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if measurement_idx is None:
        return result

    # --------------------------------------------------------
    # Measurement frame
    # --------------------------------------------------------

    measurement_path = (
        measurement_dir /
        f"{clip_path.stem}_measurement.jpg"
    )

    cv2.imwrite(
        str(measurement_path),
        frames[measurement_idx],
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_path = (
        comparison_dir /
        f"{clip_path.stem}_comparison.jpg"
    )

    if cfg.save_comparison:
        save_comparison_image(
            frames[measurement_idx],
            frames[out_idx],
            measurement_idx,
            out_idx,
            fps,
            comparison_path,
            cfg,
        )

    if cfg.save_debug_video:
        save_debug_video(
            frames,
            signal_df,
            measurement_idx,
            out_idx,
            fps,
            video_dir / f"{clip_path.stem}_debug.mp4",
            cfg,
        )

    result.update({
        "measurement_frame_idx": measurement_idx,
        "measurement_sec": measurement_idx / fps,
        "measurement_cst_dy": float(
            signal_df.iloc[measurement_idx]["cst_dy"]
        ),
        "measurement_change_ratio": float(
            signal_df.iloc[measurement_idx]["change_ratio"]
        ),
        "measurement_image": str(measurement_path),
        "comparison_image": (
            str(comparison_path)
            if cfg.save_comparison else ""
        ),
    })

    return result


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase A v2 - Blade OUT onset detection"
    )

    parser.add_argument("--clips", required=True)
    parser.add_argument("--output", default="phase_a_v2_result")

    parser.add_argument(
        "--blade-roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
    )

    parser.add_argument(
        "--blade-diff-threshold",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--min-change-ratio",
        type=float,
        default=0.008,
    )

    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--search-after-peak",
        type=float,
        default=1.2,
    )

    parser.add_argument(
        "--debug-video",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = Config(
        blade_diff_threshold=args.blade_diff_threshold,
        min_change_ratio=args.min_change_ratio,
        confirm_frames=args.confirm_frames,
        search_after_peak_sec=args.search_after_peak,
        save_debug_video=args.debug_video,
    )

    if args.blade_roi is not None:
        cfg.blade_roi = tuple(args.blade_roi)

    clips_dir = Path(args.clips)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(clips_dir.glob("*.mp4"))

    if not clips:
        raise FileNotFoundError(
            f"No MP4 clips found in {clips_dir}"
        )

    print("=" * 60)
    print("Phase A v2 - Blade OUT Measurement Selection")
    print("=" * 60)
    print(f"Clips              : {len(clips)}")
    print(f"CST ROI            : {cfg.cst_roi}")
    print(f"Blade ROI          : {cfg.blade_roi}")
    print(f"Blade diff         : {cfg.blade_diff_threshold}")
    print(f"Min change ratio   : {cfg.min_change_ratio}")
    print(f"Confirm frames     : {cfg.confirm_frames}")
    print(f"Search after peak  : {cfg.search_after_peak_sec}s")
    print()

    results = []

    for i, clip_path in enumerate(clips, start=1):
        try:
            result = analyze_clip(
                clip_path,
                output_dir,
                cfg,
            )
        except Exception as exc:
            result = {
                "clip": clip_path.name,
                "status": "error",
                "error": str(exc),
            }

        results.append(result)

        status = result["status"]

        if status == "ok":
            print(
                f"[{i:02d}/{len(clips):02d}] "
                f"{clip_path.name} | "
                f"CST={result['cst_peak_frame']} | "
                f"OUT={result['blade_out_start_frame']} | "
                f"MEASURE={result['measurement_frame_idx']} | "
                f"t={result['measurement_sec']:.3f}s"
            )
        else:
            print(
                f"[{i:02d}/{len(clips):02d}] "
                f"{clip_path.name} | FAILED: {status}"
            )

    result_df = pd.DataFrame(results)

    csv_path = output_dir / "measurement_frames.csv"
    result_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    success = int((result_df["status"] == "ok").sum())

    print()
    print("=" * 60)
    print("Phase A v2 Result")
    print("=" * 60)
    print(f"Total   : {len(result_df)}")
    print(f"Success : {success}")
    print(f"Failed  : {len(result_df) - success}")
    print()
    print(f"CSV        : {csv_path}")
    print(f"Measurement: {output_dir / 'measurement_frames'}")
    print(f"Comparison : {output_dir / 'comparison'}")
    print(f"Debug      : {output_dir / 'debug'}")


if __name__ == "__main__":
    main()
