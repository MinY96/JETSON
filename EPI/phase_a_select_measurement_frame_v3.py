
"""
Phase A v3
==========

목적:
    CST UP 이후 temporal_ratio가 감소했다가
    다시 상승하는 전환 구간을 찾는다.

    그 구간에서 Blade OUT 직전으로 추정되는
    Measurement Frame을 선택한다.

입력:
    Phase A v2의 debug/*.csv
    Phase E-3의 clips/*.mp4

출력:
    measurement_frames.csv
    measurement_frames/*.jpg
    comparison/*.jpg
    candidates/*.jpg
    debug/*.csv

특징:
    - 기존 v2 debug CSV 재사용
    - 모델 학습 없음
    - 별도 OpenCV motion 추론 없음
    - 후보 프레임 3장 비교
    - v2 measurement frame과 비교
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass
class Config:

    # CST peak 이후 탐색 범위
    search_start_offset: int = 3
    search_end_offset: int = 15

    # 움직임 저점 후보의 절대 threshold
    valley_threshold: float = 0.06

    # 저점 이후 재상승 판단
    rise_threshold: float = 0.075
    rise_delta: float = 0.055
    rise_confirm_frames: int = 2

    # 저점으로부터 재상승을 찾는 최대 범위
    rise_search_frames: int = 8

    # 재상승 시작 직전 프레임
    measurement_lead_frames: int = 1

    # 저점 근처 변동 허용
    valley_tolerance: float = 0.02

    # 추정 측정 프레임과 저점 사이 최대 거리
    max_valley_distance: int = 3

    # 후보 이미지
    candidate_offsets: tuple[int, ...] = (-1, 0, 1)

    image_quality: int = 95


def find_cst_peak(df):
    idx = int(df["cst_dy"].idxmin())

    return {
        "frame_idx": int(df.iloc[idx]["frame_idx"]),
        "cst_dy": float(df.iloc[idx]["cst_dy"]),
        "clip_sec": float(df.iloc[idx]["clip_sec"]),
    }


def find_transition(df, cfg):
    """
    1. CST peak 이후 temporal_ratio 저점 검출
    2. 저점 이후 temporal_ratio 재상승 검출
    3. 재상승 직전 프레임 선택

    저점이 여러 개면 연속 저점의 마지막 위치를 고려한다.
    """

    df = df.sort_values("frame_idx").reset_index(drop=True)

    peak = find_cst_peak(df)
    peak_frame = peak["frame_idx"]

    frame_nums = df["frame_idx"].to_numpy(dtype=int)
    temporal = df["temporal_ratio"].to_numpy(dtype=float)

    start_frame = peak_frame + cfg.search_start_offset
    end_frame = peak_frame + cfg.search_end_offset

    positions = np.flatnonzero(
        (frame_nums >= start_frame) &
        (frame_nums <= end_frame)
    )

    if len(positions) == 0:
        return {
            "status": "search_window_empty",
            "peak": peak,
        }

    # --------------------------------------------------------
    # 1. Valley 탐색
    # --------------------------------------------------------

    valid = positions[np.isfinite(temporal[positions])]

    if len(valid) == 0:
        return {
            "status": "invalid_temporal_signal",
            "peak": peak,
        }

    min_value = float(np.min(temporal[valid]))

    if min_value > cfg.valley_threshold:
        return {
            "status": "valley_not_found",
            "peak": peak,
            "valley_value": min_value,
        }

    # 전체 최저점을 우선 사용
    min_positions = valid[
        temporal[valid] <= min_value + 1e-9
    ]

    valley_pos = int(min_positions[0])

    valley_frame = int(frame_nums[valley_pos])

    # --------------------------------------------------------
    # 2. Valley 이후 상승 시점 탐색
    # --------------------------------------------------------

    rise_pos = None

    search_end = min(
        len(df) - cfg.rise_confirm_frames + 1,
        valley_pos + cfg.rise_search_frames + 1,
    )

    for pos in range(valley_pos + 1, search_end):

        segment = temporal[
            pos:pos + cfg.rise_confirm_frames
        ]

        if len(segment) < cfg.rise_confirm_frames:
            break

        if not np.all(np.isfinite(segment)):
            continue

        threshold = max(
            cfg.rise_threshold,
            min_value + cfg.rise_delta,
        )

        # 충분한 움직임이 연속적으로 나타나는지
        if np.all(segment >= threshold):
            rise_pos = pos
            break

    if rise_pos is None:
        return {
            "status": "rise_not_found",
            "peak": peak,
            "valley_frame": valley_frame,
            "valley_value": min_value,
        }

    rise_frame = int(frame_nums[rise_pos])

    # --------------------------------------------------------
    # 3. Measurement frame 선택
    # --------------------------------------------------------

    measurement_frame = (
        rise_frame - cfg.measurement_lead_frames
    )

    # Measurement가 최저점에서 너무 멀면
    # 저점 이후 가까운 프레임으로 제한
    upper_bound = (
        valley_frame + cfg.max_valley_distance
    )

    measurement_frame = min(
        measurement_frame,
        upper_bound,
    )

    measurement_frame = max(
        measurement_frame,
        valley_frame,
    )

    measurement_frame = min(
        measurement_frame,
        int(frame_nums[-1]),
    )

    if measurement_frame <= peak_frame:
        return {
            "status": "invalid_sequence",
            "peak": peak,
            "valley_frame": valley_frame,
            "rise_frame": rise_frame,
        }

    matching = df.loc[
        df["frame_idx"] == measurement_frame
    ]

    if matching.empty:
        return {
            "status": "measurement_frame_missing",
            "peak": peak,
        }

    selected = matching.iloc[0]

    return {
        "status": "ok",
        "peak": peak,
        "valley_frame": valley_frame,
        "valley_value": min_value,
        "rise_frame": rise_frame,
        "rise_value": float(temporal[rise_pos]),
        "measurement_frame": measurement_frame,
        "measurement_sec": float(selected["clip_sec"]),
        "measurement_temporal": float(
            selected["temporal_ratio"]
        ),
        "measurement_cst_dy": float(selected["cst_dy"]),
    }


def read_exact_frames(video_path, wanted_frames):
    """
    3~5개 frame을 한 번에 순차적으로 읽는다.
    반복 VideoCapture 생성/seek 비용 최소화.
    """

    wanted = sorted(set(
        int(v) for v in wanted_frames if v >= 0
    ))

    if not wanted:
        return {}

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])

    results = {}
    position = wanted[0]
    wanted_set = set(wanted)
    last = wanted[-1]

    while position <= last:
        ok, frame = cap.read()

        if not ok:
            break

        if position in wanted_set:
            results[position] = frame.copy()

        position += 1

    cap.release()

    return results


def draw_label(frame, lines, color=(0, 255, 0)):
    output = frame.copy()

    # 정보 표시 영역
    cv2.rectangle(
        output,
        (0, 0),
        (650, 145),
        (15, 15, 15),
        -1,
    )

    for i, line in enumerate(lines):
        cv2.putText(
            output,
            str(line),
            (20, 35 + i * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def make_comparison(frames, frame_ids, labels, fps):
    """
    선택 프레임 전후 3장 가로 비교
    """

    images = []

    for frame_idx, label in zip(frame_ids, labels):
        if frame_idx not in frames:
            continue

        img = draw_label(
            frames[frame_idx],
            [
                label,
                f"Frame: {frame_idx}",
                f"Time: {frame_idx / fps:.3f} sec",
            ],
        )

        height, width = img.shape[:2]

        target_width = 800
        target_height = round(
            height * target_width / width
        )

        img = cv2.resize(
            img,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

        images.append(img)

    if not images:
        return None

    return np.hstack(images)


def analyze_one_clip(
    csv_path,
    clips_dir,
    output_dir,
    cfg,
    old_result=None,
):
    df = pd.read_csv(csv_path)

    required = {
        "frame_idx",
        "clip_sec",
        "cst_dy",
        "temporal_ratio",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path.name}: missing columns {missing}"
        )

    df = df.sort_values("frame_idx").reset_index(drop=True)

    clip_stem = csv_path.stem.removesuffix("_debug")

    clip_path = clips_dir / f"{clip_stem}.mp4"

    if not clip_path.exists():
        raise FileNotFoundError(clip_path)

    fps = 1.0 / float(
        df["clip_sec"].iloc[1] - df["clip_sec"].iloc[0]
    )

    detection = find_transition(df, cfg)
    peak = detection["peak"]

    result = {
        "clip": clip_path.name,
        "status": detection["status"],
        "fps": fps,
        "cst_peak_frame": peak["frame_idx"],
        "cst_peak_dy": peak["cst_dy"],
        "valley_frame": detection.get("valley_frame"),
        "valley_value": detection.get("valley_value"),
        "rise_frame": detection.get("rise_frame"),
        "rise_value": detection.get("rise_value"),
        "measurement_frame_idx": detection.get(
            "measurement_frame"
        ),
        "measurement_sec": detection.get(
            "measurement_sec"
        ),
        "measurement_cst_dy": detection.get(
            "measurement_cst_dy"
        ),
        "measurement_temporal": detection.get(
            "measurement_temporal"
        ),
    }

    if old_result is not None:
        old_frame = old_result.get(
            "measurement_frame_idx"
        )

        result["v2_frame"] = old_frame

        if (
            detection["status"] == "ok"
            and pd.notna(old_frame)
        ):
            result["delta_from_v2"] = (
                detection["measurement_frame"]
                - int(old_frame)
            )

    # --------------------------------------------------------
    # Debug CSV
    # --------------------------------------------------------

    debug_df = df.copy()

    debug_df["is_valley"] = (
        debug_df["frame_idx"] ==
        detection.get("valley_frame", -1)
    )

    debug_df["is_rise"] = (
        debug_df["frame_idx"] ==
        detection.get("rise_frame", -1)
    )

    debug_df["is_measurement_v3"] = (
        debug_df["frame_idx"] ==
        detection.get("measurement_frame", -1)
    )

    if old_result is not None:
        old_frame = old_result.get(
            "measurement_frame_idx", -1
        )

        if pd.isna(old_frame):
            old_frame = -1

        debug_df["is_measurement_v2"] = (
            debug_df["frame_idx"] == int(old_frame)
        )

    debug_df.to_csv(
        output_dir / "debug" /
        f"{clip_stem}_debug_v3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if detection["status"] != "ok":
        return result

    measurement = detection["measurement_frame"]
    valley = detection["valley_frame"]
    rise = detection["rise_frame"]

    # --------------------------------------------------------
    # Extract measurement and candidate frames
    # --------------------------------------------------------

    candidate_ids = [
        measurement + offset
        for offset in cfg.candidate_offsets
    ]

    wanted = [
        *candidate_ids,
        valley,
        rise,
    ]

    if old_result is not None:
        old_frame = old_result.get(
            "measurement_frame_idx"
        )

        if pd.notna(old_frame):
            wanted.append(int(old_frame))

    frames = read_exact_frames(
        clip_path,
        wanted,
    )

    if measurement not in frames:
        result["status"] = "image_read_failed"
        return result

    image_path = (
        output_dir / "measurement_frames" /
        f"{clip_stem}_measurement.jpg"
    )

    cv2.imwrite(
        str(image_path),
        frames[measurement],
        [cv2.IMWRITE_JPEG_QUALITY, cfg.image_quality],
    )

    result["measurement_image"] = str(image_path)

    # --------------------------------------------------------
    # Candidate comparison
    # --------------------------------------------------------

    labels = [
        f"Candidate {offset:+d}"
        for offset in cfg.candidate_offsets
    ]

    for i, offset in enumerate(cfg.candidate_offsets):
        if offset == 0:
            labels[i] = "MEASUREMENT"

    comparison = make_comparison(
        frames,
        candidate_ids,
        labels,
        fps,
    )

    if comparison is not None:
        path = (
            output_dir / "candidates" /
            f"{clip_stem}_candidates.jpg"
        )

        cv2.imwrite(str(path), comparison)
        result["candidate_image"] = str(path)

    # --------------------------------------------------------
    # Compare v2 and v3
    # --------------------------------------------------------

    if old_result is not None:
        old_frame = old_result.get(
            "measurement_frame_idx"
        )

        if pd.notna(old_frame):
            old_frame = int(old_frame)

            comparison = make_comparison(
                frames,
                [old_frame, measurement],
                ["V2 MEASUREMENT", "V3 MEASUREMENT"],
                fps,
            )

            if comparison is not None:
                path = (
                    output_dir / "comparison" /
                    f"{clip_stem}_v2_vs_v3.jpg"
                )

                cv2.imwrite(str(path), comparison)
                result["comparison_image"] = str(path)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Phase A v3 measurement selection"
    )

    parser.add_argument(
        "--debug",
        required=True,
        help="Phase A v2 debug CSV directory",
    )

    parser.add_argument(
        "--clips",
        required=True,
        help="Phase E-3 Type2 clips directory",
    )

    parser.add_argument(
        "--v2-results",
        default=None,
        help="Optional v2 measurement_frames.csv",
    )

    parser.add_argument(
        "--output",
        default="output/05_phase_a_v3",
    )

    parser.add_argument(
        "--valley-threshold",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--rise-threshold",
        type=float,
        default=0.075,
    )

    parser.add_argument(
        "--rise-delta",
        type=float,
        default=0.055,
    )

    parser.add_argument(
        "--lead-frames",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    cfg = Config(
        valley_threshold=args.valley_threshold,
        rise_threshold=args.rise_threshold,
        rise_delta=args.rise_delta,
        measurement_lead_frames=args.lead_frames,
    )

    debug_dir = Path(args.debug)
    clips_dir = Path(args.clips)
    output_dir = Path(args.output)

    for directory in [
        output_dir,
        output_dir / "debug",
        output_dir / "measurement_frames",
        output_dir / "comparison",
        output_dir / "candidates",
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    debug_files = sorted(
        debug_dir.glob("*_debug.csv")
    )

    if not debug_files:
        raise FileNotFoundError(
            f"No debug CSV files: {debug_dir}"
        )

    old_results = {}

    if args.v2_results:
        old_df = pd.read_csv(args.v2_results)

        old_results = {
            row["clip"]: row
            for _, row in old_df.iterrows()
        }

    print("=" * 65)
    print("Phase A v3 - Temporal Valley / Rise Detection")
    print("=" * 65)
    print(f"Clips              : {len(debug_files)}")
    print(f"Valley threshold   : {cfg.valley_threshold}")
    print(f"Rise threshold     : {cfg.rise_threshold}")
    print(f"Rise delta         : {cfg.rise_delta}")
    print(f"Lead frames        : {cfg.measurement_lead_frames}")
    print()

    results = []

    for i, csv_path in enumerate(debug_files, start=1):
        clip_name = csv_path.stem.removesuffix(
            "_debug"
        ) + ".mp4"

        try:
            result = analyze_one_clip(
                csv_path=csv_path,
                clips_dir=clips_dir,
                output_dir=output_dir,
                cfg=cfg,
                old_result=old_results.get(clip_name),
            )
        except Exception as exc:
            result = {
                "clip": clip_name,
                "status": "error",
                "error": str(exc),
            }

        results.append(result)

        if result["status"] == "ok":
            print(
                f"[{i:02d}/{len(debug_files):02d}] "
                f"{clip_name} | "
                f"peak={result['cst_peak_frame']} | "
                f"valley={result['valley_frame']} | "
                f"rise={result['rise_frame']} | "
                f"measure={result['measurement_frame_idx']} | "
                f"delta_v2={result.get('delta_from_v2', '-')}"
            )
        else:
            print(
                f"[{i:02d}/{len(debug_files):02d}] "
                f"{clip_name} | "
                f"FAILED: {result['status']}"
            )

    result_df = pd.DataFrame(results)

    csv_path = output_dir / "measurement_frames.csv"

    result_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    success = int(
        (result_df["status"] == "ok").sum()
    )

    print()
    print("=" * 65)
    print("Phase A v3 Result")
    print("=" * 65)
    print(f"Total   : {len(result_df)}")
    print(f"Success : {success}")
    print(f"Failed  : {len(result_df) - success}")
    print()
    print(f"CSV        : {csv_path}")
    print(f"Measurement: {output_dir / 'measurement_frames'}")
    print(f"Candidates : {output_dir / 'candidates'}")
    print(f"Comparison : {output_dir / 'comparison'}")


if __name__ == "__main__":
    main()
