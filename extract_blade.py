"""
extract_blade_motion_clips.py

목적
----
장시간 CCTV MP4에서 Blade 동작 후보 구간을 자동 검출하여
개별 MP4 클립과 CSV로 저장한다.

검출 방식
--------
Blade ROI
    ↓
Optical Flow (Farneback)
    ↓
ROI 내 움직이는 픽셀 비율 + 평균/중앙 Flow 크기
    ↓
MOVE / STOP 판정
    ↓
MOVE -> STOP -> MOVE 패턴 검출
    ↓
앞/뒤 여유시간 포함하여 Clip 저장

학습 모델 사용 없음.

출력
----
output/
├─ clips/
│  ├─ blade_event_0001.mp4
│  ├─ blade_event_0002.mp4
│  └─ ...
├─ blade_events.csv
└─ blade_motion_debug.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 기본 설정
# ============================================================

# Optical Flow 분석용 ROI 폭
# 원본 ROI가 커도 이 크기로 축소해서 분석하므로 속도가 빨라짐
ANALYSIS_WIDTH = 320

# Optical Flow magnitude가 이 값 이상인 픽셀만
# "실제 움직임 픽셀"로 취급
PIXEL_MOTION_THRESHOLD = 0.7

# ROI 전체 픽셀 중 움직이는 픽셀 비율
# 0.02 = 2%
MIN_ACTIVE_RATIO = 0.02

# 움직이는 픽셀들의 Optical Flow magnitude 중앙값
MIN_FLOW_MAGNITUDE = 0.8

# MOVE 판정이 최소 이 시간 이상 지속되어야 실제 MOVE로 인정
MIN_MOVE_DURATION_SEC = 0.20

# STOP 판정이 최소 이 시간 이상 지속되어야 실제 STOP으로 인정
MIN_STOP_DURATION_SEC = 0.30

# 첫 MOVE와 두 번째 MOVE 사이의 STOP 허용 범위
# "Blade가 들어감 -> 잠시 멈춤 -> 다시 움직임" 패턴의 핵심
INTERMEDIATE_STOP_MIN_SEC = 0.20
INTERMEDIATE_STOP_MAX_SEC = 5.0

# 하나의 MOVE가 너무 짧으면 노이즈로 취급
EVENT_MIN_MOVE_SEC = 0.20

# 클립 시작 전 추가할 시간
PRE_EVENT_SEC = 2.0

# 클립 종료 후 추가할 시간
POST_EVENT_SEC = 2.0

# 이벤트 전체가 지나치게 길면 제외
MAX_EVENT_DURATION_SEC = 20.0

# Debug CSV 저장 여부
SAVE_DEBUG_CSV = True


# ============================================================
# Utility
# ============================================================

def format_time(seconds: float) -> str:
    """초 -> HH:MM:SS.mmm"""

    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def resize_keep_ratio(image, width: int):
    h, w = image.shape[:2]

    if w <= width:
        return image

    scale = width / w
    new_h = int(round(h * scale))

    return cv2.resize(
        image,
        (width, new_h),
        interpolation=cv2.INTER_AREA,
    )


def select_roi_scaled(frame, max_width=1280, max_height=800):
    """
    큰 CCTV 영상에서도 ROI 선택창이 화면 밖으로 나가지 않도록
    Preview를 축소해서 ROI를 선택한다.
    """

    h, w = frame.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1.0,
    )

    if scale < 1.0:
        preview = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = frame.copy()

    print()
    print("Blade가 움직이는 영역을 ROI로 선택하세요.")
    print("선택 후 ENTER 또는 SPACE")
    print("취소: ESC")

    roi = cv2.selectROI(
        "Select Blade ROI",
        preview,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow("Select Blade ROI")

    x, y, rw, rh = roi

    if rw == 0 or rh == 0:
        raise RuntimeError("ROI가 선택되지 않았습니다.")

    # 원본 좌표 복원
    x = int(round(x / scale))
    y = int(round(y / scale))
    rw = int(round(rw / scale))
    rh = int(round(rh / scale))

    return x, y, rw, rh


# ============================================================
# Motion Analysis
# ============================================================

@dataclass
class MotionFrame:
    frame_idx: int
    time_sec: float
    motion_score: float
    active_ratio: float
    flow_x: float
    flow_y: float
    moving: bool


def calculate_motion(prev_gray, curr_gray):
    """
    Dense Optical Flow를 이용하여 Blade ROI의 움직임 계산
    """

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    dx = flow[..., 0]
    dy = flow[..., 1]

    magnitude = np.sqrt(dx * dx + dy * dy)

    active_mask = magnitude >= PIXEL_MOTION_THRESHOLD

    active_ratio = float(np.mean(active_mask))

    if np.any(active_mask):

        active_mag = magnitude[active_mask]

        motion_score = float(
            np.median(active_mag)
        )

        flow_x = float(
            np.median(dx[active_mask])
        )

        flow_y = float(
            np.median(dy[active_mask])
        )

    else:

        motion_score = 0.0
        flow_x = 0.0
        flow_y = 0.0

    moving = (
        active_ratio >= MIN_ACTIVE_RATIO
        and motion_score >= MIN_FLOW_MAGNITUDE
    )

    return (
        motion_score,
        active_ratio,
        flow_x,
        flow_y,
        moving,
    )


# ============================================================
# MOVE 구간 생성
# ============================================================

def build_motion_segments(
    motion_frames: list[MotionFrame],
):
    """
    Frame 단위 MOVE/STOP 결과를 연속 Segment로 변환

    반환:
        [
            {
                state: MOVE / STOP,
                start_time:
                end_time:
                duration:
            }
        ]
    """

    if not motion_frames:
        return []

    raw_segments = []

    start_idx = 0
    current_state = motion_frames[0].moving

    for i in range(1, len(motion_frames)):

        state = motion_frames[i].moving

        if state != current_state:

            start = motion_frames[start_idx]
            end = motion_frames[i - 1]

            raw_segments.append(
                {
                    "moving": current_state,
                    "start_frame": start.frame_idx,
                    "end_frame": end.frame_idx,
                    "start_time": start.time_sec,
                    "end_time": end.time_sec,
                    "duration": end.time_sec - start.time_sec,
                }
            )

            start_idx = i
            current_state = state

    start = motion_frames[start_idx]
    end = motion_frames[-1]

    raw_segments.append(
        {
            "moving": current_state,
            "start_frame": start.frame_idx,
            "end_frame": end.frame_idx,
            "start_time": start.time_sec,
            "end_time": end.time_sec,
            "duration": end.time_sec - start.time_sec,
        }
    )

    return raw_segments


# ============================================================
# MOVE -> STOP -> MOVE 패턴 검색
# ============================================================

def detect_blade_events(segments):
    """
    목표 패턴

        MOVE
          ↓
        STOP
          ↓
        MOVE

    예:
        Blade 이동
        잠시 정지
        다시 Blade 이동

    현재 단계에서는 OUT/IN 방향은 구분하지 않는다.
    """

    events = []

    for i in range(len(segments) - 2):

        s1 = segments[i]
        s2 = segments[i + 1]
        s3 = segments[i + 2]

        # MOVE -> STOP -> MOVE
        if not s1["moving"]:
            continue

        if s2["moving"]:
            continue

        if not s3["moving"]:
            continue

        # MOVE 길이
        if s1["duration"] < EVENT_MIN_MOVE_SEC:
            continue

        if s3["duration"] < EVENT_MIN_MOVE_SEC:
            continue

        # 중간 정지 시간
        if not (
            INTERMEDIATE_STOP_MIN_SEC
            <= s2["duration"]
            <= INTERMEDIATE_STOP_MAX_SEC
        ):
            continue

        start_time = s1["start_time"]
        end_time = s3["end_time"]

        total_duration = end_time - start_time

        if total_duration > MAX_EVENT_DURATION_SEC:
            continue

        events.append(
            {
                "move1_start": s1["start_time"],
                "move1_end": s1["end_time"],

                "stop_start": s2["start_time"],
                "stop_end": s2["end_time"],
                "stop_duration": s2["duration"],

                "move2_start": s3["start_time"],
                "move2_end": s3["end_time"],

                "event_start": start_time,
                "event_end": end_time,
            }
        )

    return events


# ============================================================
# Clip 저장
# ============================================================

def extract_clip(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"영상 열기 실패: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    start_frame = max(
        0,
        int(start_sec * fps),
    )

    end_frame = int(
        end_sec * fps
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    frame_idx = start_frame

    while frame_idx <= end_frame:

        ret, frame = cap.read()

        if not ret:
            break

        writer.write(frame)

        frame_idx += 1

    writer.release()
    cap.release()


# ============================================================
# Main Analysis
# ============================================================

def analyze_video(video_path: Path, output_dir: Path):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clips_dir = output_dir / "clips"

    clips_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"영상 열기 실패: {video_path}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    print()
    print("===================================")
    print("Video")
    print("===================================")
    print(f"path       : {video_path}")
    print(f"fps        : {fps:.3f}")
    print(f"frames     : {frame_count}")
    print(f"duration   : {format_time(duration)}")
    print()

    ret, first_frame = cap.read()

    if not ret:
        raise RuntimeError(
            "첫 Frame을 읽을 수 없습니다."
        )

    roi = select_roi_scaled(
        first_frame
    )

    x, y, w, h = roi

    print()
    print(
        f"Blade ROI = "
        f"x={x}, y={y}, w={w}, h={h}"
    )

    # 첫 ROI
    prev_roi = first_frame[
        y:y + h,
        x:x + w,
    ]

    prev_roi = resize_keep_ratio(
        prev_roi,
        ANALYSIS_WIDTH,
    )

    prev_gray = cv2.cvtColor(
        prev_roi,
        cv2.COLOR_BGR2GRAY,
    )

    prev_gray = cv2.GaussianBlur(
        prev_gray,
        (5, 5),
        0,
    )

    motion_frames = []

    frame_idx = 1

    print()
    print("Motion 분석 시작...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        roi_img = frame[
            y:y + h,
            x:x + w,
        ]

        roi_img = resize_keep_ratio(
            roi_img,
            ANALYSIS_WIDTH,
        )

        gray = cv2.cvtColor(
            roi_img,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        (
            motion_score,
            active_ratio,
            flow_x,
            flow_y,
            moving,
        ) = calculate_motion(
            prev_gray,
            gray,
        )

        time_sec = frame_idx / fps

        motion_frames.append(
            MotionFrame(
                frame_idx=frame_idx,
                time_sec=time_sec,
                motion_score=motion_score,
                active_ratio=active_ratio,
                flow_x=flow_x,
                flow_y=flow_y,
                moving=moving,
            )
        )

        prev_gray = gray

        frame_idx += 1

        if frame_idx % int(max(fps * 60, 1)) == 0:

            current_time = frame_idx / fps

            print(
                f"processing "
                f"{format_time(current_time)} "
                f"/ {format_time(duration)}"
            )

    cap.release()

    print()
    print("Motion 분석 완료")

    # --------------------------------------------------------
    # Debug CSV
    # --------------------------------------------------------

    if SAVE_DEBUG_CSV:

        debug_path = (
            output_dir
            / "blade_motion_debug.csv"
        )

        with open(
            debug_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "frame_idx",
                    "time_sec",
                    "video_time",
                    "motion_score",
                    "active_ratio",
                    "flow_x",
                    "flow_y",
                    "moving",
                ]
            )

            for m in motion_frames:

                writer.writerow(
                    [
                        m.frame_idx,
                        f"{m.time_sec:.3f}",
                        format_time(m.time_sec),
                        f"{m.motion_score:.5f}",
                        f"{m.active_ratio:.5f}",
                        f"{m.flow_x:.5f}",
                        f"{m.flow_y:.5f}",
                        int(m.moving),
                    ]
                )

        print(
            f"Debug CSV: {debug_path}"
        )

    # --------------------------------------------------------
    # Segment 생성
    # --------------------------------------------------------

    segments = build_motion_segments(
        motion_frames
    )

    events = detect_blade_events(
        segments
    )

    print()
    print(
        f"Blade event 후보: "
        f"{len(events)}개"
    )

    # --------------------------------------------------------
    # Event CSV + Clip
    # --------------------------------------------------------

    csv_path = (
        output_dir
        / "blade_events.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "event_id",

                "clip_file",

                "original_event_start_sec",
                "original_event_start",

                "move1_end",

                "stop_start",
                "stop_end",
                "stop_duration_sec",

                "move2_start",

                "original_event_end_sec",
                "original_event_end",

                "clip_start_sec",
                "clip_start",

                "clip_end_sec",
                "clip_end",
            ]
        )

        for event_id, event in enumerate(
            events,
            start=1,
        ):

            clip_start = max(
                0.0,
                event["event_start"]
                - PRE_EVENT_SEC,
            )

            clip_end = min(
                duration,
                event["event_end"]
                + POST_EVENT_SEC,
            )

            clip_name = (
                f"blade_event_"
                f"{event_id:04d}.mp4"
            )

            clip_path = (
                clips_dir
                / clip_name
            )

            print()
            print(
                f"[{event_id:04d}] "
                f"{format_time(clip_start)}"
                f" -> "
                f"{format_time(clip_end)}"
            )

            extract_clip(
                video_path,
                clip_path,
                clip_start,
                clip_end,
            )

            writer.writerow(
                [
                    event_id,

                    clip_name,

                    f"{event['event_start']:.3f}",
                    format_time(
                        event["event_start"]
                    ),

                    format_time(
                        event["move1_end"]
                    ),

                    format_time(
                        event["stop_start"]
                    ),

                    format_time(
                        event["stop_end"]
                    ),

                    f"{event['stop_duration']:.3f}",

                    format_time(
                        event["move2_start"]
                    ),

                    f"{event['event_end']:.3f}",
                    format_time(
                        event["event_end"]
                    ),

                    f"{clip_start:.3f}",
                    format_time(
                        clip_start
                    ),

                    f"{clip_end:.3f}",
                    format_time(
                        clip_end
                    ),
                ]
            )

    print()
    print("===================================")
    print("완료")
    print("===================================")
    print(
        f"Event CSV : {csv_path}"
    )
    print(
        f"Clips     : {clips_dir}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="분석할 MP4 영상",
    )

    parser.add_argument(
        "--output",
        default="./blade_motion_result",
        help="결과 저장 폴더",
    )

    args = parser.parse_args()

    analyze_video(
        Path(args.video),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
