"""
extract_blade_motion_clips.py

장시간 CCTV MP4에서 Blade 동작 후보를 자동 검출하여
개별 MP4 Clip + Event CSV + Frame Debug CSV로 저장한다.

검출 패턴
---------
MOVE -> 짧은 STOP -> MOVE

예상 실제 동작
-------------
Blade 진입
    ↓
순간 정지
    ↓
Blade 복귀

특징
----
- 학습 모델 사용 없음
- OpenCV Farneback Optical Flow 사용
- Blade ROI만 분석
- --roi 미입력 시 마우스로 ROI 선택
- --roi 입력 시 ROI 선택창 생략
- 주요 Threshold CLI 조절 가능
- 원본 영상 기준 시간 CSV 기록
- 이벤트 앞/뒤 여유시간 포함 Clip 저장

Example
-------
python extract_blade_motion_clips.py \
    --video cctv.mp4 \
    --output ./blade_result \
    --roi 1200 350 500 400

민감도 높이기:
python extract_blade_motion_clips.py \
    --video cctv.mp4 \
    --output ./blade_result_sensitive \
    --roi 1200 350 500 400 \
    --pixel-motion-thr 0.35 \
    --active-ratio-thr 0.008 \
    --flow-thr 0.35
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

@dataclass
class MotionConfig:

    # Optical Flow 계산용 ROI 크기
    analysis_width: int = 320

    # 개별 pixel을 motion pixel로 판단하는 최소 magnitude
    pixel_motion_threshold: float = 0.5

    # 전체 ROI 중 움직이는 pixel의 최소 비율
    # 0.015 = 1.5%
    min_active_ratio: float = 0.015

    # 활성 pixel들의 Optical Flow magnitude 중앙값 기준
    min_flow_magnitude: float = 0.5

    # 하나의 MOVE가 최소한 지속되어야 하는 시간
    event_min_move_sec: float = 0.15

    # MOVE -> STOP -> MOVE 사이의 STOP 허용시간
    intermediate_stop_min_sec: float = 0.05
    intermediate_stop_max_sec: float = 0.80

    # 전체 Blade 왕복 이벤트 최대시간
    max_event_duration_sec: float = 6.0

    # 추출 Clip 앞뒤 여유
    pre_event_sec: float = 1.0
    post_event_sec: float = 1.0


# ============================================================
# Data
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


# ============================================================
# Utility
# ============================================================

def format_time(seconds: float) -> str:
    """
    seconds -> HH:MM:SS.mmm
    """

    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def resize_keep_ratio(
    image: np.ndarray,
    width: int,
) -> np.ndarray:

    h, w = image.shape[:2]

    if w <= width:
        return image

    scale = width / w

    new_h = max(
        1,
        int(round(h * scale)),
    )

    return cv2.resize(
        image,
        (width, new_h),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# ROI
# ============================================================

def select_roi_scaled(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 800,
):
    """
    영상이 너무 큰 경우 Preview를 축소하여 ROI를 선택하고,
    선택 좌표를 다시 원본 영상 좌표로 변환한다.
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
    print("========================================")
    print("Blade ROI 선택")
    print("========================================")
    print("Blade가 실제로 움직이는 영역을 선택하세요.")
    print("확정 : ENTER / SPACE")
    print("취소 : ESC")
    print()

    roi = cv2.selectROI(
        "Select Blade ROI",
        preview,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        "Select Blade ROI"
    )

    x, y, rw, rh = roi

    if rw == 0 or rh == 0:
        raise RuntimeError(
            "ROI가 선택되지 않았습니다."
        )

    # Preview 좌표 -> 원본 영상 좌표
    x = int(round(x / scale))
    y = int(round(y / scale))

    rw = int(round(rw / scale))
    rh = int(round(rh / scale))

    return x, y, rw, rh


def validate_roi(
    roi,
    frame_width: int,
    frame_height: int,
):

    x, y, w, h = roi

    if x < 0 or y < 0:
        raise ValueError(
            f"ROI x/y는 0 이상이어야 합니다: {roi}"
        )

    if w <= 0 or h <= 0:
        raise ValueError(
            f"ROI width/height는 0보다 커야 합니다: {roi}"
        )

    if x + w > frame_width:
        raise ValueError(
            "ROI가 영상 width를 벗어났습니다.\n"
            f"Video width : {frame_width}\n"
            f"ROI         : {roi}"
        )

    if y + h > frame_height:
        raise ValueError(
            "ROI가 영상 height를 벗어났습니다.\n"
            f"Video height : {frame_height}\n"
            f"ROI          : {roi}"
        )


# ============================================================
# Optical Flow
# ============================================================

def calculate_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    config: MotionConfig,
):
    """
    Farneback Dense Optical Flow.

    반환값
    ------
    motion_score
        움직이는 pixel들의 magnitude 중앙값

    active_ratio
        전체 ROI 중 움직이는 pixel 비율

    flow_x
        움직이는 pixel의 x 방향 flow 중앙값

    flow_y
        움직이는 pixel의 y 방향 flow 중앙값

    moving
        최종 MOVE / STOP 판정
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

    magnitude = np.sqrt(
        dx * dx + dy * dy
    )

    # 충분히 움직인 pixel만 사용
    active_mask = (
        magnitude
        >= config.pixel_motion_threshold
    )

    active_ratio = float(
        np.mean(active_mask)
    )

    if np.any(active_mask):

        active_mag = magnitude[
            active_mask
        ]

        motion_score = float(
            np.median(active_mag)
        )

        flow_x = float(
            np.median(
                dx[active_mask]
            )
        )

        flow_y = float(
            np.median(
                dy[active_mask]
            )
        )

    else:

        motion_score = 0.0
        flow_x = 0.0
        flow_y = 0.0

    moving = (
        active_ratio
        >= config.min_active_ratio
        and
        motion_score
        >= config.min_flow_magnitude
    )

    return (
        motion_score,
        active_ratio,
        flow_x,
        flow_y,
        moving,
    )


# ============================================================
# MOVE / STOP Segment
# ============================================================

def build_motion_segments(
    motion_frames: list[MotionFrame],
):
    """
    Frame 단위 MOVE/STOP 판정을 연속 Segment로 변환.

    예:

        STOP STOP STOP
        MOVE MOVE MOVE
        STOP STOP
        MOVE MOVE MOVE

    ->

        STOP segment
        MOVE segment
        STOP segment
        MOVE segment
    """

    if not motion_frames:
        return []

    segments = []

    start_idx = 0

    current_state = (
        motion_frames[0].moving
    )

    for i in range(
        1,
        len(motion_frames),
    ):

        state = motion_frames[i].moving

        if state != current_state:

            start = motion_frames[
                start_idx
            ]

            end = motion_frames[
                i - 1
            ]

            segments.append(
                {
                    "moving": current_state,

                    "start_frame":
                        start.frame_idx,

                    "end_frame":
                        end.frame_idx,

                    "start_time":
                        start.time_sec,

                    "end_time":
                        end.time_sec,

                    "duration":
                        max(
                            0.0,
                            end.time_sec
                            - start.time_sec,
                        ),
                }
            )

            start_idx = i
            current_state = state

    # 마지막 Segment
    start = motion_frames[
        start_idx
    ]

    end = motion_frames[-1]

    segments.append(
        {
            "moving": current_state,

            "start_frame":
                start.frame_idx,

            "end_frame":
                end.frame_idx,

            "start_time":
                start.time_sec,

            "end_time":
                end.time_sec,

            "duration":
                max(
                    0.0,
                    end.time_sec
                    - start.time_sec,
                ),
        }
    )

    return segments


# ============================================================
# Blade Event Detection
# ============================================================

def detect_blade_events(
    segments,
    config: MotionConfig,
):
    """
    Blade 왕복 동작 후보 검출.

    목표 Pattern:

        MOVE
          ↓
        STOP
          ↓
        MOVE

    현재 단계에서는 OUT / IN 방향을 구분하지 않는다.

    이유:
    우선 후보 Clip을 최대한 확보하고,
    실제 flow_x / flow_y 값을 확인한 뒤
    OUT/IN 방향 threshold를 결정하는 것이 안전하다.
    """

    events = []

    for i in range(
        len(segments) - 2
    ):

        s1 = segments[i]
        s2 = segments[i + 1]
        s3 = segments[i + 2]

        # -----------------------------------------
        # MOVE -> STOP -> MOVE
        # -----------------------------------------

        if not s1["moving"]:
            continue

        if s2["moving"]:
            continue

        if not s3["moving"]:
            continue

        # -----------------------------------------
        # 첫 번째 MOVE 길이
        # -----------------------------------------

        if (
            s1["duration"]
            < config.event_min_move_sec
        ):
            continue

        # -----------------------------------------
        # 두 번째 MOVE 길이
        # -----------------------------------------

        if (
            s3["duration"]
            < config.event_min_move_sec
        ):
            continue

        # -----------------------------------------
        # 중간 STOP 길이
        # -----------------------------------------

        stop_duration = (
            s2["duration"]
        )

        if not (
            config.intermediate_stop_min_sec
            <= stop_duration
            <= config.intermediate_stop_max_sec
        ):
            continue

        # -----------------------------------------
        # 전체 Event
        # -----------------------------------------

        event_start = (
            s1["start_time"]
        )

        event_end = (
            s3["end_time"]
        )

        event_duration = (
            event_end
            - event_start
        )

        if (
            event_duration
            > config.max_event_duration_sec
        ):
            continue

        events.append(
            {
                "move1_start":
                    s1["start_time"],

                "move1_end":
                    s1["end_time"],

                "stop_start":
                    s2["start_time"],

                "stop_end":
                    s2["end_time"],

                "stop_duration":
                    stop_duration,

                "move2_start":
                    s3["start_time"],

                "move2_end":
                    s3["end_time"],

                "event_start":
                    event_start,

                "event_end":
                    event_end,

                "event_duration":
                    event_duration,
            }
        )

    return events


# ============================================================
# Clip Extraction
# ============================================================

def extract_clip(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
):
    """
    원본 영상에서 지정된 시간 구간을 MP4로 저장.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"영상 열기 실패: {video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if fps <= 0:

        cap.release()

        raise RuntimeError(
            "FPS 정보를 읽을 수 없습니다."
        )

    start_frame = max(
        0,
        int(round(
            start_sec * fps
        )),
    )

    end_frame = max(
        start_frame,
        int(round(
            end_sec * fps
        )),
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"VideoWriter 생성 실패: "
            f"{output_path}"
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
# Debug CSV
# ============================================================

def save_debug_csv(
    motion_frames: list[MotionFrame],
    output_path: Path,
):

    with open(
        output_path,
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

                    format_time(
                        m.time_sec
                    ),

                    f"{m.motion_score:.6f}",

                    f"{m.active_ratio:.6f}",

                    f"{m.flow_x:.6f}",

                    f"{m.flow_y:.6f}",

                    int(m.moving),
                ]
            )


# ============================================================
# Event CSV
# ============================================================

def save_event_csv(
    events,
    output_path: Path,
    video_path: Path,
    clips_dir: Path,
    video_duration: float,
    config: MotionConfig,
):

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "event_id",
                "clip_file",

                "event_start_sec",
                "event_start",

                "move1_end",

                "stop_start",
                "stop_end",
                "stop_duration_sec",

                "move2_start",

                "event_end_sec",
                "event_end",

                "event_duration_sec",

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
                - config.pre_event_sec,
            )

            clip_end = min(
                video_duration,
                event["event_end"]
                + config.post_event_sec,
            )

            clip_name = (
                f"blade_event_"
                f"{event_id:04d}.mp4"
            )

            clip_path = (
                clips_dir
                / clip_name
            )

            print(
                f"[EVENT {event_id:04d}] "
                f"{format_time(event['event_start'])}"
                f" -> "
                f"{format_time(event['event_end'])}"
                f" | clip "
                f"{format_time(clip_start)}"
                f" -> "
                f"{format_time(clip_end)}"
            )

            # Clip 생성
            extract_clip(
                video_path=video_path,
                output_path=clip_path,
                start_sec=clip_start,
                end_sec=clip_end,
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

                    f"{event['event_duration']:.3f}",

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


# ============================================================
# Main Analysis
# ============================================================

def analyze_video(
    video_path: Path,
    output_dir: Path,
    roi=None,
    config: MotionConfig | None = None,
):

    if config is None:
        config = MotionConfig()

    # --------------------------------------------------------
    # Output Directory
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clips_dir = (
        output_dir
        / "clips"
    )

    clips_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Open Video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"영상 열기 실패: {video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if fps <= 0:

        cap.release()

        raise RuntimeError(
            "영상 FPS 정보를 읽을 수 없습니다."
        )

    duration = (
        frame_count / fps
        if frame_count > 0
        else 0.0
    )

    print()
    print("========================================")
    print("VIDEO INFORMATION")
    print("========================================")
    print(f"Path       : {video_path}")
    print(f"Resolution : {frame_width} x {frame_height}")
    print(f"FPS        : {fps:.3f}")
    print(f"Frames     : {frame_count}")
    print(f"Duration   : {format_time(duration)}")

    # --------------------------------------------------------
    # First Frame
    # --------------------------------------------------------

    ret, first_frame = cap.read()

    if not ret:

        cap.release()

        raise RuntimeError(
            "첫 Frame을 읽을 수 없습니다."
        )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    if roi is None:

        print()
        print(
            "ROI Mode   : MANUAL"
        )

        roi = select_roi_scaled(
            first_frame
        )

    else:

        print()
        print(
            "ROI Mode   : CLI"
        )

    roi = tuple(
        int(v)
        for v in roi
    )

    validate_roi(
        roi,
        frame_width,
        frame_height,
    )

    x, y, w, h = roi

    print(
        f"Blade ROI  : "
        f"x={x}, y={y}, "
        f"w={w}, h={h}"
    )

    # --------------------------------------------------------
    # Configuration 출력
    # --------------------------------------------------------

    print()
    print("========================================")
    print("MOTION CONFIG")
    print("========================================")

    print(
        f"analysis_width        : "
        f"{config.analysis_width}"
    )

    print(
        f"pixel_motion_thr      : "
        f"{config.pixel_motion_threshold}"
    )

    print(
        f"active_ratio_thr      : "
        f"{config.min_active_ratio}"
    )

    print(
        f"flow_thr              : "
        f"{config.min_flow_magnitude}"
    )

    print(
        f"min_move_sec          : "
        f"{config.event_min_move_sec}"
    )

    print(
        f"stop_min_sec          : "
        f"{config.intermediate_stop_min_sec}"
    )

    print(
        f"stop_max_sec          : "
        f"{config.intermediate_stop_max_sec}"
    )

    print(
        f"max_event_sec         : "
        f"{config.max_event_duration_sec}"
    )

    print(
        f"pre_event_sec         : "
        f"{config.pre_event_sec}"
    )

    print(
        f"post_event_sec        : "
        f"{config.post_event_sec}"
    )

    # --------------------------------------------------------
    # First ROI
    # --------------------------------------------------------

    prev_roi = first_frame[
        y:y + h,
        x:x + w,
    ]

    prev_roi = resize_keep_ratio(
        prev_roi,
        config.analysis_width,
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

    # --------------------------------------------------------
    # Frame Analysis
    # --------------------------------------------------------

    motion_frames: list[
        MotionFrame
    ] = []

    frame_idx = 1

    print()
    print("========================================")
    print("MOTION ANALYSIS START")
    print("========================================")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # -----------------------------------------
        # ROI Crop
        # -----------------------------------------

        roi_img = frame[
            y:y + h,
            x:x + w,
        ]

        roi_img = resize_keep_ratio(
            roi_img,
            config.analysis_width,
        )

        # -----------------------------------------
        # Gray + Blur
        # -----------------------------------------

        gray = cv2.cvtColor(
            roi_img,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        # -----------------------------------------
        # Optical Flow
        # -----------------------------------------

        (
            motion_score,
            active_ratio,
            flow_x,
            flow_y,
            moving,
        ) = calculate_motion(
            prev_gray,
            gray,
            config,
        )

        time_sec = (
            frame_idx / fps
        )

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

        # -----------------------------------------
        # 진행 상황
        # 1분마다 출력
        # -----------------------------------------

        progress_interval = max(
            1,
            int(round(fps * 60))
        )

        if (
            frame_idx
            % progress_interval
            == 0
        ):

            current_time = (
                frame_idx / fps
            )

            if duration > 0:

                progress = (
                    current_time
                    / duration
                    * 100
                )

                print(
                    f"Processing "
                    f"{format_time(current_time)} "
                    f"/ "
                    f"{format_time(duration)} "
                    f"({progress:.1f}%)"
                )

            else:

                print(
                    f"Processing "
                    f"{format_time(current_time)}"
                )

    cap.release()

    print()
    print("Motion analysis completed.")
    print(
        f"Analyzed frames: "
        f"{len(motion_frames)}"
    )

    # --------------------------------------------------------
    # Debug CSV
    # --------------------------------------------------------

    debug_csv_path = (
        output_dir
        / "blade_motion_debug.csv"
    )

    save_debug_csv(
        motion_frames,
        debug_csv_path,
    )

    print(
        f"Debug CSV   : "
        f"{debug_csv_path}"
    )

    # --------------------------------------------------------
    # Segment 생성
    # --------------------------------------------------------

    segments = build_motion_segments(
        motion_frames
    )

    print(
        f"Segments    : "
        f"{len(segments)}"
    )

    # --------------------------------------------------------
    # Event 검출
    # --------------------------------------------------------

    events = detect_blade_events(
        segments,
        config,
    )

    print(
        f"Blade Events: "
        f"{len(events)}"
    )

    # --------------------------------------------------------
    # Event CSV + Clip
    # --------------------------------------------------------

    event_csv_path = (
        output_dir
        / "blade_events.csv"
    )

    save_event_csv(
        events=events,
        output_path=event_csv_path,
        video_path=video_path,
        clips_dir=clips_dir,
        video_duration=duration,
        config=config,
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("========================================")
    print("COMPLETE")
    print("========================================")

    print(
        f"Event CSV : "
        f"{event_csv_path}"
    )

    print(
        f"Debug CSV : "
        f"{debug_csv_path}"
    )

    print(
        f"Clips     : "
        f"{clips_dir}"
    )

    print(
        f"Events    : "
        f"{len(events)}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "CCTV Blade Motion "
            "Clip Extractor"
        )
    )

    # ========================================================
    # Input / Output
    # ========================================================

    parser.add_argument(
        "--video",
        required=True,
        help="분석할 MP4 영상 경로",
    )

    parser.add_argument(
        "--output",
        default="./blade_motion_result",
        help=(
            "결과 저장 폴더 "
            "(default: ./blade_motion_result)"
        ),
    )

    # ========================================================
    # ROI
    # ========================================================

    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=(
            "X",
            "Y",
            "WIDTH",
            "HEIGHT",
        ),
        default=None,
        help=(
            "Blade ROI: "
            "x y width height. "
            "미입력 시 첫 프레임에서 직접 선택"
        ),
    )

    # ========================================================
    # Optical Flow
    # ========================================================

    parser.add_argument(
        "--analysis-width",
        type=int,
        default=320,
        help=(
            "Optical Flow 분석용 ROI width "
            "(default: 320)"
        ),
    )

    parser.add_argument(
        "--pixel-motion-thr",
        type=float,
        default=0.5,
        help=(
            "Motion pixel magnitude threshold "
            "(default: 0.5)"
        ),
    )

    parser.add_argument(
        "--active-ratio-thr",
        type=float,
        default=0.015,
        help=(
            "ROI 내 Motion pixel 최소 비율 "
            "(default: 0.015)"
        ),
    )

    parser.add_argument(
        "--flow-thr",
        type=float,
        default=0.5,
        help=(
            "MOVE 판정 Optical Flow "
            "magnitude threshold "
            "(default: 0.5)"
        ),
    )

    # ========================================================
    # Event
    # ========================================================

    parser.add_argument(
        "--min-move-sec",
        type=float,
        default=0.15,
        help=(
            "MOVE 최소 지속시간 "
            "(default: 0.15 sec)"
        ),
    )

    parser.add_argument(
        "--stop-min-sec",
        type=float,
        default=0.05,
        help=(
            "중간 STOP 최소 지속시간 "
            "(default: 0.05 sec)"
        ),
    )

    parser.add_argument(
        "--stop-max-sec",
        type=float,
        default=0.80,
        help=(
            "중간 STOP 최대 지속시간 "
            "(default: 0.80 sec)"
        ),
    )

    parser.add_argument(
        "--max-event-sec",
        type=float,
        default=6.0,
        help=(
            "전체 Blade Event 최대시간 "
            "(default: 6.0 sec)"
        ),
    )

    # ========================================================
    # Clip
    # ========================================================

    parser.add_argument(
        "--pre-sec",
        type=float,
        default=1.0,
        help=(
            "Event 시작 전 Clip 포함시간 "
            "(default: 1.0 sec)"
        ),
    )

    parser.add_argument(
        "--post-sec",
        type=float,
        default=1.0,
        help=(
            "Event 종료 후 Clip 포함시간 "
            "(default: 1.0 sec)"
        ),
    )

    # ========================================================
    # Parse
    # ========================================================

    args = parser.parse_args()

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    roi = (
        tuple(args.roi)
        if args.roi is not None
        else None
    )

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config = MotionConfig(

        analysis_width=(
            args.analysis_width
        ),

        pixel_motion_threshold=(
            args.pixel_motion_thr
        ),

        min_active_ratio=(
            args.active_ratio_thr
        ),

        min_flow_magnitude=(
            args.flow_thr
        ),

        event_min_move_sec=(
            args.min_move_sec
        ),

        intermediate_stop_min_sec=(
            args.stop_min_sec
        ),

        intermediate_stop_max_sec=(
            args.stop_max_sec
        ),

        max_event_duration_sec=(
            args.max_event_sec
        ),

        pre_event_sec=(
            args.pre_sec
        ),

        post_event_sec=(
            args.post_sec
        ),
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    analyze_video(
        video_path=Path(
            args.video
        ),

        output_dir=Path(
            args.output
        ),

        roi=roi,

        config=config,
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
