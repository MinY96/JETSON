"""
extract_blade_depth_motion_clips.py

Fixed-camera CCTV video에서 카메라 방향으로 접근/복귀하는 Blade 동작을 검출한다.

핵심 패턴:
    HOME -> EXPANDING -> SHRINKING -> HOME

Blade가 카메라 쪽으로 접근하면 ROI 내부 optical flow가 대체로 방사형 바깥 방향,
복귀하면 방사형 안쪽 방향이 된다는 점을 이용한다.

Outputs:
    blade_motion_debug.mp4
    blade_motion_debug.csv
    blade_events.csv
    clips/blade_event_XXXX.mp4

Example:
    python extract_blade_depth_motion_clips.py ^
        --video cctv.mp4 ^
        --output blade_result ^
        --roi 1200 350 500 400

Threshold tuning example:
    python extract_blade_depth_motion_clips.py ^
        --video cctv.mp4 ^
        --output blade_result ^
        --roi 1200 350 500 400 ^
        --pixel-motion-thr 0.35 ^
        --active-ratio-thr 0.008 ^
        --radial-thr 0.10 ^
        --min-expand-sec 0.15 ^
        --min-shrink-sec 0.15
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class MotionConfig:
    analysis_width: int = 320

    # Flow를 "움직이는 픽셀"로 인정하는 최소 magnitude
    pixel_motion_threshold: float = 0.35

    # ROI 전체 중 active pixel 최소 비율
    min_active_ratio: float = 0.008

    # radial flow 판정 threshold
    radial_threshold: float = 0.10

    # EXPANDING / SHRINKING 최소 지속시간
    min_expand_sec: float = 0.15
    min_shrink_sec: float = 0.15

    # 한 Blade 왕복 이벤트 허용 범위
    min_event_sec: float = 0.8
    max_event_sec: float = 6.0

    # SHRINKING 후 정지상태가 이 시간 이상 지속되면 이벤트 종료
    home_stable_sec: float = 0.15

    # 방향 반전 사이에 허용할 짧은 애매 구간
    transition_gap_sec: float = 0.30

    # clip 여유
    pre_event_sec: float = 1.0
    post_event_sec: float = 1.0

    save_debug_video: bool = True


@dataclass
class MotionFrame:
    frame_idx: int
    time_sec: float
    motion_score: float
    active_ratio: float
    flow_x: float
    flow_y: float
    radial_score: float
    radial_abs: float
    direction: str
    event_state: str


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def resize_keep_ratio(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if width <= 0 or w <= width:
        return image
    scale = width / w
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)


def select_roi_scaled(
    frame: np.ndarray,
    max_width: int = 1280,
    max_height: int = 800,
):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)

    if scale < 1.0:
        preview = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        preview = frame.copy()

    print("\nSelect Blade ROI. ENTER/SPACE: confirm, ESC: cancel")
    roi = cv2.selectROI(
        "Select Blade ROI", preview, showCrosshair=True, fromCenter=False
    )
    cv2.destroyWindow("Select Blade ROI")

    x, y, rw, rh = roi
    if rw <= 0 or rh <= 0:
        raise RuntimeError("ROI가 선택되지 않았습니다.")

    return (
        int(round(x / scale)),
        int(round(y / scale)),
        int(round(rw / scale)),
        int(round(rh / scale)),
    )


def validate_roi(roi, frame_width: int, frame_height: int):
    x, y, w, h = roi
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"잘못된 ROI입니다: {roi}")
    if x + w > frame_width or y + h > frame_height:
        raise ValueError(
            f"ROI가 영상 범위를 벗어났습니다. "
            f"video={frame_width}x{frame_height}, roi={roi}"
        )


def make_radial_unit_vectors(height: int, width: int):
    """
    ROI 중심으로부터 각 pixel 방향의 unit vector를 미리 생성한다.
    매 프레임 재계산하지 않아 CPU 비용을 줄인다.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    rx = xx - cx
    ry = yy - cy

    norm = np.sqrt(rx * rx + ry * ry)
    norm = np.maximum(norm, 1.0)

    return rx / norm, ry / norm


def calculate_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    radial_x: np.ndarray,
    radial_y: np.ndarray,
    config: MotionConfig,
):
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

    magnitude = cv2.magnitude(dx, dy)
    active_mask = magnitude >= config.pixel_motion_threshold

    active_ratio = float(np.mean(active_mask))

    if not np.any(active_mask):
        return 0.0, active_ratio, 0.0, 0.0, 0.0, 0.0, "STABLE"

    active_mag = magnitude[active_mask]

    motion_score = float(np.median(active_mag))
    flow_x = float(np.median(dx[active_mask]))
    flow_y = float(np.median(dy[active_mask]))

    # + : ROI 중심에서 바깥쪽으로 퍼짐 = 확대/접근 후보
    # - : ROI 중심으로 모임 = 축소/복귀 후보
    radial = dx * radial_x + dy * radial_y
    active_radial = radial[active_mask]

    radial_score = float(np.median(active_radial))
    radial_abs = float(np.median(np.abs(active_radial)))

    # 충분한 영역에서 실제 motion이 존재할 때만 방향 판정
    if active_ratio < config.min_active_ratio:
        direction = "STABLE"
    elif radial_score >= config.radial_threshold:
        direction = "EXPANDING"
    elif radial_score <= -config.radial_threshold:
        direction = "SHRINKING"
    else:
        direction = "STABLE"

    return (
        motion_score,
        active_ratio,
        flow_x,
        flow_y,
        radial_score,
        radial_abs,
        direction,
    )


class BladeEventDetector:
    """
    STOP을 중간조건으로 요구하지 않는다.

    IDLE
      -> EXPANDING이 min_expand_sec 지속
      -> WAIT_SHRINK
      -> SHRINKING이 min_shrink_sec 지속
      -> SHRINKING
      -> STABLE이 home_stable_sec 지속
      -> EVENT 완료

    EXPANDING -> SHRINKING 사이의 순간적인 STABLE/noisy 구간은
    transition_gap_sec까지 허용한다.
    """

    def __init__(self, fps: float, config: MotionConfig):
        self.fps = fps
        self.config = config

        self.min_expand_frames = max(1, int(round(config.min_expand_sec * fps)))
        self.min_shrink_frames = max(1, int(round(config.min_shrink_sec * fps)))
        self.home_frames = max(1, int(round(config.home_stable_sec * fps)))
        self.gap_frames = max(1, int(round(config.transition_gap_sec * fps)))

        self.reset()

    def reset(self):
        self.state = "IDLE"

        self.expand_count = 0
        self.shrink_count = 0
        self.stable_count = 0
        self.gap_count = 0

        self.expand_candidate_start_frame = None
        self.expand_candidate_start_time = None

        self.event_start_frame = None
        self.event_start_time = None

        self.peak_frame = None
        self.peak_time = None

        self.shrink_start_frame = None
        self.shrink_start_time = None

        self.max_radial_score = None
        self.min_radial_score = None

    def _update_extremes(self, radial_score: float):
        if self.max_radial_score is None:
            self.max_radial_score = radial_score
            self.min_radial_score = radial_score
        else:
            self.max_radial_score = max(self.max_radial_score, radial_score)
            self.min_radial_score = min(self.min_radial_score, radial_score)

    def update(
        self,
        frame_idx: int,
        time_sec: float,
        direction: str,
        radial_score: float,
    ):
        completed_event = None

        if self.state == "IDLE":
            if direction == "EXPANDING":
                if self.expand_count == 0:
                    self.expand_candidate_start_frame = frame_idx
                    self.expand_candidate_start_time = time_sec

                self.expand_count += 1

                if self.expand_count >= self.min_expand_frames:
                    self.state = "EXPANDING"
                    self.event_start_frame = self.expand_candidate_start_frame
                    self.event_start_time = self.expand_candidate_start_time
                    self.max_radial_score = radial_score
                    self.min_radial_score = radial_score
                    self.gap_count = 0
            else:
                self.expand_count = 0
                self.expand_candidate_start_frame = None
                self.expand_candidate_start_time = None

        elif self.state == "EXPANDING":
            self._update_extremes(radial_score)

            if direction == "EXPANDING":
                self.gap_count = 0

            elif direction == "SHRINKING":
                self.shrink_count += 1

                if self.shrink_count == 1:
                    self.shrink_start_frame = frame_idx
                    self.shrink_start_time = time_sec

                if self.shrink_count >= self.min_shrink_frames:
                    self.state = "SHRINKING"
                    self.peak_frame = self.shrink_start_frame
                    self.peak_time = self.shrink_start_time
                    self.stable_count = 0
                    self.gap_count = 0

            else:
                # 순간 정지/애매구간 허용
                self.gap_count += 1

                if self.gap_count > self.gap_frames:
                    # 아직 SHRINKING이 시작되지 않은 채 너무 오래 멈추면
                    # 잘못 시작한 이벤트로 보고 초기화
                    self.reset()

        elif self.state == "SHRINKING":
            self._update_extremes(radial_score)

            if direction == "SHRINKING":
                self.stable_count = 0

            elif direction == "STABLE":
                self.stable_count += 1

                if self.stable_count >= self.home_frames:
                    end_frame = frame_idx - self.stable_count + 1
                    end_time = end_frame / self.fps

                    duration = end_time - self.event_start_time

                    if (
                        self.config.min_event_sec
                        <= duration
                        <= self.config.max_event_sec
                    ):
                        completed_event = {
                            "event_start_frame": self.event_start_frame,
                            "event_start_time": self.event_start_time,
                            "peak_frame": self.peak_frame,
                            "peak_time": self.peak_time,
                            "shrink_start_frame": self.shrink_start_frame,
                            "shrink_start_time": self.shrink_start_time,
                            "event_end_frame": end_frame,
                            "event_end_time": end_time,
                            "event_duration": duration,
                            "max_radial_score": self.max_radial_score,
                            "min_radial_score": self.min_radial_score,
                        }

                    self.reset()

            elif direction == "EXPANDING":
                # SHRINK 도중 짧은 noise/반전은 gap만큼 허용
                self.gap_count += 1
                if self.gap_count > self.gap_frames:
                    self.reset()

        return completed_event, self.state


def draw_debug_overlay(
    frame: np.ndarray,
    roi,
    frame_idx: int,
    time_sec: float,
    motion_score: float,
    active_ratio: float,
    flow_x: float,
    flow_y: float,
    radial_score: float,
    radial_abs: float,
    direction: str,
    event_state: str,
    config: MotionConfig,
):
    out = frame.copy()
    x, y, w, h = roi

    if direction == "EXPANDING":
        roi_color = (0, 255, 0)
    elif direction == "SHRINKING":
        roi_color = (255, 180, 0)
    else:
        roi_color = (0, 200, 255)

    cv2.rectangle(out, (x, y), (x + w, y + h), roi_color, 3)

    cv2.putText(
        out,
        f"{direction} | {event_state}",
        (x, max(32, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        roi_color,
        2,
        cv2.LINE_AA,
    )

    panel_x, panel_y = 20, 20
    panel_w, panel_h = 650, 345

    overlay = out.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.68, out, 0.32, 0, out)

    radial_pass = abs(radial_score) >= config.radial_threshold
    active_pass = active_ratio >= config.min_active_ratio

    lines = [
        ("BLADE DEPTH MOTION MONITOR", (255, 255, 255)),
        (f"Direction     : {direction}", roi_color),
        (f"Event State   : {event_state}", roi_color),
        (
            f"Radial Flow   : {radial_score:+.4f}  "
            f"[THR +/-{config.radial_threshold:.4f}] "
            f"{'PASS' if radial_pass else 'FAIL'}",
            (0, 255, 0) if radial_pass else (0, 0, 255),
        ),
        (f"Radial Abs    : {radial_abs:.4f}", (255, 255, 255)),
        (
            f"Active Ratio  : {active_ratio:.4f}  "
            f"[THR {config.min_active_ratio:.4f}] "
            f"{'PASS' if active_pass else 'FAIL'}",
            (0, 255, 0) if active_pass else (0, 0, 255),
        ),
        (
            f"Motion Score  : {motion_score:.4f}  "
            f"Pixel THR {config.pixel_motion_threshold:.4f}",
            (255, 255, 255),
        ),
        (f"Flow X / Y    : {flow_x:+.4f} / {flow_y:+.4f}", (255, 255, 255)),
        (f"Video Time    : {format_time(time_sec)}", (255, 255, 255)),
        (f"Frame         : {frame_idx}", (255, 255, 255)),
    ]

    tx = panel_x + 15
    ty = panel_y + 30

    for text, color in lines:
        cv2.putText(
            out,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        ty += 30

    # radial score bar
    bar_x = panel_x + 390
    bar_y = panel_y + panel_h - 28
    bar_w = 230
    center = bar_x + bar_w // 2

    cv2.line(out, (bar_x, bar_y), (bar_x + bar_w, bar_y), (180, 180, 180), 2)
    cv2.line(out, (center, bar_y - 10), (center, bar_y + 10), (255, 255, 255), 2)

    # display range: threshold의 최소 4배 또는 1.0
    display_range = max(1.0, config.radial_threshold * 4.0)
    normalized = float(np.clip(radial_score / display_range, -1.0, 1.0))
    score_x = int(round(center + normalized * (bar_w / 2)))

    cv2.line(out, (center, bar_y), (score_x, bar_y), roi_color, 6)

    return out


def extract_clip(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("FPS 정보를 읽을 수 없습니다.")

    start_frame = max(0, int(np.floor(start_sec * fps)))
    end_frame = max(start_frame, int(np.ceil(end_sec * fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"VideoWriter 생성 실패: {output_path}")

    idx = start_frame

    while idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        writer.write(frame)
        idx += 1

    writer.release()
    cap.release()


def save_event_csv(events, path: Path):
    fields = [
        "event_id",
        "clip_file",
        "event_start_frame",
        "event_start_sec",
        "event_start",
        "peak_frame",
        "peak_sec",
        "peak_time",
        "shrink_start_frame",
        "shrink_start_sec",
        "event_end_frame",
        "event_end_sec",
        "event_end",
        "event_duration_sec",
        "max_radial_score",
        "min_radial_score",
        "clip_start_sec",
        "clip_end_sec",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for event in events:
            writer.writerow(event)


def analyze_video(
    video_path: Path,
    output_dir: Path,
    roi=None,
    config: Optional[MotionConfig] = None,
):
    if config is None:
        config = MotionConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("FPS 정보를 읽을 수 없습니다.")

    duration = frame_count / fps if frame_count > 0 else 0.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("첫 프레임을 읽을 수 없습니다.")

    if roi is None:
        roi = select_roi_scaled(first_frame)

    roi = tuple(int(v) for v in roi)
    validate_roi(roi, frame_width, frame_height)
    x, y, w, h = roi

    print("\n========================================")
    print("VIDEO / ROI")
    print("========================================")
    print(f"Video      : {video_path}")
    print(f"Resolution : {frame_width} x {frame_height}")
    print(f"FPS        : {fps:.3f}")
    print(f"Duration   : {format_time(duration)}")
    print(f"ROI        : x={x}, y={y}, w={w}, h={h}")

    print("\n========================================")
    print("THRESHOLDS")
    print("========================================")
    print(f"pixel_motion_thr : {config.pixel_motion_threshold}")
    print(f"active_ratio_thr : {config.min_active_ratio}")
    print(f"radial_thr       : +/-{config.radial_threshold}")
    print(f"min_expand_sec   : {config.min_expand_sec}")
    print(f"min_shrink_sec   : {config.min_shrink_sec}")
    print(f"event range      : {config.min_event_sec} ~ {config.max_event_sec} sec")
    print(f"home_stable_sec  : {config.home_stable_sec}")
    print(f"transition_gap   : {config.transition_gap_sec}")

    prev_roi = first_frame[y:y+h, x:x+w]
    prev_roi = resize_keep_ratio(prev_roi, config.analysis_width)
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    ah, aw = prev_gray.shape[:2]
    radial_x, radial_y = make_radial_unit_vectors(ah, aw)

    debug_video_path = output_dir / "blade_motion_debug.mp4"
    debug_writer = None

    if config.save_debug_video:
        debug_writer = cv2.VideoWriter(
            str(debug_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not debug_writer.isOpened():
            cap.release()
            raise RuntimeError(f"Debug VideoWriter 생성 실패: {debug_video_path}")

    debug_csv_path = output_dir / "blade_motion_debug.csv"
    debug_file = debug_csv_path.open("w", newline="", encoding="utf-8-sig")
    debug_csv = csv.writer(debug_file)
    debug_csv.writerow([
        "frame_idx",
        "time_sec",
        "video_time",
        "motion_score",
        "active_ratio",
        "flow_x",
        "flow_y",
        "radial_score",
        "radial_abs",
        "direction",
        "event_state",
    ])

    detector = BladeEventDetector(fps, config)
    events = []

    frame_idx = 1
    progress_interval = max(1, int(round(fps * 60)))

    print("\nAnalyzing...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            roi_img = frame[y:y+h, x:x+w]
            roi_img = resize_keep_ratio(roi_img, config.analysis_width)

            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            (
                motion_score,
                active_ratio,
                flow_x,
                flow_y,
                radial_score,
                radial_abs,
                direction,
            ) = calculate_motion(
                prev_gray,
                gray,
                radial_x,
                radial_y,
                config,
            )

            time_sec = frame_idx / fps

            completed_event, event_state = detector.update(
                frame_idx,
                time_sec,
                direction,
                radial_score,
            )

            if completed_event is not None:
                event_id = len(events) + 1

                clip_start = max(
                    0.0,
                    completed_event["event_start_time"] - config.pre_event_sec,
                )
                clip_end = min(
                    duration,
                    completed_event["event_end_time"] + config.post_event_sec,
                ) if duration > 0 else (
                    completed_event["event_end_time"] + config.post_event_sec
                )

                completed_event.update({
                    "event_id": event_id,
                    "clip_file": f"blade_event_{event_id:04d}.mp4",
                    "event_start_sec": f'{completed_event["event_start_time"]:.3f}',
                    "event_start": format_time(completed_event["event_start_time"]),
                    "peak_sec": (
                        f'{completed_event["peak_time"]:.3f}'
                        if completed_event["peak_time"] is not None else ""
                    ),
                    "peak_time": (
                        format_time(completed_event["peak_time"])
                        if completed_event["peak_time"] is not None else ""
                    ),
                    "shrink_start_sec": (
                        f'{completed_event["shrink_start_time"]:.3f}'
                        if completed_event["shrink_start_time"] is not None else ""
                    ),
                    "event_end_sec": f'{completed_event["event_end_time"]:.3f}',
                    "event_end": format_time(completed_event["event_end_time"]),
                    "event_duration_sec": f'{completed_event["event_duration"]:.3f}',
                    "max_radial_score": f'{completed_event["max_radial_score"]:.6f}',
                    "min_radial_score": f'{completed_event["min_radial_score"]:.6f}',
                    "clip_start_sec": f"{clip_start:.3f}",
                    "clip_end_sec": f"{clip_end:.3f}",
                })

                events.append(completed_event)

                print(
                    f"[EVENT {event_id:04d}] "
                    f"{format_time(completed_event['event_start_time'])} -> "
                    f"{format_time(completed_event['event_end_time'])} "
                    f"({completed_event['event_duration']:.2f}s)"
                )

            debug_csv.writerow([
                frame_idx,
                f"{time_sec:.3f}",
                format_time(time_sec),
                f"{motion_score:.6f}",
                f"{active_ratio:.6f}",
                f"{flow_x:.6f}",
                f"{flow_y:.6f}",
                f"{radial_score:.6f}",
                f"{radial_abs:.6f}",
                direction,
                event_state,
            ])

            if debug_writer is not None:
                debug_frame = draw_debug_overlay(
                    frame,
                    roi,
                    frame_idx,
                    time_sec,
                    motion_score,
                    active_ratio,
                    flow_x,
                    flow_y,
                    radial_score,
                    radial_abs,
                    direction,
                    event_state,
                    config,
                )
                debug_writer.write(debug_frame)

            prev_gray = gray
            frame_idx += 1

            if frame_idx % progress_interval == 0:
                current = frame_idx / fps
                if duration > 0:
                    print(
                        f"Processing {format_time(current)} / "
                        f"{format_time(duration)} "
                        f"({current / duration * 100:.1f}%)"
                    )
                else:
                    print(f"Processing {format_time(current)}")

    finally:
        cap.release()
        debug_file.close()
        if debug_writer is not None:
            debug_writer.release()

    # 원본 영상에서 이벤트 클립 생성
    print(f"\nDetected events: {len(events)}")

    for event in events:
        clip_path = clips_dir / event["clip_file"]
        extract_clip(
            video_path,
            clip_path,
            float(event["clip_start_sec"]),
            float(event["clip_end_sec"]),
        )

    event_csv_path = output_dir / "blade_events.csv"
    save_event_csv(events, event_csv_path)

    print("\n========================================")
    print("COMPLETE")
    print("========================================")
    print(f"Debug video : {debug_video_path if config.save_debug_video else 'disabled'}")
    print(f"Debug CSV   : {debug_csv_path}")
    print(f"Event CSV   : {event_csv_path}")
    print(f"Clips       : {clips_dir}")
    print(f"Events      : {len(events)}")


def main():
    parser = argparse.ArgumentParser(
        description="Blade depth-motion detector (EXPANDING -> SHRINKING)"
    )

    parser.add_argument("--video", required=True, help="입력 MP4")
    parser.add_argument(
        "--output",
        default="./blade_motion_result",
        help="결과 저장 폴더",
    )

    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        default=None,
        help="Blade ROI. 미입력 시 첫 프레임에서 직접 선택",
    )

    parser.add_argument("--analysis-width", type=int, default=320)
    parser.add_argument("--pixel-motion-thr", type=float, default=0.35)
    parser.add_argument("--active-ratio-thr", type=float, default=0.008)
    parser.add_argument("--radial-thr", type=float, default=0.10)

    parser.add_argument("--min-expand-sec", type=float, default=0.15)
    parser.add_argument("--min-shrink-sec", type=float, default=0.15)

    parser.add_argument("--min-event-sec", type=float, default=0.8)
    parser.add_argument("--max-event-sec", type=float, default=6.0)

    parser.add_argument("--home-stable-sec", type=float, default=0.15)
    parser.add_argument("--transition-gap-sec", type=float, default=0.30)

    parser.add_argument("--pre-sec", type=float, default=1.0)
    parser.add_argument("--post-sec", type=float, default=1.0)

    parser.add_argument(
        "--no-debug-video",
        action="store_true",
        help="Debug MP4 생성을 끈다",
    )

    args = parser.parse_args()

    if args.analysis_width <= 0:
        parser.error("--analysis-width는 0보다 커야 합니다.")
    if args.pixel_motion_thr < 0:
        parser.error("--pixel-motion-thr는 0 이상이어야 합니다.")
    if args.active_ratio_thr < 0 or args.active_ratio_thr > 1:
        parser.error("--active-ratio-thr는 0~1 범위여야 합니다.")
    if args.radial_thr < 0:
        parser.error("--radial-thr는 0 이상이어야 합니다.")
    if args.min_event_sec > args.max_event_sec:
        parser.error("--min-event-sec는 --max-event-sec보다 클 수 없습니다.")

    roi = tuple(args.roi) if args.roi is not None else None

    config = MotionConfig(
        analysis_width=args.analysis_width,
        pixel_motion_threshold=args.pixel_motion_thr,
        min_active_ratio=args.active_ratio_thr,
        radial_threshold=args.radial_thr,
        min_expand_sec=args.min_expand_sec,
        min_shrink_sec=args.min_shrink_sec,
        min_event_sec=args.min_event_sec,
        max_event_sec=args.max_event_sec,
        home_stable_sec=args.home_stable_sec,
        transition_gap_sec=args.transition_gap_sec,
        pre_event_sec=args.pre_sec,
        post_event_sec=args.post_sec,
        save_debug_video=not args.no_debug_video,
    )

    analyze_video(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        roi=roi,
        config=config,
    )


if __name__ == "__main__":
    main()
