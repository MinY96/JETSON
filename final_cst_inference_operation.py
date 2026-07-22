#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_cst_inference_operation.py

RMV CST(Cassette) 최종 운영 추론 스크립트
- 24시간 상시 운영 / Warning Lock / Daily CSV Log / 재학습용 Auto Frame 수집

===============================================================================
전체 파이프라인 개요
===============================================================================
카메라(또는 mp4) 프레임을 입력받아 좌(LEFT)/우(RIGHT) CST 영역(ROI)을 잘라내고,
아래 순서로 하나의 "Cycle"을 처리한다.

    1) WAIT_ANY_CST : 좌/우 CST가 모두 Empty인 대기 상태.
                       좌/우 중 하나라도 NonEmpty가 되면 Cycle이 시작된다.
    2) STABILIZE     : 좌/우가 동시에 NonEmpty인 상태가 N프레임 이상 유지되는지
                       확인하는 안정화 구간. (진동/오검출 방지용 디바운스)
                       단, RIGHT CST가 먼저 NonEmpty가 되고 그 안에서 Wafer가
                       검출되면 안정화를 기다리지 않고 즉시 WARNING으로 Lock한다.
    3) EVALUATE      : 안정화가 끝나면 1회 색상(Pink/Non-pink) 및 Wafer 존재
                       여부를 판정하여 최종 상태(NORMAL/ABNORMAL/WARNING/
                       INCOMPLETE/UNKNOWN)를 확정한다.
    4) LOCKED        : 판정 결과를 고정(Lock)한 상태. 이 상태에서는 재판정을
                       하지 않고, 좌/우 CST가 다시 Empty로 돌아오면(= 물건이
                       빠지면) Cycle을 종료하고 초기 상태로 Reset한다.

부가 기능
    - PATLITE 경광등/부저 알람 연동 (WARNING=RED, ABNORMAL=ORANGE)
    - 이상(WARNING/ABNORMAL) 발생 시 전/후 구간을 포함한 mp4 이벤트 클립 저장
    - 일 단위(daily) CSV 로그 자동 생성/보관기한 관리
    - 모델 재학습을 위한 프레임 자동 수집(랜덤 샘플링 + 이상 케이스 큐잉),
      디스크 용량 쿼터 관리

===============================================================================
코드 구성 (섹션 순서)
===============================================================================
    1. 상수/기본 경로 설정
    2. 공용 유틸리티 함수 (argparse 헬퍼, 파일/디렉터리 유틸)
    3. 일 단위 CSV 로거 (DailyCsvLogger)
    4. 이미지 전처리 유틸 (center_crop, crop_roi)
    5. 이산값(bool) 디바운스 홀드 상태 (BoolHoldState / update_hold_bool)
    6. PATLITE 경광등 컨트롤러
    7. Empty/NonEmpty 판정 (참조 이미지 기반 차분 비교)
    8. Pink/Non-pink 판정 (색상 규칙 기반)
    9. Wafer 존재 판정 (TensorRT 분류 모델)
   10. 최종 상태 판정 로직 (derive_final_status)
   11. Cycle 상태 머신 (CycleStateMachine)
   12. 재학습용 프레임 자동 수집기 (AutoFrameCollector)
   13. 이상 이벤트 클립 레코더 (EventClipRecorder)
   14. 화면 오버레이 렌더링
   15. 영상 소스 오픈 / CLI 인자 정의
   16. CSTInferencePipeline : 위 구성요소를 조립해 한 프레임씩 처리하는 실행기
   17. main() : 엔트리 포인트
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd  # noqa: F401  (외부 후처리/분석 스크립트와의 호환을 위해 유지)
from PIL import Image

# TensorRT / torch 계열은 GPU 추론 환경에서만 필요하다.
# 개발 PC 등 해당 패키지가 없는 환경에서도 스크립트 자체는 import 가능하도록
# 실패 시 None 처리 후, 실제 사용 시점(TensorRTWaferClassifier)에서 에러를 낸다.
try:
    import torch
    import tensorrt as trt
    from torchvision import transforms
except Exception:
    torch = None
    trt = None
    transforms = None


# ==============================================================================
# 1. 상수 / 기본 경로 설정
# ==============================================================================

# 이 파일 기준 프로젝트 루트 (예: project_root/scripts/final_cst_inference_operation.py
# 형태로 배치되어 있다고 가정. 실제 배치 구조가 다르면 이 값만 수정하면 된다.
BASE_DIR = Path(__file__).resolve().parent.parent

# 좌/우 CST ROI 기본값: (x, y, w, h) - 카메라 셋업이 바뀌면 CLI 인자로 덮어쓴다.
DEFAULT_LEFT_ROI = (3, 85, 300, 410)
DEFAULT_RIGHT_ROI = (1210, 240, 540, 600)

DEFAULT_REFERENCE_EMPTY_DIR = BASE_DIR / "reference_empty"
DEFAULT_PINK_RULE_FILE = BASE_DIR / "pink_rule_search" / "pink_rule_thresholds.json"
DEFAULT_ENGINE_PATH = BASE_DIR / "models" / "wafer_mobilenetv3_small_fp16.engine"
DEFAULT_META_PATH = BASE_DIR / "models" / "wafer_class_meta.json"

DEFAULT_LOG_DIR = BASE_DIR / "logs"
DEFAULT_HISTORY_DIR = BASE_DIR / "history"
DEFAULT_EVENT_DIR = BASE_DIR / "event"
DEFAULT_AUTO_FRAME_DIR = BASE_DIR / "auto_frames"

# Empty 판정 임계값(작을수록 엄격). 좌/우 카메라 조명 조건이 달라 별도로 관리.
DEFAULT_LEFT_EMPTY_THRESHOLD = 19.0
DEFAULT_RIGHT_EMPTY_THRESHOLD = 30.0  # 과거값 18.0에서 현장 튜닝을 거쳐 상향 조정됨

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 화면 오버레이에 사용하는 BGR 색상표
DISPLAY_COLORS = {
    "pink": (203, 72, 255),
    "non_pink": (180, 180, 180),
    "empty": (0, 255, 255),
    "unknown": (255, 255, 255),
    "normal": (0, 255, 0),
    "incomplete": (0, 255, 255),
    "abnormal": (0, 165, 255),
    "warning": (0, 0, 255),
    "idle": (180, 180, 180),
}


# ==============================================================================
# 2. 공용 유틸리티 함수
# ==============================================================================

def str2bool(v: Union[str, bool]) -> bool:
    """argparse용 문자열 -> bool 변환기.

    "true/false", "1/0", "yes/no", "y/n", "t/f" (대소문자 무관)를 허용한다.
    argparse의 ``type=str2bool`` 로 사용한다.
    """
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("true/false 값을 입력하세요.")


def ensure_dir(path: Union[str, Path]) -> Path:
    """디렉터리가 없으면 생성하고 Path 객체를 반환한다."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_roi(values: List[int]) -> Tuple[int, int, int, int]:
    """``--left_roi x y w h`` 형태의 4개 정수 리스트를 (x, y, w, h) 튜플로 변환한다."""
    if len(values) != 4:
        raise ValueError("ROI는 x y w h 4개 정수여야 합니다.")
    return tuple(map(int, values))


def safe_json_load(path: Union[str, Path]) -> dict:
    """UTF-8 JSON 파일을 읽어 dict로 반환한다."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_images(folder: Union[str, Path]) -> List[Path]:
    """폴더 내 이미지 파일(png/jpg/jpeg/bmp/webp)을 이름순으로 정렬해 반환한다.

    폴더가 존재하지 않으면 빈 리스트를 반환한다.
    """
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def cleanup_old_files(folder: Union[str, Path], keep_days: int = 7) -> None:
    """``folder`` 하위(재귀)에서 수정시각이 ``keep_days`` 일보다 오래된 파일을 삭제한다.

    로그/이벤트/자동수집 프레임 등이 무한히 쌓이는 것을 방지하기 위한
    보관기한(retention) 정리 함수다. 개별 파일 삭제 실패는 경고만 출력하고
    계속 진행한다.
    """
    folder = Path(folder)
    if not folder.exists():
        return
    cutoff = datetime.now() - timedelta(days=keep_days)
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        try:
            if datetime.fromtimestamp(p.stat().st_mtime) < cutoff:
                p.unlink()
                print(f"[CLEANUP] deleted old file: {p}")
        except Exception as e:
            print(f"[WARN] cleanup failed: {p}, error={e}")


# ==============================================================================
# 3. 일 단위(daily) CSV 로거
# ==============================================================================

class DailyCsvLogger:
    """일 단위 CSV 로그 파일을 실시간 append 방식으로 기록하는 로거.

    특징
        - 파일명 규칙: ``{prefix}_YYYYMMDD.csv`` (날짜가 바뀌면 자동으로 새 파일 생성)
        - ``keep_days`` 가 지난 로그 파일은 자동 삭제
        - 프로그램 종료 시 한꺼번에 저장하지 않고, ``write()`` 호출마다 즉시 flush
          하므로 프로세스가 비정상 종료되어도 그 시점까지의 로그는 보존된다.

    사용 예시
        >>> logger = DailyCsvLogger("logs", prefix="frame_results", keep_days=7)
        >>> logger.write({"time": "2026-07-22 10:00:00", "status": "NORMAL"})
    """

    def __init__(self, log_dir: Union[str, Path], prefix: str, keep_days: int = 7):
        self.log_dir = ensure_dir(log_dir)
        self.prefix = prefix
        self.keep_days = keep_days
        self.current_date: Optional[str] = None
        self.current_path: Optional[Path] = None
        self.fieldnames: Optional[List[str]] = None
        cleanup_old_files(self.log_dir, keep_days=self.keep_days)

    def _today_path(self) -> Path:
        """오늘 날짜 기준 CSV 파일 경로를 반환한다."""
        today = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"{self.prefix}_{today}.csv"

    def write(self, row: dict) -> None:
        """한 줄(dict)을 오늘자 CSV에 append 한다.

        - 날짜가 바뀌면 새 파일을 시작하고, 그 시점에 오래된 로그를 정리한다.
        - 컬럼(fieldnames)은 최초 write 시점의 키로 고정되며, 이후 새로운 키가
          들어오면 컬럼 목록 뒤에 확장 추가한다. 단, 이미 파일 헤더가 쓰여진
          뒤 컬럼이 확장되면 기존 파일 헤더와 어긋날 수 있으므로, 운영 시에는
          가급적 매 호출마다 동일한 키 집합을 사용하는 것을 권장한다.
        """
        path = self._today_path()
        today = datetime.now().strftime("%Y%m%d")
        if self.current_date != today:
            self.current_date = today
            cleanup_old_files(self.log_dir, keep_days=self.keep_days)
        file_exists = path.exists() and path.stat().st_size > 0

        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        else:
            for key in row.keys():
                if key not in self.fieldnames:
                    self.fieldnames.append(key)

        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})
            f.flush()


# ==============================================================================
# 4. 이미지 전처리 유틸
# ==============================================================================

def center_crop(img: np.ndarray, ratio: float) -> np.ndarray:
    """이미지 중앙을 ``ratio`` 비율(가로/세로 동일)만큼 잘라낸다.

    ROI 가장자리(케이스 프레임, 그림자 등)의 잡음을 배제하고 실제 관심
    영역만 비교하기 위해 사용한다. ``ratio >= 0.999`` 이면 원본을 그대로
    반환한다.
    """
    if ratio >= 0.999:
        return img
    h, w = img.shape[:2]
    ch = max(1, int(round(h * ratio)))
    cw = max(1, int(round(w * ratio)))
    y1 = (h - ch) // 2
    x1 = (w - cw) // 2
    return img[y1:y1 + ch, x1:x1 + cw].copy()


def crop_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """원본 프레임에서 ROI(x, y, w, h) 영역을 잘라낸다.

    ROI가 프레임 경계를 벗어나면 프레임 경계로 clip 하며, clip 후에도
    유효한 영역이 없으면 ``ValueError`` 를 발생시킨다.
    """
    x, y, w, h = roi
    H, W = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid ROI: {roi}, frame_size=({W},{H})")
    return frame[y1:y2, x1:x2].copy()


# ==============================================================================
# 5. 이산값(bool) 디바운스 홀드 상태
# ==============================================================================

class BoolHoldState:
    """bool 신호(예: Empty 여부)를 디바운스(hold)하기 위한 상태 컨테이너.

    프레임 단위 원시 예측값(pred)이 흔들리더라도, 같은 값이 ``hold_frames``
    프레임 연속으로 관측되어야만 안정값(``stable_label``)을 바꾸는 방식으로
    노이즈에 의한 순간적 오검출을 억제한다.

    Attributes:
        stable_label: 현재 확정(안정)된 값. 외부에서 실제로 사용하는 값.
        candidate_label: 다음 안정값 후보로 누적 중인 값.
        candidate_count: candidate_label이 연속으로 관측된 프레임 수.
    """

    def __init__(self, stable_label: bool = True, candidate_label: bool = True, candidate_count: int = 0):
        self.stable_label = stable_label
        self.candidate_label = candidate_label
        self.candidate_count = candidate_count


def update_hold_bool(state: BoolHoldState, pred: bool, hold_frames: int) -> BoolHoldState:
    """새 원시 예측값(``pred``)으로 ``state``를 갱신한다.

    Args:
        state: 이전 프레임까지의 홀드 상태.
        pred: 이번 프레임의 원시(raw) 예측값.
        hold_frames: 값이 바뀌기 위해 필요한 연속 프레임 수.
                     1 이하이면 디바운스 없이 매 프레임 즉시 반영한다.

    Returns:
        갱신된 ``state`` (in-place 수정 후 동일 객체 반환).
    """
    if hold_frames <= 1:
        state.stable_label = pred
        state.candidate_label = pred
        state.candidate_count = 0
        return state

    # 이번 예측이 이미 안정값과 같다면 흔들림이 없는 것이므로 후보 카운트를 리셋한다.
    if pred == state.stable_label:
        state.candidate_label = pred
        state.candidate_count = 0
        return state

    # 안정값과 다른 예측이 나온 경우: 새로운 후보로 누적을 시작하거나 이어간다.
    if pred == state.candidate_label:
        state.candidate_count += 1
    else:
        state.candidate_label = pred
        state.candidate_count = 1

    # 후보가 hold_frames 이상 연속되면 비로소 안정값을 교체한다.
    if state.candidate_count >= hold_frames:
        state.stable_label = state.candidate_label
        state.candidate_count = 0
    return state


# ==============================================================================
# 6. PATLITE 경광등 컨트롤러
# ==============================================================================

class PatliteLR6USBController:
    """PATLITE LR6-USB 적층 경광등(램프+부저)을 제어하는 컨트롤러.

    ``enabled=False`` 이거나 장치 연결/초기화에 실패하면 자동으로 비활성화
    모드로 동작하며, 이 경우 실제 하드웨어 제어 대신 콘솔에 알람 상태만
    출력한다(현장 장비 없이도 스크립트 실행/테스트가 가능하도록 함).
    """

    VENDOR_ID = 0x191A
    DEVICE_ID = 0x8003
    ENDPOINT_ADDRESS = 1
    SEND_TIMEOUT = 1000
    COMMAND_VERSION = 0x00
    COMMAND_ID = 0x00
    LED_OFF = 0x0
    LED_PATTERN1 = 0x2
    BUZZER_OFF = 0x0
    BUZZER_PATTERN1 = 0x2
    BUZZER_PATTERN2 = 0x3
    BUZZER_PITCH_OFF = 0x0
    BUZZER_PITCH1 = 0x1
    BUZZER_PITCH2 = 0x2

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.dev = None
        self.current_alarm = None
        if not self.enabled:
            print("[INFO] PATLITE disabled")
            return
        try:
            import usb.core
            self.dev = usb.core.find(idVendor=self.VENDOR_ID, idProduct=self.DEVICE_ID)
            if self.dev is None:
                raise RuntimeError("PATLITE LR6-USB device not found")
            try:
                if self.dev.is_kernel_driver_active(0):
                    self.dev.detach_kernel_driver(0)
            except Exception:
                pass
            self.dev.set_configuration()
            self.reset()
            print("[INFO] PATLITE LR6-USB connected")
        except Exception as e:
            self.enabled = False
            self.dev = None
            print(f"[WARN] PATLITE init failed. Run without lamp. error={e}")

    def _send(
        self,
        red: int,
        yellow: int,
        green: int = 0,
        blue: int = 0,
        white: int = 0,
        buzzer: int = 0,
        buzzer_pitch: int = 0,
    ) -> bool:
        """PATLITE 프로토콜 규격에 맞춰 8바이트 커맨드를 USB로 전송한다."""
        if not self.enabled or self.dev is None:
            return False
        led_ry = (red << 4) | yellow
        led_gb = (green << 4) | blue
        led_w = white << 4
        data = bytes([
            self.COMMAND_VERSION, self.COMMAND_ID,
            buzzer, buzzer_pitch,
            led_ry, led_gb, led_w, 0x00,
        ])
        try:
            return self.dev.write(self.ENDPOINT_ADDRESS, data, self.SEND_TIMEOUT) == len(data)
        except Exception as e:
            print(f"[WARN] PATLITE send failed: {e}")
            return False

    def reset(self) -> None:
        """램프/부저를 모두 끄고 알람 상태를 PASS(정상)로 초기화한다."""
        self._send(red=self.LED_OFF, yellow=self.LED_OFF, buzzer=self.BUZZER_OFF, buzzer_pitch=self.BUZZER_PITCH_OFF)
        self.current_alarm = "PASS"

    def set_alarm(self, alarm: str) -> None:
        """알람 상태를 설정한다. 동일 상태 재호출 시 불필요한 USB 전송을 생략한다.

        Args:
            alarm: "RED"(적색+고음 부저), "ORANGE"(황색+저음 부저), 그 외는 정상(PASS)으로 처리.
        """
        alarm = str(alarm).upper()
        if alarm == self.current_alarm:
            return
        self.current_alarm = alarm
        if not self.enabled:
            print(f"[ALARM] {alarm}")
            return
        if alarm == "RED":
            self._send(red=self.LED_PATTERN1, yellow=self.LED_OFF, buzzer=self.BUZZER_PATTERN2, buzzer_pitch=self.BUZZER_PITCH2)
        elif alarm == "ORANGE":
            self._send(red=self.LED_OFF, yellow=self.LED_PATTERN1, buzzer=self.BUZZER_PATTERN1, buzzer_pitch=self.BUZZER_PITCH1)
        else:
            self.reset()

    def cleanup(self) -> None:
        """프로그램 종료 시 램프/부저를 끈다."""
        self.reset()


def alarm_from_status(status: str) -> str:
    """판정 상태(status) 문자열을 PATLITE 알람 등급으로 매핑한다.

    WARNING -> RED(긴급), ABNORMAL -> ORANGE(주의), 그 외 -> PASS(정상).
    """
    if status == "WARNING":
        return "RED"
    if status == "ABNORMAL":
        return "ORANGE"
    return "PASS"


# ==============================================================================
# 7. Empty / NonEmpty 판정 (참조 이미지 기반 차분 비교)
# ==============================================================================

def preprocess_for_empty(img_bgr: np.ndarray, center_crop_ratio: float, blur_ksize: int) -> np.ndarray:
    """Empty 판정을 위한 전처리: 그레이스케일 변환 -> 중앙 크롭 -> 가우시안 블러.

    조명/미세 텍스처 노이즈를 줄여 참조 이미지와의 차분 비교를 안정화한다.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = center_crop(gray, center_crop_ratio)
    if blur_ksize >= 3:
        if blur_ksize % 2 == 0:
            blur_ksize += 1  # GaussianBlur 커널 크기는 홀수여야 함
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray.astype(np.float32)


def load_empty_refs(
    reference_dir: Union[str, Path],
    center_crop_ratio: float,
    blur_ksize: int,
) -> Dict[str, List[np.ndarray]]:
    """``reference_dir/left``, ``reference_dir/right`` 폴더의 Empty 상태 참조
    이미지들을 읽어 동일한 전처리를 적용한 뒤 리스트로 반환한다.

    Returns:
        {"left": [전처리된 참조 이미지, ...], "right": [...]}
    """
    reference_dir = Path(reference_dir)
    refs: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
    for side in ["left", "right"]:
        for p in list_images(reference_dir / side):
            img = cv2.imread(str(p))
            if img is not None:
                refs[side].append(preprocess_for_empty(img, center_crop_ratio, blur_ksize))
    return refs


def compare_empty(curr_gray: np.ndarray, ref_gray: np.ndarray) -> float:
    """현재 프레임과 참조(Empty) 이미지의 밝기 차이를 점수화한다(낮을수록 유사=Empty).

    절차:
        1) 크기가 다르면 참조 이미지를 현재 이미지 크기로 리사이즈
        2) 두 이미지의 평균 밝기를 맞춰 조명 변화의 영향을 보정(gain 정규화)
        3) 절대 차분을 계산하되, 3 미만의 미세한 차이는 0으로 눌러 잡음 제거
        4) 차분의 평균값을 최종 점수로 반환
    """
    curr = curr_gray.astype(np.float32)
    ref = ref_gray.astype(np.float32)
    if curr.shape != ref.shape:
        ref = cv2.resize(ref, (curr.shape[1], curr.shape[0]), interpolation=cv2.INTER_LINEAR)

    curr_mean = float(curr.mean())
    ref_mean = float(ref.mean())
    if curr_mean > 1e-6:
        curr = curr * (ref_mean / curr_mean)
    curr = np.clip(curr, 0, 255)

    diff = np.abs(curr - ref)
    diff[diff < 3] = 0  # 미세 잡음 제거용 데드존
    return float(diff.mean())


def predict_empty(
    roi_bgr: np.ndarray,
    refs: List[np.ndarray],
    threshold: float,
    center_crop_ratio: float,
    blur_ksize: int,
) -> Tuple[bool, float]:
    """ROI가 Empty인지 판정한다.

    등록된 모든 참조 이미지와 비교해 가장 낮은(가장 유사한) 차분 점수를
    최종 점수로 삼고, ``threshold`` 이하이면 Empty로 판정한다.

    Returns:
        (is_empty, score) - score는 로그/디버깅용으로 함께 반환된다.
    """
    if not refs:
        return False, float("inf")
    curr = preprocess_for_empty(roi_bgr, center_crop_ratio, blur_ksize)
    score = float(np.min([compare_empty(curr, ref) for ref in refs]))
    return score <= threshold, score


# ==============================================================================
# 8. Pink / Non-pink 판정 (색상 규칙 기반)
# ==============================================================================

def normalize_rule_config(data: dict) -> Dict[str, dict]:
    """Pink 판정 규칙 JSON을 검증하고 {"left": rule, "right": rule} 형태로 반환한다."""
    if "left" in data and "right" in data:
        return {"left": data["left"], "right": data["right"]}
    raise ValueError("pink rule json에는 left/right rule이 필요합니다.")


def extract_pink_features(img_bgr: np.ndarray, center_crop_ratio: float, blur_ksize: int) -> Dict[str, float]:
    """Pink 판정에 사용할 색상 특징(R-G, R-B, B-G 채널 차분의 중앙값/평균)을 추출한다.

    RGB 채널 간 차분을 쓰는 이유: 전체 밝기(조명) 변화에 비교적 강건하면서도
    핑크색 특유의 R 채널 우세 경향을 잘 드러내기 때문이다.
    """
    img = center_crop(img_bgr, center_crop_ratio)
    if blur_ksize >= 3:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    b, g, r = cv2.split(img)
    r_f = r.astype(np.float32)
    g_f = g.astype(np.float32)
    b_f = b.astype(np.float32)
    rg = r_f - g_f
    rb = r_f - b_f
    bg = b_f - g_f

    return {
        "rg_diff_median": float(np.median(rg)),
        "rb_diff_median": float(np.median(rb)),
        "bg_diff_median": float(np.median(bg)),
        "rg_diff_mean": float(rg.mean()),
        "rb_diff_mean": float(rb.mean()),
        "bg_diff_mean": float(bg.mean()),
    }


def eval_condition(value: float, op: str, threshold: float) -> bool:
    """부등호 문자열(op)에 따라 ``value {op} threshold`` 조건을 평가한다."""
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    raise ValueError(f"Unsupported op: {op}")


def apply_pink_rule(features: Dict[str, float], rule: dict) -> Tuple[str, dict]:
    """색상 특징(features)에 Pink 판정 규칙(rule)을 적용해 "pink"/"non_pink"를 결정한다.

    rule_type:
        - "single": feature1 조건 하나만으로 판정
        - "and2"  : feature1, feature2 두 조건을 모두 만족해야 pink로 판정

    Returns:
        (label, debug) - debug 딕셔너리는 각 조건의 실제 값/결과를 담아
        추후 오탐 분석 시 원인 추적에 사용한다.
    """
    f1 = rule["feature1"]
    op1 = rule["op1"]
    th1 = float(rule["threshold1"])
    v1 = float(features[f1])
    cond1 = eval_condition(v1, op1, th1)
    debug = {
        "rule_type": rule["rule_type"],
        "feature1": f1, "value1": v1, "op1": op1, "threshold1": th1, "cond1": cond1,
    }

    if rule["rule_type"] == "single":
        is_pink = cond1
    elif rule["rule_type"] == "and2":
        f2 = rule["feature2"]
        op2 = rule["op2"]
        th2 = float(rule["threshold2"])
        v2 = float(features[f2])
        cond2 = eval_condition(v2, op2, th2)
        debug.update({"feature2": f2, "value2": v2, "op2": op2, "threshold2": th2, "cond2": cond2})
        is_pink = cond1 and cond2
    else:
        raise ValueError(f"Unsupported rule_type: {rule['rule_type']}")

    return ("pink" if is_pink else "non_pink"), debug


def predict_pink(
    roi_bgr: np.ndarray,
    side: str,
    rules: Dict[str, dict],
    center_crop_ratio: float,
    blur_ksize: int,
) -> Tuple[str, Dict[str, float], dict]:
    """ROI에서 색상 특징을 추출하고 side("left"/"right")에 해당하는 규칙을 적용해
    Pink 여부를 판정한다.

    Returns:
        (label, features, debug)
    """
    features = extract_pink_features(roi_bgr, center_crop_ratio, blur_ksize)
    label, debug = apply_pink_rule(features, rules[side])
    return label, features, debug


# ==============================================================================
# 9. Wafer 존재 판정 (TensorRT 분류 모델)
# ==============================================================================

class TensorRTWaferClassifier:
    """TensorRT 엔진으로 RIGHT CST 내부의 Wafer 존재 여부를 분류하는 모듈.

    ``models/wafer_class_meta.json`` 에 정의된 클래스 매핑/이미지 크기/
    양성(positive) 라벨명을 사용하며, 입력 이미지 전처리(Resize/ToTensor/
    Normalize)는 학습 시 사용한 것과 동일해야 한다.

    주의: 이 클래스는 torch/tensorrt/torchvision이 설치된, CUDA GPU가 있는
    환경에서만 동작한다. 해당 패키지가 없으면 생성 시점에 RuntimeError가 발생한다.
    """

    def __init__(self, engine_path: Union[str, Path], meta_path: Union[str, Path]):
        if torch is None or trt is None or transforms is None:
            raise RuntimeError("torch/tensorrt/torchvision import failed")

        meta = safe_json_load(meta_path)
        self.class_to_idx = meta["class_to_idx"]
        self.idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
        self.img_size = int(meta.get("img_size", 224))
        self.positive_label = meta.get("positive_label", "wafer_exist")

        # TensorRT 엔진 로드 및 실행 컨텍스트 생성
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"TensorRT engine load failed: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # 매 프레임 재할당을 피하기 위해 GPU 입출력 버퍼를 미리 확보해둔다.
        self.input_tensor = torch.empty((1, 3, self.img_size, self.img_size), dtype=torch.float32, device="cuda")
        self.output_tensor = torch.empty((1, len(self.class_to_idx)), dtype=torch.float32, device="cuda")

        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, roi_bgr: np.ndarray) -> "torch.Tensor":
        """BGR ndarray -> RGB PIL 변환 후 학습 시와 동일한 transform을 적용한다."""
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb).convert("RGB")
        return self.tf(pil_img).unsqueeze(0)

    def predict(self, roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """RIGHT CST ROI 이미지에 대해 클래스 예측을 수행한다.

        Returns:
            (예측 라벨, 예측 confidence, 전체 클래스별 확률 딕셔너리)
        """
        x_cpu = self.preprocess(roi_bgr)
        self.input_tensor.copy_(x_cpu, non_blocking=True)
        self.context.set_tensor_address(self.input_name, int(self.input_tensor.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(self.output_tensor.data_ptr()))

        ok = self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        torch.cuda.current_stream().synchronize()

        probs = torch.softmax(self.output_tensor[0], dim=0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = self.idx_to_class[pred_idx]
        conf = float(probs[pred_idx])
        prob_dict = {self.idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
        return pred_label, conf, prob_dict


class DummyWaferClassifier:
    """Wafer 분류 모델을 비활성화(``--disable_wafer_model true``)했을 때 사용하는 더미 구현.

    GPU/TensorRT 엔진 없이 로직/카메라 파이프라인만 테스트하고 싶을 때 유용하며,
    항상 "no_wafer"(확신도 1.0)를 반환한다.
    """

    def __init__(self):
        self.positive_label = "wafer_exist"

    def predict(self, roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        return "no_wafer", 1.0, {"no_wafer": 1.0, "wafer_exist": 0.0}


# ==============================================================================
# 10. 최종 상태 판정 로직
# ==============================================================================

def derive_final_status(
    left_empty: bool,
    right_empty: bool,
    left_pink_label: str,
    right_wafer_label: str,
    right_pink_label: str,
    wafer_positive_label: str,
) -> Tuple[str, str]:
    """EVALUATE 단계에서 수집한 판정값들을 조합해 최종 Cycle 상태를 결정한다.

    우선순위(위에서부터 순서대로 검사):
        1) RIGHT CST가 NonEmpty이면서 Wafer가 검출됨      -> WARNING
        2) 좌/우 중 하나라도 Empty                          -> INCOMPLETE
        3) 좌/우 Pink 판정이 유효하지 않음(unknown 등)      -> UNKNOWN
        4) 좌/우 Pink 판정 결과가 서로 일치(둘 다 pink거나 둘 다 non_pink) -> NORMAL
        5) 그 외(좌/우 색상 불일치)                          -> ABNORMAL

    Returns:
        (status, message)
    """
    if (not right_empty) and (right_wafer_label == wafer_positive_label):
        return "WARNING", "Wafer detected inside RIGHT CST at cycle start"
    if left_empty or right_empty:
        return "INCOMPLETE", "At least one CST is EMPTY at cycle start"
    if left_pink_label not in ("pink", "non_pink"):
        return "UNKNOWN", "LEFT pink decision unknown"
    if right_pink_label not in ("pink", "non_pink"):
        return "UNKNOWN", "RIGHT pink decision unknown"
    if (left_pink_label == "pink") == (right_pink_label == "pink"):
        return "NORMAL", "Left/Right CST colors are matched"
    return "ABNORMAL", "Left/Right CST colors are mismatched"


# ==============================================================================
# 11. Cycle 상태 머신
# ==============================================================================

class CycleStateMachine:
    """좌/우 CST의 Empty/NonEmpty 흐름에 따라 하나의 검사 Cycle을 관리하는 상태 머신.

    상태 전이 개요
        WAIT_ANY_CST --(하나라도 NonEmpty)--> STABILIZE
        STABILIZE     --(둘 다 NonEmpty가 stabilize_frames 유지, 혹은 timeout)--> EVALUATE
        EVALUATE      --(외부에서 lock() 호출로 판정 확정)--> LOCKED
        LOCKED        --(WARNING이면 RIGHT Empty / 그 외엔 좌우 중 하나라도 Empty가
                         reset_empty_frames 유지)--> WAIT_ANY_CST (reset)

    EVALUATE 자체의 판정 로직(색상/Wafer 추론)은 이 클래스가 아니라 외부
    (파이프라인의 EVALUATE 처리 블록)에서 수행하고, 그 결과를 ``lock()`` 으로
    이 상태머신에 반영한다. 즉 이 클래스는 "언제 판정할지/언제 리셋할지"의
    타이밍(디바운스 포함)만 책임진다.
    """

    WAIT = "WAIT_ANY_CST"
    STABILIZE = "STABILIZE"
    EVALUATE = "EVALUATE"
    LOCKED = "LOCKED"

    def __init__(self, stabilize_frames: int = 10, stabilize_timeout_frames: int = 60, reset_empty_frames: int = 5):
        self.state = self.WAIT
        self.stabilize_frames = max(1, stabilize_frames)
        self.stabilize_timeout_frames = max(self.stabilize_frames, stabilize_timeout_frames)
        self.reset_empty_frames = max(1, reset_empty_frames)

        self.both_non_empty_count = 0
        self.stabilize_elapsed = 0
        self.reset_empty_count = 0

        self.locked_status = "IDLE"
        self.locked_message = "Waiting for CST"
        self.locked_info: dict = {}

        # 이벤트 클립/재학습 프레임을 Cycle당 1회만 저장하기 위한 플래그
        self.event_triggered = False
        self.review_saved = False

    def on_frame(self, left_empty: bool, right_empty: bool) -> str:
        """매 프레임 좌/우 Empty 여부를 입력받아 상태를 갱신하고 현재 상태를 반환한다."""
        any_non_empty = (not left_empty) or (not right_empty)
        both_non_empty = (not left_empty) and (not right_empty)
        any_empty = left_empty or right_empty

        if self.state == self.WAIT:
            self.locked_status = "IDLE"
            self.locked_message = "Waiting for Left/Right CST"
            self.locked_info = {}
            self.event_triggered = False
            self.review_saved = False
            if any_non_empty:
                self.state = self.STABILIZE
                self.both_non_empty_count = 1 if both_non_empty else 0
                self.stabilize_elapsed = 1
                self.reset_empty_count = 0
                print("[CYCLE] CST detected -> STABILIZE")

        elif self.state == self.STABILIZE:
            self.stabilize_elapsed += 1
            if both_non_empty:
                self.both_non_empty_count += 1
            else:
                self.both_non_empty_count = 0

            if self.both_non_empty_count >= self.stabilize_frames:
                self.state = self.EVALUATE
                print("[CYCLE] both CST stable -> EVALUATE")
            elif self.stabilize_elapsed >= self.stabilize_timeout_frames:
                # 한쪽이 계속 NonEmpty가 안 되어도(예: 반대편만 계속 들락날락) 타임아웃 후
                # 강제로 EVALUATE 단계로 넘어가 INCOMPLETE 등으로 1회 판정을 확정한다.
                self.state = self.EVALUATE
                print("[CYCLE] stabilize timeout -> EVALUATE")

        elif self.state == self.LOCKED:
            if self.locked_status == "WARNING":
                # WARNING은 RIGHT CST에 Wafer가 존재한다고 판정된 Cycle이므로,
                # 이후 Wafer가 사라져도(오검출 흔들림) RIGHT CST 자체가 빠져
                # Empty가 될 때까지는 WARNING을 유지한다(안전 최우선).
                reset_condition = right_empty
                reset_message = "Right CST removed after WARNING -> RESET"
            else:
                # NORMAL/ABNORMAL/UNKNOWN/INCOMPLETE는 좌/우 중 하나라도
                # Empty가 되면(= 물건이 빠지면) Cycle을 종료한다.
                reset_condition = any_empty
                reset_message = "At least one CST removed -> RESET"

            if reset_condition:
                self.reset_empty_count += 1
                if self.reset_empty_count >= self.reset_empty_frames:
                    print(f"[CYCLE] {reset_message}")
                    self.reset()
            else:
                self.reset_empty_count = 0

        return self.state

    def lock(self, status: str, message: str, info: dict) -> None:
        """EVALUATE 결과를 확정하여 LOCKED 상태로 전이한다."""
        self.locked_status = status
        self.locked_message = message
        self.locked_info = info
        self.event_triggered = False
        self.review_saved = False
        self.reset_empty_count = 0
        self.state = self.LOCKED
        print(f"[CYCLE] LOCKED status={status}, msg={message}")

    def reset(self) -> None:
        """Cycle을 종료하고 WAIT_ANY_CST(초기) 상태로 되돌린다."""
        self.state = self.WAIT
        self.both_non_empty_count = 0
        self.stabilize_elapsed = 0
        self.reset_empty_count = 0
        self.locked_status = "IDLE"
        self.locked_message = "Waiting for Left/Right CST"
        self.locked_info = {}
        self.event_triggered = False


# ==============================================================================
# 12. 재학습용 프레임 자동 수집기
# ==============================================================================

class AutoFrameCollector:
    """모델 재학습 후보 프레임을 자동으로 저장하고 디스크 용량을 관리하는 컬렉터.

    저장 구조
        - ``auto_frames/random/YYYYMMDD/{full,left,right,meta}/``
          : Cycle 상태와 무관하게 일정 확률로 무작위 샘플링하여 저장(전체 분포 학습용).
        - ``auto_frames/review_queue/{WARNING,ABNORMAL,UNKNOWN,INCOMPLETE}/YYYYMMDD/``
          : 이상(주의) 판정이 나온 Cycle의 프레임을 별도 큐에 저장(오탐/난케이스 학습용).

    주의: review_queue에 기록되는 라벨은 모델/규칙의 예측값(pseudo label)이므로,
    실제 재학습에 사용하기 전에 반드시 사람이 검수한 뒤 정식 학습 데이터 폴더로
    옮겨야 한다.

    용량 관리
        - ``max_total_gb`` 를 ``random_quota_ratio`` 비율로 random/review에 배분한다.
        - 각 영역이 할당량을 초과하면 오래된 파일부터 삭제한다(FIFO retention).
        - 디스크 여유 공간이 ``min_free_gb`` 미만이면 random 저장을 일시 중지한다.
    """

    REVIEW_STATUSES = {"WARNING", "ABNORMAL", "UNKNOWN", "INCOMPLETE"}

    def __init__(
        self,
        root_dir: Union[str, Path],
        enabled: bool = True,
        random_interval_sec: float = 30.0,
        random_probability: float = 0.2,
        max_total_gb: float = 5.0,
        random_quota_ratio: float = 0.8,
        storage_check_interval_sec: float = 600.0,
        min_free_gb: float = 5.0,
        jpeg_quality: int = 90,
    ):
        self.root_dir = ensure_dir(root_dir)
        self.random_root = ensure_dir(self.root_dir / "random")
        self.review_root = ensure_dir(self.root_dir / "review_queue")
        self.enabled = enabled

        self.random_interval_sec = max(1.0, random_interval_sec)
        self.random_probability = min(1.0, max(0.0, random_probability))

        self.max_total_bytes = int(max_total_gb * 1024 ** 3)
        self.random_max_bytes = int(self.max_total_bytes * random_quota_ratio)
        self.review_max_bytes = self.max_total_bytes - self.random_max_bytes

        self.storage_check_interval_sec = max(30.0, storage_check_interval_sec)
        self.min_free_bytes = int(min_free_gb * 1024 ** 3)
        self.jpeg_quality = int(min(100, max(50, jpeg_quality)))

        self.last_random_candidate_time = 0.0
        self.last_storage_check_time = 0.0
        self.random_save_paused = False

    # ---- 내부 유틸 --------------------------------------------------------

    @staticmethod
    def _dir_size(folder: Path) -> int:
        """폴더 하위(재귀) 전체 파일 크기 합계(byte)를 계산한다."""
        total = 0
        for p in folder.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    @staticmethod
    def _files_oldest_first(folder: Path) -> List[Path]:
        """폴더 하위 파일들을 수정시각(오래된 순) 기준으로 정렬해 반환한다."""
        files = [p for p in folder.rglob("*") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
        return files

    def _enforce_quota(self, folder: Path, max_bytes: int) -> None:
        """``folder`` 의 총 용량이 ``max_bytes`` 를 넘으면 오래된 파일부터 삭제한다."""
        total = self._dir_size(folder)
        if total <= max_bytes:
            return
        for p in self._files_oldest_first(folder):
            if total <= max_bytes:
                break
            try:
                size = p.stat().st_size
                p.unlink()
                total -= size
                print(f"[AUTO_FRAME] quota delete: {p}")
            except OSError as e:
                print(f"[WARN] auto-frame quota delete failed: {p}, error={e}")

    def check_storage(self, force: bool = False) -> None:
        """디스크 여유공간을 확인하고, 용량 초과 시 정리(quota enforcement)를 수행한다.

        호출 빈도를 줄이기 위해 ``storage_check_interval_sec`` 마다만 실제 검사를
        수행하며, ``force=True`` 로 즉시 실행할 수 있다.
        """
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self.last_storage_check_time < self.storage_check_interval_sec:
            return
        self.last_storage_check_time = now

        usage = shutil.disk_usage(self.root_dir)
        self.random_save_paused = usage.free < self.min_free_bytes
        if self.random_save_paused:
            print(f"[WARN] random frame save paused: free={usage.free / 1024 ** 3:.2f}GB")

        self._enforce_quota(self.random_root, self.random_max_bytes)
        self._enforce_quota(self.review_root, self.review_max_bytes)

    def _write_image(self, path: Path, image: np.ndarray) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]))

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_bundle(
        self,
        base_dir: Path,
        prefix: str,
        frame: np.ndarray,
        left_img: np.ndarray,
        right_img: np.ndarray,
        metadata: dict,
    ) -> Dict[str, str]:
        """전체 프레임 + 좌/우 크롭 이미지 + 메타데이터 JSON을 한 세트로 저장한다."""
        paths = {
            "full": base_dir / "full" / f"{prefix}_full.jpg",
            "left": base_dir / "left" / f"{prefix}_left.jpg",
            "right": base_dir / "right" / f"{prefix}_right.jpg",
            "meta": base_dir / "meta" / f"{prefix}.json",
        }
        self._write_image(paths["full"], frame)
        self._write_image(paths["left"], left_img)
        self._write_image(paths["right"], right_img)

        metadata = dict(metadata)
        metadata.update({k + "_path": str(v) for k, v in paths.items() if k != "meta"})
        self._write_json(paths["meta"], metadata)
        return {k: str(v) for k, v in paths.items()}

    # ---- 외부 공개 API ------------------------------------------------------

    def maybe_save_random(
        self,
        frame: np.ndarray,
        left_img: np.ndarray,
        right_img: np.ndarray,
        cycle_state: str,
        status: str,
        info: dict,
    ) -> Optional[Dict[str, str]]:
        """일정 주기(interval)마다 확률적으로 프레임을 무작위 저장한다(전체 분포 샘플링).

        Cycle이 LOCKED 상태이면 그 판정 결과를 pseudo_label로 함께 기록하고,
        그렇지 않으면 "UNLABELED"로 표시한다.
        """
        if not self.enabled:
            return None
        now = time.time()
        if now - self.last_random_candidate_time < self.random_interval_sec:
            return None
        self.last_random_candidate_time = now

        self.check_storage()
        if self.random_save_paused or random.random() > self.random_probability:
            return None

        dt = datetime.now()
        prefix = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_dir = self.random_root / dt.strftime("%Y%m%d")
        metadata = {
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "collection_type": "random",
            "cycle_state": cycle_state,
            "status": status if cycle_state == CycleStateMachine.LOCKED else "UNLABELED",
            "pseudo_label": cycle_state == CycleStateMachine.LOCKED,
            "reviewed": False,
            **info,
        }
        saved = self._save_bundle(base_dir, prefix, frame, left_img, right_img, metadata)
        print(f"[AUTO_FRAME] random saved: {saved['full']}")
        return saved

    def save_review_case(
        self,
        status: str,
        frame: np.ndarray,
        left_img: np.ndarray,
        right_img: np.ndarray,
        info: dict,
        message: str,
    ) -> Optional[Dict[str, str]]:
        """이상(WARNING/ABNORMAL/UNKNOWN/INCOMPLETE) 판정이 확정된 Cycle의 프레임을
        검수 대기 큐(review_queue)에 저장한다.

        디스크 여유공간이 2GB 미만이면 저장을 건너뛴다(로그 등 필수 기록 공간 보호).
        """
        status = str(status).upper()
        if not self.enabled or status not in self.REVIEW_STATUSES:
            return None

        self.check_storage()
        usage = shutil.disk_usage(self.root_dir)
        if usage.free < 2 * 1024 ** 3:
            print("[WARN] review frame save skipped: free disk below 2GB")
            return None

        dt = datetime.now()
        prefix = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_dir = self.review_root / status / dt.strftime("%Y%m%d")
        metadata = {
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "collection_type": "review_queue",
            "status": status,
            "message": message,
            "pseudo_label": True,
            "reviewed": False,
            **info,
        }
        saved = self._save_bundle(base_dir, prefix, frame, left_img, right_img, metadata)
        print(f"[AUTO_FRAME] review saved: status={status}, path={saved['full']}")
        return saved


# ==============================================================================
# 13. 이상 이벤트 클립 레코더
# ==============================================================================

class EventClipRecorder:
    """WARNING/ABNORMAL 발생 시점 전/후 구간을 포함한 mp4 클립을 저장하는 레코더.

    동작 방식
        - 매 프레임 ``append_pre_frame()`` 으로 오버레이된 프레임을 순환 버퍼
          (pre_buffer, 길이=pre_frames)에 계속 쌓아둔다.
        - 이상 이벤트가 발생하면 ``trigger()`` 를 호출한다. 이때 지금까지 쌓인
          pre_buffer 전체 + 현재 프레임을 새 mp4 파일에 먼저 기록해 "사건 발생
          이전" 상황을 함께 남긴다.
        - 이후 ``update()`` 가 ``post_frames`` 만큼 더 호출되면(=이벤트 발생 후
          일정 시간) 자동으로 클립을 닫는다.
        - 같은 이벤트가 짧은 간격으로 반복되어 클립이 과도하게 생성되는 것을
          막기 위해 ``cooldown_seconds`` 동안은 새 트리거를 무시한다.
    """

    def __init__(
        self,
        event_dir: Union[str, Path],
        fps: float,
        frame_size: Tuple[int, int],
        pre_seconds: float = 10.0,
        post_seconds: float = 10.0,
        codec: str = "mp4v",
        cooldown_seconds: float = 10.0,
    ):
        self.event_dir = ensure_dir(event_dir)
        self.fps = float(fps)
        self.frame_size = frame_size
        self.pre_frames = max(1, int(round(self.fps * pre_seconds)))
        self.post_frames = max(1, int(round(self.fps * post_seconds)))
        self.codec = codec
        self.cooldown_seconds = cooldown_seconds

        self.pre_buffer: Deque[np.ndarray] = deque(maxlen=self.pre_frames)
        self.writer: Optional[cv2.VideoWriter] = None
        self.active_path: Optional[Path] = None
        self.remaining_post_frames = 0
        self.last_event_time = 0.0

    def append_pre_frame(self, frame: np.ndarray) -> None:
        """이벤트 발생 이전 구간을 위해 매 프레임 순환 버퍼에 저장한다."""
        self.pre_buffer.append(frame.copy())

    def trigger(self, event_type: str, frame: np.ndarray) -> Optional[Path]:
        """이상 이벤트 클립 녹화를 시작한다.

        이미 녹화 중이거나 쿨다운 시간이 지나지 않았으면 아무 것도 하지 않고
        ``None`` 을 반환한다. 정상적으로 시작되면 pre_buffer 전체와 현재
        프레임을 먼저 기록한 뒤, 새로 생성된 클립 경로를 반환한다.
        """
        now = time.time()
        if self.writer is not None:
            return None
        if (now - self.last_event_time) < self.cooldown_seconds:
            return None

        event_type = str(event_type).upper()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.event_dir / f"{ts}_{event_type}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*self.codec), self.fps, self.frame_size)
        if not writer.isOpened():
            print(f"[WARN] event writer open failed: {path}")
            return None

        for buffered in self.pre_buffer:
            writer.write(buffered)
        writer.write(frame)

        self.writer = writer
        self.active_path = path
        self.remaining_post_frames = self.post_frames
        self.last_event_time = now
        print(f"[EVENT] clip started: {path}")
        return path

    def update(self, frame: np.ndarray) -> None:
        """녹화 중이면 post 구간 프레임을 기록하고, 다 채워지면 클립을 닫는다."""
        if self.writer is None:
            return
        if self.remaining_post_frames > 0:
            self.writer.write(frame)
            self.remaining_post_frames -= 1
        if self.remaining_post_frames <= 0:
            self.writer.release()
            print(f"[EVENT] clip saved: {self.active_path}")
            self.writer = None
            self.active_path = None

    def close(self) -> None:
        """프로그램 종료 시 녹화 중이던 클립이 있으면 안전하게 마무리한다."""
        if self.writer is not None:
            self.writer.release()
            print(f"[EVENT] clip saved: {self.active_path}")
            self.writer = None
            self.active_path = None


# ==============================================================================
# 14. 화면 오버레이 렌더링
# ==============================================================================

def draw_transparent_box(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color, alpha: float = 0.35) -> np.ndarray:
    """``pt1``~``pt2`` 사각형 영역에 반투명 색상 박스를 그려 텍스트 가독성을 높인다."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_overlay(
    frame: np.ndarray,
    left_roi: Tuple[int, int, int, int],
    right_roi: Tuple[int, int, int, int],
    cycle_state: str,
    left_empty: bool,
    right_empty: bool,
    left_empty_score: float,
    right_empty_score: float,
    status: str,
    message: str,
    alarm_action: str,
    info: dict,
    stabilize_info: str,
) -> np.ndarray:
    """운영 화면(모니터링/녹화용)에 ROI 박스, Empty 상태, Cycle 상태, 최종 판정
    결과를 그려 넣은 오버레이 프레임을 생성한다.

    ``stabilize_info`` 인자는 현재 렌더링에는 사용되지 않지만(디버깅 시
    로그/툴팁 확장 지점으로 남겨둠), 호출부와의 인터페이스 호환을 위해
    시그니처를 유지한다.
    """
    out = frame.copy()

    # --- 좌/우 ROI 박스 + Empty 상태 라벨 ---
    roi_infos = [
        (left_roi, "LEFT", left_empty, left_empty_score),
        (right_roi, "RIGHT", right_empty, right_empty_score),
    ]
    for roi, name, empty, score in roi_infos:
        x, y, w, h = roi
        color = DISPLAY_COLORS["empty"] if empty else DISPLAY_COLORS["non_pink"]
        label = f"{name}: {'EMPTY' if empty else 'NON_EMPTY'} | e={score:.2f}"
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        cv2.putText(out, label, (x, max(25, y + h + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

    # --- 하단 상태 패널 ---
    y_offset = 700
    x_box_max = 700
    status_color_map = {
        "IDLE": DISPLAY_COLORS["idle"],
        "NORMAL": DISPLAY_COLORS["normal"],
        "INCOMPLETE": DISPLAY_COLORS["incomplete"],
        "UNKNOWN": DISPLAY_COLORS["unknown"],
        "ABNORMAL": DISPLAY_COLORS["abnormal"],
        "WARNING": DISPLAY_COLORS["warning"],
    }
    status_color = status_color_map.get(status, (255, 255, 255))

    out = draw_transparent_box(out, (30, 30 + y_offset), (x_box_max, 255 + y_offset), (0, 0, 0), alpha=0.58)
    cv2.rectangle(out, (30, 30 + y_offset), (x_box_max, 255 + y_offset), status_color, 3)
    cv2.putText(out, f"STATE: {cycle_state}", (55, 75 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"STATUS: {status}", (55, 120 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.15, status_color, 3, cv2.LINE_AA)
    cv2.putText(out, message[:110], (55, 160 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2, cv2.LINE_AA)

    left_pink_text = info.get("left_pink", "unknown")
    right_pink_text = info.get("right_pink", "unknown")
    cv2.putText(
        out,
        f"Left CST={left_pink_text}, Right CST={right_pink_text}",
        (55, 200 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA,
    )

    right_wafer_label = info.get("right_wafer_label", "unknown")
    right_wafer_conf = float(info.get("right_wafer_conf", 0.0))
    cv2.putText(
        out,
        f"Right CST WAFER={right_wafer_label}({right_wafer_conf:.3f})",
        (55, 235 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


# ==============================================================================
# 15. 영상 소스 오픈 / CLI 인자 정의
# ==============================================================================

def open_video_source(args: argparse.Namespace) -> cv2.VideoCapture:
    """``--input_mode`` 설정에 따라 mp4 파일 또는 실시간 캡처 장치를 연다."""
    if args.input_mode == "mp4":
        if not args.video_path:
            raise ValueError("--input_mode mp4 사용 시 --video_path가 필요합니다.")
        cap = cv2.VideoCapture(args.video_path)
    else:
        backend = cv2.CAP_V4L2 if args.capture_backend == "v4l2" else 0
        cap = cv2.VideoCapture(args.capture_index, backend)
        if args.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
        cap.set(cv2.CAP_PROP_FPS, args.capture_fps)

    if not cap.isOpened():
        raise RuntimeError(f"video source open failed: mode={args.input_mode}")
    return cap


def build_argparser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 구성한다. 각 그룹의 의미는 아래 주석을 참고.

    그룹 구성: 입력 소스 / ROI / Empty 판정 / Pink 판정 / Wafer 모델 /
    Cycle 타이밍 / PATLITE / 이벤트 클립 / 로그 / 재학습 프레임 자동수집 / 화면표시
    """
    p = argparse.ArgumentParser(description="RMV CST Both-CST Cycle Lock Final Inference")

    # --- 입력 소스 ---
    p.add_argument("--input_mode", choices=["capture", "mp4"], default="capture")
    p.add_argument("--video_path", default="")
    p.add_argument("--capture_index", type=int, default=0)
    p.add_argument("--capture_backend", choices=["default", "v4l2"], default="v4l2")
    p.add_argument("--capture_width", type=int, default=1920)
    p.add_argument("--capture_height", type=int, default=1080)
    p.add_argument("--capture_fps", type=float, default=30.0)
    p.add_argument("--fourcc", default="MJPG")

    # --- ROI ---
    p.add_argument("--left_roi", nargs=4, type=int, default=list(DEFAULT_LEFT_ROI))
    p.add_argument("--right_roi", nargs=4, type=int, default=list(DEFAULT_RIGHT_ROI))

    # --- Empty 판정 ---
    p.add_argument("--reference_empty_dir", default=str(DEFAULT_REFERENCE_EMPTY_DIR))
    p.add_argument("--left_empty_threshold", type=float, default=DEFAULT_LEFT_EMPTY_THRESHOLD)
    p.add_argument("--right_empty_threshold", type=float, default=DEFAULT_RIGHT_EMPTY_THRESHOLD)
    p.add_argument("--empty_center_crop_ratio", type=float, default=0.7)
    p.add_argument("--blur_ksize", type=int, default=5)
    p.add_argument("--empty_hold_frames", type=int, default=5)

    # --- Pink 판정 ---
    p.add_argument("--color_center_crop_ratio", type=float, default=0.8)
    p.add_argument("--pink_rule_file", default=str(DEFAULT_PINK_RULE_FILE))

    # --- Wafer 모델 ---
    p.add_argument("--engine_path", default=str(DEFAULT_ENGINE_PATH))
    p.add_argument("--meta_path", default=str(DEFAULT_META_PATH))
    p.add_argument("--wafer_conf_threshold", type=float, default=0.5)
    p.add_argument("--disable_wafer_model", type=str2bool, default=False)
    p.add_argument("--wafer_check_interval", type=int, default=2)

    # --- Cycle 타이밍 ---
    p.add_argument("--stabilize_frames", type=int, default=10)
    p.add_argument("--stabilize_timeout_frames", type=int, default=60)
    p.add_argument("--reset_empty_frames", type=int, default=5)

    # --- PATLITE ---
    p.add_argument("--use_patlite", type=str2bool, default=False)

    # --- 이벤트 클립 ---
    p.add_argument("--event_dir", default=str(DEFAULT_EVENT_DIR))
    p.add_argument("--event_pre_seconds", type=float, default=10.0)
    p.add_argument("--event_post_seconds", type=float, default=10.0)
    p.add_argument("--event_cooldown_seconds", type=float, default=10.0)
    p.add_argument("--event_codec", default="mp4v")

    # --- 로그 ---
    p.add_argument("--log_dir", default=str(DEFAULT_LOG_DIR))
    p.add_argument("--history_dir", default=str(DEFAULT_HISTORY_DIR))
    p.add_argument("--keep_log_days", type=int, default=7)
    p.add_argument("--log_every_n_frames", type=int, default=1)

    # --- 재학습 프레임 자동수집 ---
    p.add_argument("--enable_auto_frame_save", type=str2bool, default=True)
    p.add_argument("--auto_frame_dir", default=str(DEFAULT_AUTO_FRAME_DIR))
    p.add_argument("--random_save_interval_sec", type=float, default=30.0)
    p.add_argument("--random_save_probability", type=float, default=0.2)
    p.add_argument("--auto_frame_max_gb", type=float, default=5.0)
    p.add_argument("--random_quota_ratio", type=float, default=0.8)
    p.add_argument("--storage_check_interval_sec", type=float, default=600.0)
    p.add_argument("--min_free_gb", type=float, default=5.0)
    p.add_argument("--auto_frame_jpeg_quality", type=int, default=90)

    # --- 화면 표시 / 실행 제어 ---
    p.add_argument("--show_window", type=str2bool, default=False)
    p.add_argument("--display_scale", type=float, default=0.5)
    p.add_argument("--max_frames", type=int, default=-1)
    return p


# ==============================================================================
# 16. CSTInferencePipeline : 프레임 단위 처리 실행기
# ==============================================================================
#
# main() 함수 하나에 모든 컴포넌트 초기화 + while 루프 로직이 섞여 있으면
# (1) 다른 스크립트/노트북에서 이 파이프라인을 재사용하기 어렵고
# (2) 단위 테스트가 어렵다는 문제가 있어, 아래와 같이 클래스로 캡슐화했다.
# 프레임 1장을 처리하는 알고리즘(추론/상태전이/판정/로그/이벤트/자동수집)
# 자체는 원본과 동일하며, 호출 구조만 재구성했다.
#
# 사용 예시(다른 스크립트에서 재사용할 경우)
#     args = build_argparser().parse_args([])
#     pipeline = CSTInferencePipeline(args)
#     pipeline.run()
# ==============================================================================

class CSTInferencePipeline:
    """좌/우 CST 영상 프레임을 입력받아 Cycle 판정을 수행하는 전체 파이프라인.

    구성요소(참조 이미지, 색상 규칙, Wafer 분류기, PATLITE, 영상 소스, 로거,
    이벤트 레코더, 자동 프레임 수집기, Cycle 상태 머신)를 ``__init__`` 에서
    한 번에 초기화하고, ``run()`` 을 호출하면 영상 소스가 끝나거나(mp4)
    사용자가 종료할 때까지 프레임을 계속 처리한다.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.left_roi = parse_roi(args.left_roi)
        self.right_roi = parse_roi(args.right_roi)

        self.log_dir = ensure_dir(args.log_dir)
        self.history_dir = ensure_dir(args.history_dir)
        self.event_dir = ensure_dir(args.event_dir)
        cleanup_old_files(self.log_dir, keep_days=args.keep_log_days)
        cleanup_old_files(self.history_dir, keep_days=args.keep_log_days)

        # --- Empty 판정용 참조 이미지 ---
        self.refs = load_empty_refs(args.reference_empty_dir, args.empty_center_crop_ratio, args.blur_ksize)
        if not self.refs["left"]:
            raise FileNotFoundError(f"left empty reference not found: {Path(args.reference_empty_dir) / 'left'}")
        if not self.refs["right"]:
            raise FileNotFoundError(f"right empty reference not found: {Path(args.reference_empty_dir) / 'right'}")

        # --- Pink 판정 규칙 / Wafer 분류기 / PATLITE ---
        self.pink_rules = normalize_rule_config(safe_json_load(args.pink_rule_file))
        self.wafer_classifier = (
            DummyWaferClassifier() if args.disable_wafer_model
            else TensorRTWaferClassifier(args.engine_path, args.meta_path)
        )
        self.patlite = PatliteLR6USBController(enabled=args.use_patlite)

        # --- 영상 소스 ---
        self.cap = open_video_source(args)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0:
            fps = args.capture_fps if args.capture_fps > 0 else 30.0
        self.fps = fps
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.capture_width)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.capture_height)
        self.frame_size = (width, height)

        print(f"[INFO] source opened: mode={args.input_mode}, {width}x{height}, fps={fps:.2f}")
        print(f"[INFO] left_roi={self.left_roi}, right_roi={self.right_roi}")
        print(
            f"[INFO] cycle: stabilize_frames={args.stabilize_frames}, "
            f"stabilize_timeout_frames={args.stabilize_timeout_frames}, "
            f"reset_empty_frames={args.reset_empty_frames}"
        )

        # --- 이벤트 클립 / Cycle 상태 머신 / 디바운스 상태 ---
        self.event_recorder = EventClipRecorder(
            event_dir=self.event_dir, fps=fps, frame_size=self.frame_size,
            pre_seconds=args.event_pre_seconds, post_seconds=args.event_post_seconds,
            codec=args.event_codec, cooldown_seconds=args.event_cooldown_seconds,
        )
        self.cycle = CycleStateMachine(
            stabilize_frames=args.stabilize_frames,
            stabilize_timeout_frames=args.stabilize_timeout_frames,
            reset_empty_frames=args.reset_empty_frames,
        )
        self.left_empty_hold = BoolHoldState(True, True, 0)
        self.right_empty_hold = BoolHoldState(True, True, 0)

        # --- 로거 / 자동 프레임 수집기 ---
        self.frame_logger = DailyCsvLogger(self.log_dir, prefix="frame_results", keep_days=args.keep_log_days)
        self.event_logger = DailyCsvLogger(self.log_dir, prefix="event_summary", keep_days=args.keep_log_days)
        self.run_logger = DailyCsvLogger(self.history_dir, prefix="run_status", keep_days=args.keep_log_days)
        self.auto_collector = AutoFrameCollector(
            root_dir=args.auto_frame_dir,
            enabled=args.enable_auto_frame_save,
            random_interval_sec=args.random_save_interval_sec,
            random_probability=args.random_save_probability,
            max_total_gb=args.auto_frame_max_gb,
            random_quota_ratio=args.random_quota_ratio,
            storage_check_interval_sec=args.storage_check_interval_sec,
            min_free_gb=args.min_free_gb,
            jpeg_quality=args.auto_frame_jpeg_quality,
        )
        self.auto_collector.check_storage(force=True)

        self.run_logger.write({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": "RUN_START",
            "input_mode": args.input_mode,
            "video_path": args.video_path,
            "left_roi": str(list(self.left_roi)),
            "right_roi": str(list(self.right_roi)),
            "engine_path": str(args.engine_path),
            "pink_rule_file": str(args.pink_rule_file),
            "auto_frame_dir": str(args.auto_frame_dir),
            "auto_frame_max_gb": args.auto_frame_max_gb,
        })

        self.frame_log_count = 0
        self.event_log_count = 0

    # -------------------------------------------------------------------
    # 내부 헬퍼: STABILIZE 단계의 "RIGHT 즉시 Wafer 체크"
    # -------------------------------------------------------------------
    def _check_immediate_right_wafer(
        self,
        right_img: np.ndarray,
        left_empty: bool,
        right_empty: bool,
        le_score: float,
        re_score: float,
        frame_idx: int,
    ) -> Optional[Tuple[str, str, dict]]:
        """STABILIZE 단계에서 RIGHT CST가 NonEmpty가 되는 즉시 Wafer 존재 여부를
        선제적으로 확인한다(1순위 안전 규칙).

        좌/우 안정화 완료 여부와 무관하게, RIGHT CST 내부에 Wafer가 있다고
        판정되면 즉시 WARNING으로 Lock하고, RIGHT CST가 Empty가 될 때까지
        그 상태를 유지한다(안전 최우선 정책).

        Returns:
            즉시 WARNING이 확정된 경우 (status, message, info) 튜플, 그렇지
            않으면 ``None`` (평소처럼 STABILIZE/IDLE 표시를 유지).
        """
        args = self.args
        if right_empty:
            return None
        if frame_idx % max(1, args.wafer_check_interval) != 0:
            return None

        _, _, wafer_probs = self.wafer_classifier.predict(right_img)
        wafer_prob = float(wafer_probs.get(self.wafer_classifier.positive_label, 0.0))
        right_wafer_label = (
            self.wafer_classifier.positive_label
            if wafer_prob >= args.wafer_conf_threshold
            else "no_wafer"
        )
        if right_wafer_label != self.wafer_classifier.positive_label:
            return None

        status = "WARNING"
        message = "Wafer detected inside RIGHT CST before color evaluation"
        info = {
            "left_pink": "unknown",
            "right_pink": "unknown",
            "right_wafer_label": right_wafer_label,
            "right_wafer_conf": wafer_prob,
            "left_empty_at_eval": left_empty,
            "right_empty_at_eval": right_empty,
            "left_empty_score_at_eval": le_score,
            "right_empty_score_at_eval": re_score,
            "wafer_check_stage": "STABILIZE_IMMEDIATE",
        }
        self.cycle.lock(status=status, message=message, info=info)
        if not self.cycle.review_saved:
            self.auto_collector.save_review_case(status, self._current_frame, self._current_left_img, right_img, info, message)
            self.cycle.review_saved = True
        return status, message, info

    # -------------------------------------------------------------------
    # 내부 헬퍼: EVALUATE 단계의 최종(1회) 색상/Wafer 판정
    # -------------------------------------------------------------------
    def _evaluate_cycle(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        left_empty: bool,
        right_empty: bool,
        le_score: float,
        re_score: float,
    ) -> Tuple[str, str, dict]:
        """EVALUATE 단계에서 좌/우 색상 및 RIGHT Wafer 여부를 1회 판정하고 Lock한다."""
        left_pink = "unknown"
        right_pink = "unknown"
        right_wafer_label = "unknown"
        right_wafer_conf = 0.0

        if not left_empty:
            left_pink, _, _ = predict_pink(left_img, "left", self.pink_rules, self.args.color_center_crop_ratio, self.args.blur_ksize)

        if not right_empty:
            _, _, wafer_probs = self.wafer_classifier.predict(right_img)
            wafer_prob = float(wafer_probs.get(self.wafer_classifier.positive_label, 0.0))
            right_wafer_label = (
                self.wafer_classifier.positive_label
                if wafer_prob >= self.args.wafer_conf_threshold
                else "no_wafer"
            )
            right_wafer_conf = wafer_prob
            # RIGHT에 Wafer가 있으면(WARNING 사유) 굳이 색상까지 볼 필요는 없다.
            if right_wafer_label != self.wafer_classifier.positive_label:
                right_pink, _, _ = predict_pink(right_img, "right", self.pink_rules, self.args.color_center_crop_ratio, self.args.blur_ksize)

        status, message = derive_final_status(
            left_empty=left_empty, right_empty=right_empty,
            left_pink_label=left_pink, right_wafer_label=right_wafer_label,
            right_pink_label=right_pink, wafer_positive_label=self.wafer_classifier.positive_label,
        )
        info = {
            "left_pink": left_pink,
            "right_pink": right_pink,
            "right_wafer_label": right_wafer_label,
            "right_wafer_conf": right_wafer_conf,
            "left_empty_at_eval": left_empty,
            "right_empty_at_eval": right_empty,
            "left_empty_score_at_eval": le_score,
            "right_empty_score_at_eval": re_score,
        }
        self.cycle.lock(status=status, message=message, info=info)
        if not self.cycle.review_saved and status in AutoFrameCollector.REVIEW_STATUSES:
            self.auto_collector.save_review_case(status, self._current_frame, left_img, right_img, info, message)
            self.cycle.review_saved = True
        return status, message, info

    # -------------------------------------------------------------------
    # 프레임 1장 처리
    # -------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """프레임 1장에 대해 전체 파이프라인(추론 -> 상태전이 -> 판정 -> 로그 ->
        이벤트/자동수집 -> 오버레이 렌더링)을 실행하고, 화면 표시/녹화에 쓸
        오버레이 이미지를 반환한다.
        """
        args = self.args
        # _check_immediate_right_wafer / _evaluate_cycle 내부에서 review 저장 시
        # 원본 프레임/좌측 이미지가 필요하므로 인스턴스에 잠시 보관해둔다.
        self._current_frame = frame
        t0 = time.perf_counter()

        left_img = crop_roi(frame, self.left_roi)
        right_img = crop_roi(frame, self.right_roi)
        self._current_left_img = left_img

        # --- 1) Empty/NonEmpty 판정 (+ 디바운스) ---
        le_raw, le_score = predict_empty(left_img, self.refs["left"], args.left_empty_threshold, args.empty_center_crop_ratio, args.blur_ksize)
        re_raw, re_score = predict_empty(right_img, self.refs["right"], args.right_empty_threshold, args.empty_center_crop_ratio, args.blur_ksize)
        self.left_empty_hold = update_hold_bool(self.left_empty_hold, le_raw, args.empty_hold_frames)
        self.right_empty_hold = update_hold_bool(self.right_empty_hold, re_raw, args.empty_hold_frames)
        left_empty = self.left_empty_hold.stable_label
        right_empty = self.right_empty_hold.stable_label

        # --- 2) Cycle 상태 갱신 ---
        current_state = self.cycle.on_frame(left_empty=left_empty, right_empty=right_empty)
        status = self.cycle.locked_status
        message = self.cycle.locked_message
        info = dict(self.cycle.locked_info)

        if current_state == CycleStateMachine.STABILIZE:
            status = "IDLE"
            message = "CST detected. Stabilizing before color evaluation"
            info = {}
            immediate = self._check_immediate_right_wafer(right_img, left_empty, right_empty, le_score, re_score, frame_idx)
            if immediate is not None:
                status, message, info = immediate

        if current_state == CycleStateMachine.EVALUATE:
            status, message, info = self._evaluate_cycle(left_img, right_img, left_empty, right_empty, le_score, re_score)

        # --- 3) 알람(PATLITE) ---
        alarm_action = alarm_from_status(status)
        self.patlite.set_alarm(alarm_action)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # --- 4) 오버레이 렌더링 ---
        stabilize_info = (
            f"both_non_empty_count={self.cycle.both_non_empty_count}/{self.cycle.stabilize_frames}, "
            f"reset_empty_count={self.cycle.reset_empty_count}/{self.cycle.reset_empty_frames}"
        )
        overlay = draw_overlay(
            frame=frame, left_roi=self.left_roi, right_roi=self.right_roi,
            cycle_state=self.cycle.state, left_empty=left_empty, right_empty=right_empty,
            left_empty_score=le_score, right_empty_score=re_score,
            status=status, message=message, alarm_action=alarm_action,
            info=info, stabilize_info=stabilize_info,
        )

        # --- 5) 이벤트 클립 처리 ---
        self.event_recorder.append_pre_frame(overlay)
        if self.cycle.state == CycleStateMachine.LOCKED and status in ("WARNING", "ABNORMAL") and not self.cycle.event_triggered:
            event_path = self.event_recorder.trigger(status, overlay)
            self.cycle.event_triggered = True
            if event_path is not None:
                self.event_logger.write({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "frame_idx": frame_idx,
                    "event_type": status,
                    "clip_path": str(event_path),
                    "message": message,
                })
                self.event_log_count += 1
        self.event_recorder.update(overlay)

        # --- 6) 재학습용 랜덤 프레임 수집 ---
        self.auto_collector.maybe_save_random(
            frame=frame, left_img=left_img, right_img=right_img,
            cycle_state=self.cycle.state, status=status, info=info,
        )

        # --- 7) 프레임 단위 CSV 로그 ---
        if frame_idx % max(1, args.log_every_n_frames) == 0:
            self.frame_logger.write({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "frame_idx": frame_idx,
                "cycle_state": self.cycle.state,
                "both_non_empty_count": self.cycle.both_non_empty_count,
                "reset_empty_count": self.cycle.reset_empty_count,
                "inference_time_ms": round(float(infer_ms), 3),
                "left_empty": left_empty,
                "right_empty": right_empty,
                "left_empty_score": round(float(le_score), 6),
                "right_empty_score": round(float(re_score), 6),
                "status": status,
                "message": message,
                "alarm_action": alarm_action,
                "left_pink": info.get("left_pink", ""),
                "right_pink": info.get("right_pink", ""),
                "right_wafer_label": info.get("right_wafer_label", ""),
                "right_wafer_conf": round(float(info.get("right_wafer_conf", 0.0)), 6),
            })
            self.frame_log_count += 1

        return overlay

    # -------------------------------------------------------------------
    # 메인 루프
    # -------------------------------------------------------------------
    def run(self) -> None:
        """영상 소스가 끝나거나(mp4) 사용자가 종료(``q``/``ESC``)할 때까지 프레임을
        계속 읽어 ``process_frame()`` 을 호출하는 메인 루프.
        """
        args = self.args
        frame_idx = -1
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    if args.input_mode == "mp4":
                        print("[INFO] mp4 end")
                        break
                    print("[WARN] frame read failed")
                    time.sleep(0.05)
                    continue

                frame_idx += 1
                if args.max_frames > 0 and frame_idx >= args.max_frames:
                    break

                overlay = self.process_frame(frame, frame_idx)

                if args.show_window:
                    display = overlay
                    if args.display_scale != 1.0:
                        display = cv2.resize(overlay, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("RMV CST Both-CST Cycle Inference", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
        except KeyboardInterrupt:
            print("[INFO] interrupted by user")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """영상 소스/이벤트 클립/PATLITE/윈도우를 정리하고 RUN_END 로그를 남긴다."""
        args = self.args
        self.cap.release()
        self.event_recorder.close()
        self.patlite.cleanup()
        if args.show_window:
            cv2.destroyAllWindows()
        try:
            self.run_logger.write({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event": "RUN_END",
                "input_mode": args.input_mode,
                "video_path": args.video_path,
                "frame_log_count": self.frame_log_count,
                "event_log_count": self.event_log_count,
                "left_empty_threshold": args.left_empty_threshold,
                "right_empty_threshold": args.right_empty_threshold,
                "stabilize_frames": args.stabilize_frames,
                "stabilize_timeout_frames": args.stabilize_timeout_frames,
                "reset_empty_frames": args.reset_empty_frames,
                "empty_hold_frames": args.empty_hold_frames,
                "wafer_check_interval": args.wafer_check_interval,
            })
        except Exception as e:
            print(f"[WARN] run end log failed: {e}")
        print("[DONE] realtime daily logs updated")
        print(f"- logs: {self.log_dir}")
        print(f"- history: {self.history_dir}")
        print("=== DONE ===")


# ==============================================================================
# 17. 엔트리 포인트
# ==============================================================================

def main() -> None:
    """CLI 인자를 파싱하고 :class:`CSTInferencePipeline` 을 생성/실행한다."""
    args = build_argparser().parse_args()
    pipeline = CSTInferencePipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
