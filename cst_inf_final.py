"""
Remover CST 이상탐지
"""

# region Load Modules

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import math

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# endregion


# region Local Variables

DEFAULT_RIGHT_ROI = (1621, 320, 931, 914)
DEFAULT_LEFT_ROI = (4, 219, 388, 702)
DEFAULT_EMPTY_REFERENCE_DIR = "./reference_empty"
DEFAULT_RIGHT_EMPTY_THRESHOLD = 14.0
DEFAULT_LEFT_EMPTY_THRESHOLD = 15.0
DEFAULT_WAFER_MODEL_FILE = "./best_mobilenetv3_small_wafer_cls.pt"
DEFAULT_THRESHOLD_FILE = "./thresholds_all.json"
VALID_CLASSES = ["black", "gray", "blue", "green", "pink"]
DISPLAY_COLORS = {
    "black": (30, 30, 30),
    "gray": (160, 160, 160),
    "blue": (255, 0, 0),
    "green": (0, 180, 0),
    "pink": (203, 72, 255),
    "empty": (0, 255, 255),
    "unknown": (255, 255, 255),
    "no_cst": (180, 180, 180),
    "empty_or_no_wafer": (180, 180, 180),
}

# endregion


# region Classes

# =========================
# Hold State 클래스
# =========================

@dataclass
class BoolHoldState:
    """ 상태 유지 클래스(Bool 버전), N회 이상 연속으로 새로운 상태 유지 시 기존 상태 변경
    """
    stable_label: bool
    candidate_label: bool
    candidate_count: int = 0

@dataclass
class LabelHoldState:
    """ 상태 유지 클래스(Label 버전), N회 이상 연속으로 새로운 상태 유지 시 기존 상태 변경
    """
    stable_label: str
    candidate_label: str
    candidate_count: int = 0

def update_temporal_hold_bool(state: BoolHoldState, predicted_label: bool, hold_frames: int) -> BoolHoldState:
    """N회 이상 연속으로 새로운 상태 유지 시 기존 상태 변경

    Args:
        state (BoolHoldState): 현재 상태
        predicted_label (bool): 새로운 상태
        hold_frames (int): Hold 카운트

    Returns:
        BoolHoldState: 최종 업데이트된 현재 상태
    """    
    if predicted_label == state.stable_label:
        state.candidate_label = predicted_label
        state.candidate_count = 0
        return state
    if predicted_label == state.candidate_label:
        state.candidate_count += 1
    else:
        state.candidate_label = predicted_label
        state.candidate_count = 1
    if state.candidate_count >= hold_frames:
        state.stable_label = state.candidate_label
        state.candidate_count = 0
    return state

def update_temporal_hold_label(state: LabelHoldState, predicted_label: str, hold_frames: int) -> LabelHoldState:
    """ N회 이상 연속으로 새로운 상태 유지 시 기존 상태 변경

    Args:
        state (LabelHoldState): 현재 상태
        predicted_label (str): 새로운 상태
        hold_frames (int): Hold 상태

    Returns:
        LabelHoldState: 최종 업데이트된 현재 상태
    """    
    if predicted_label == state.stable_label:
        state.candidate_label = predicted_label
        state.candidate_count = 0
        return state
    if predicted_label == state.candidate_label:
        state.candidate_count += 1
    else:
        state.candidate_label = predicted_label
        state.candidate_count = 1
    if state.candidate_count >= hold_frames:
        state.stable_label = state.candidate_label
        state.candidate_count = 0
    return state

# =========================
# Wafer 존재 판정 클래스
# =========================

def get_device() -> torch.device:
    """ cuda 사용 가능하면 cuda, 아니면 cpu """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_wafer_model(num_classes: int = 2) -> nn.Module:
    """ 학습된 가중치 베이스 모델 생성 """
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

class WaferClassifier:
    def __init__(self, model_path: Union[str, Path], device: Optional[torch.device] = None):
        self.device = device if device is not None else get_device()
        ckpt = torch.load(str(model_path), map_location=self.device)
        self.class_to_idx = ckpt["class_to_idx"]
        raw_idx_to_class = ckpt["idx_to_class"]
        self.idx_to_class = {int(k): v for k, v in raw_idx_to_class.items()}
        self.img_size = int(ckpt.get("img_size", 224))
        self.model = build_wafer_model(num_classes=len(self.class_to_idx))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, roi_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb).convert("RGB")
        x = self.tf(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = self.idx_to_class[pred_idx]
        prob_dict = {label: float(probs[int(idx)]) for idx, label in self.idx_to_class.items()}
        return pred_label, float(probs[pred_idx]), prob_dict

# endregion


# region Functions

def ensure_dir(path: str | Path) -> Path:
    """ 폴더 상태 보장, 확인(생성) 후 경로 반환

    Args:
        path (str | Path): 폴더 경로

    Returns:
        Path: 폴더 경로
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_roi(values: List[str]) -> Tuple[int, int, int, int]:
    """ ROI 인자 파싱 후 결과 반환

    Args:
        values (List[str]): 입력받은 인자

    Returns:
        Tuple[int, int, int, int]: ROI 정보(x, y, w, h)
    """
    if len(values) != 4:
        raise ValueError("ROI는 x y w h 4개 정수여야 합니다.")
    return tuple(map(int, values))

def load_reference_images(reference_dir: str | Path) -> Dict[str, List[np.ndarray]]:
    """ ROI(Left, Right) 별 Empty 레퍼런스 이미지 로드

    Args:
        reference_dir (str | Path): 레퍼런스 이미지 폴더 경로

    Returns:
        Dict[str, List[np.ndarray]]: 레퍼런스 이미지 딕셔너리
    """
    reference_dir = Path(reference_dir)
    out: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
    for part in ["left", "right"]:
        d = reference_dir / part
        if not d.exists():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for fp in sorted(d.glob(ext)):
                img = cv2.imread(str(fp))
                if img is not None:
                    out[part].append(img)
    return out

def safe_json_load(path: str | Path) -> dict:
    """ JSON 파일 로드(safe 모드)

    Args:
        path (str | Path): JSON 파일 경로

    Returns:
        dict: 로드 정보
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normarlize_threshold_structure(data: dict) -> Dict[str, Dict[str, dict]]:
    """ Threshold 파일 구조 파싱

    Args:
        data (dict): threshold JSON 파일

    Returns:
        Dict[str, Dict[str, dict]]: LEFT/RIGHT ROI 별 Threshold 추출
    """
    def _extract_side(side_name: str) -> Optional[dict]:
        if side_name in data and isinstance(data[side_name], dict):
            return data[side_name]
        alt = f"thresholds_{side_name}"
        if alt in data and isinstance(data[alt], dict):
            return data[alt]
        return None

    right = _extract_side("right")
    left = _extract_side("left")

    if right is None and left is None:
        if all(k in data for k in VALID_CLASSES):
            right = data
            left = data
        else:
            raise ValueError("threshold JSON 구조를 해석할 수 없습니다. right/left 또는 공통 색상 키 구조가 필요합니다.")
    if right is None:
        right = left
    if left is None:
        left = right
    return {"right": right, "left": left}

# =========================
# 이미지 처리
# =========================
def crop_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """ 프레임 내 ROI 영역 추출

    Args:
        frame (np.ndarray): 프레임
        roi (Tuple[int, int, int, int]): 추출할 ROI 정보 (x, y, w, h)

    Returns:
        np.ndarray: 추출된 영역
    """
    x, y, w, h = roi
    H, W = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"유효하지 않은 ROI입니다: {roi}, frame size=({W},{H})")
    return frame[y1:y2, x1:x2].copy()

def center_crop(img: np.ndarray, ratio: float) -> np.ndarray:
    """ 이미지 중심으로부터 설정한 비율만큼 잘라낸 이미지 반환 """
    if ratio >= 0.999:
        return img
    h, w = img.shape[:2]
    ch = max(1, int(round(h * ratio)))
    cw = max(1, int(round(w * ratio)))
    y1 = (h - ch) // 2
    x1 = (w - cw) // 2
    return img[y1:y1 + ch, x1:x1 + cw].copy()

# =========================
# Empty 판정
# =========================
def preprocess_for_empty(img_bgr: np.ndarray, center_crop_ratio: float = 0.7, blur_ksize: int = 5) -> np.ndarray:
    """ Empty 판정을 위한 전처리 """
    """
    1) RGB 이미지 → 그레이스케일
    2) 중앙 부분만 잘라내기 : 선택한 비율
    3) 가우시안 블러 필터 적용 : 미세 노이즈 제거하여 비교 용이 목적
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = center_crop(gray, center_crop_ratio)
    if blur_ksize >= 3:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray

def compare_empty_absdiff(curr_img_bgr: np.ndarray, ref_img_bgr: np.ndarray,
                          center_crop_ratio: float = 0.7, blur_ksize: int = 5) -> float:
    """ ROI 이미지와 레퍼런스 이미지 비교 """
    """
    1) 동일 방식으로 전처리
    2) 이미지 사이즈 맞춤
    3) ☆ 밝기 명도 정규화 : ROI 이미지의 밝기를 레퍼런스 이미지의 밝기로 정규화
    4) 이미지와 레퍼런스 이미지 절대값 차이의 평균 반환
    """
    curr = preprocess_for_empty(curr_img_bgr, center_crop_ratio, blur_ksize).astype(np.float32)
    ref = preprocess_for_empty(ref_img_bgr, center_crop_ratio, blur_ksize).astype(np.float32)
    if curr.shape != ref.shape:
        ref = cv2.resize(ref, (curr.shape[1], curr.shape[0]), interpolation=cv2.INTER_LINEAR)
    curr_mean = float(curr.mean()) # 이미지 평균 밝기
    ref_mean = float(ref.mean()) # 레퍼런스 이미지 평균 밝기
    if curr_mean > 1e-6:
        curr = curr * (ref_mean / curr_mean)
    curr = np.clip(curr, 0, 255) # 밝기 스케일링 후 255 초과되는 픽셀값 255로 고정
    return float(np.abs(curr - ref).mean())

def predict_empty_min_score(curr_img_bgr: np.ndarray, ref_imgs_bgr: List[np.ndarray],
                            threshold: float, center_crop_ratio: float = 0.7,
                            blur_ksize: int = 5) -> Tuple[bool, float]:
    """ Empty 이미지 여부 판정 """
    """
    curr_img_bgr : ROI 이미지
    ref_imgs_bgr : 해당 ROI의 레퍼런스 이미지(N장)
    threshold : Empty 판정 임계값
    center_crop_ratio : ROI 중심부로부터 잘라내는 비율
    blur_ksize : blur 처리 마스크 사이즈
    """
    """
    1) ROI 이미지와 레퍼런스 이미지 사이 스코어 리스트 저장
    2) 차이가 가장 작은 레퍼런스 인덱스 추출
    3) 해당 인덱스의 스코어 기준 Threshold와 비교 : 작으면 Empty, 크면 NonEmpty
    """
    if not ref_imgs_bgr:
        return False, float("inf"), -1, []
    scores = [
        compare_empty_absdiff(curr_img_bgr, ref, center_crop_ratio=center_crop_ratio, blur_ksize=blur_ksize)
        for ref in ref_imgs_bgr
    ]
    best_idx = int(np.argmin(scores))
    best_score = float(scores[best_idx])
    return best_score <= threshold, best_score

# =========================
# 대표 색상 판정
# =========================
def hue_in_range(h: np.ndarray, low: float, high: float, wraps: bool = False) -> np.ndarray:
    if wraps or low > high:
        return (h >= low) | (h <= high)
    return (h >= low) & (h <= high)

def score_color_by_threshold(hsv: np.ndarray, th: dict) -> float:
    """ Threshold 기반 대표 색상 스코어 반환 """
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    mask = np.ones(h.shape, dtype=bool)

    if "s_min" in th:
        mask &= s >= float(th["s_min"])
    if "s_max" in th:
        mask &= s <= float(th["s_max"])
    if "v_min" in th:
        mask &= v >= float(th["v_min"])
    if "v_max" in th:
        mask &= v <= float(th["v_max"])

    if "h_low" in th and "h_high" in th:
        wraps = bool(th.get("wraps", False))
        mask &= hue_in_range(h, float(th["h_low"]), float(th["h_high"]), wraps=wraps)
    elif "h_center" in th and "h_radius" in th:
        center = float(th["h_center"])
        radius = float(th["h_radius"])
        low = (center - radius) % 180
        high = (center + radius) % 180
        wraps = (center - radius) < 0 or (center + radius) > 179
        mask &= hue_in_range(h, low, high, wraps=wraps)

    return float(mask.mean())

def preprocess_for_color(img_bgr: np.ndarray, center_crop_ratio: float = 0.8, blur_ksize: int = 5) -> np.ndarray:
    """ 대표 색상을 위한 전처리 """
    """
    1) 이미지 중심 기준으로 Ratio만큼 자른 이미지
    2) 가우시안 블러 적용
    3) RGB → HSV 변환
    """
    img = center_crop(img_bgr, center_crop_ratio)
    if blur_ksize >= 3:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def predict_color_rect(img_bgr: np.ndarray, thresholds_side: Dict[str, dict],
                       center_crop_ratio: float = 0.8, blur_ksize: int = 5,
                       min_confidence: float = 0.01) -> Tuple[str, Dict[str, float]]:
    """ ROI 영역의 대표 색상 예측 """
    """
    1) 이미지 전처리 : HSV 색상 변환
    2) 대표 색상 별 스코어 계산: 개별 Threshold 기준
    3) 가장 스코어가 높은 대표 색상 선정
        스코어가 너무 작을 경우 UnKnown 처리
    대표 색상과, 점수 반환
    """
    hsv = preprocess_for_color(img_bgr, center_crop_ratio=center_crop_ratio, blur_ksize=blur_ksize)
    scores: Dict[str, float] = {}
    for c in VALID_CLASSES:
        th = thresholds_side.get(c, {}) # 대표 색상 별 Threshold
        scores[c] = score_color_by_threshold(hsv, th) if isinstance(th, dict) else 0.0
    pred = max(scores, key=scores.get)
    if scores[pred] < min_confidence:
        return "unknown", None
    return pred, scores[pred]

def derive_final_status(left_empty: bool, right_empty: bool, right_wafer_label: str, left_color: str, right_color: str, wafer_positive_label: str) -> Tuple[str, str]:
    """ 종합 결과 판정

    Args:
        left_empty (bool): 좌측 CST Empty 여부
        right_empty (bool): 우측 CST Empty 여부
        right_wafer_label (str): 우측 CST Wafer 존재 결과
        left_color (str): 좌측 CST 대표 색상
        right_color (str): 우측 CST 대표 색상
        wafer_positive_label (str): CST 내 Wafer 존재

    Returns:
        Tuple[str, str]: Status, 설명
    """    
    if (right_wafer_label == wafer_positive_label) and (not right_empty):
        return "WARNING", "WARNING: wafer detected inside right CST"
    if left_empty or right_empty:
        return "INCOMPLETE", "At least one ROI is EMPTY"
    if left_color == "unknown" or right_color == "unknown":
        return "INCOMPLETE", "Color decision uncertain"
    if (left_color == "pink") ^ (right_color == "pink"):
        return "ABNORMAL", "Both CST colors are not pink"
    return "NORMAL", "Normal"

# =========================
# Overlay 프레임 생성
# =========================
def color_for_roi(empty_flag: bool, color_label: str) -> Tuple[int, int, int]:
    if empty_flag:
        return DISPLAY_COLORS["empty"]
    return DISPLAY_COLORS.get(color_label, DISPLAY_COLORS["unknown"])

def draw_transparent_box(img: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def draw_overlay(frame: np.ndarray, left_rect, right_rect, left_empty, right_empty, left_empty_score, right_empty_score,
                 right_wafer_label, right_wafer_conf,
                 left_color, right_color, left_color_score, right_color_score,
                 final_status, final_message) -> np.ndarray:
    out = frame.copy()
    left_box_color = color_for_roi(left_empty, left_color)
    right_box_color = color_for_roi(right_empty, right_color)
    for rect, box_color, name, empty_flag, empty_score, color_label, color_score in [
        (left_rect, left_box_color, "LEFT", left_empty, left_empty_score, left_color, left_color_score),
        (right_rect, right_box_color, "RIGHT", right_empty, right_empty_score, right_color, right_color_score),
    ]:
        x, y, w, h = rect
        thickness = 5 if final_status == "WARNING" and name == "RIGHT" else 3
        cv2.rectangle(out, (x, y), (x + w, y + h), box_color, thickness)
        if empty_flag:
            label = f"{name}: EMPTY | empty_score={empty_score:.2f}"
        else:
            label = f"{name}: {color_label} | color_score={color_score:.2f}"
        cv2.putText(out, label, (x, max(25, y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, box_color, 3, cv2.LINE_AA)
    status_color = {"NORMAL": (0, 255, 0), "ABNORMAL": (0, 0, 255), "WARNING": (0, 0, 255), "INCOMPLETE": (0, 255, 255)}.get(final_status, (255, 255, 255))
    if final_status == "WARNING":
        out = draw_transparent_box(out, (780, 45), (1600, 220), (0, 0, 255), alpha=0.28)
        cv2.rectangle(out, (780, 45), (1600, 220), (0, 0, 255), 5)
        cv2.putText(out, "!!! WARNING !!!", (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(out, "WAFER DETECTED INSIDE RIGHT CST", (800, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        out = draw_transparent_box(out, (780, 45), (1600, 205), (0, 0, 0), alpha=0.5)
        cv2.rectangle(out, (780, 45), (1600, 205), (0, 0, 0), 5)
        cv2.putText(out, f"STATUS: {final_status}", (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.6, status_color, 3, cv2.LINE_AA)
        cv2.putText(out, final_message[:90], (800, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 2, cv2.LINE_AA)
    if right_wafer_label == "no_cst":
        wafer_text = f"RIGHT CST: {right_wafer_label}"
    else:
        wafer_text = f"RIGHT CST: {right_wafer_label} | prob={right_wafer_conf:.3f}"
    cv2.putText(out, wafer_text, (800, 190 if final_status != "WARNING" else 203), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def build_argparser() -> argparse.ArgumentParser:
    """ 프로그램 실행 시 커맨드라인 인수 설정 함수

    Returns:
        argparse.ArgumentParser: 커맨드라인 인수 파서
    """
    p = argparse.ArgumentParser(description="영상 추론 및 overlay video 생성")
    p.add_argument("--video_path", required=True, help="입력 영상 경로")
    p.add_argument("--output_dir", required=True, help="출력 결과물 저장 폴더")
    # p.add_argument("--video_path", type=str, default="./test.mp4")
    # p.add_argument("--roi_json", type=str, default="./roi_info.json")
    # p.add_argument("--output_dir", type=str, default="./final_out_v3")
    p.add_argument("--left_roi", nargs=4, type=int, default=list(DEFAULT_LEFT_ROI))
    p.add_argument("--right_roi", nargs=4, type=int, default=list(DEFAULT_RIGHT_ROI))
    p.add_argument("--empty_reference_dir", default=DEFAULT_EMPTY_REFERENCE_DIR)
    p.add_argument("--left_empty_threshold", type=float, default=DEFAULT_LEFT_EMPTY_THRESHOLD)
    p.add_argument("--right_empty_threshold", type=float, default=DEFAULT_RIGHT_EMPTY_THRESHOLD)
    p.add_argument("--wafer_model_path", default=DEFAULT_WAFER_MODEL_FILE)
    p.add_argument("--threshold_file", default=DEFAULT_THRESHOLD_FILE)
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--empty_center_crop_ratio", type=float, default=0.7)
    p.add_argument("--color_center_crop_ratio", type=float, default=0.8)
    p.add_argument("--blur_ksize", type=float, default=0.8)
    p.add_argument("--temporal_hold", type=int, default=5)
    p.add_argument("--wafer_hold_frames", type=int, default=3)
    p.add_argument("--final_hold_frames", type=int, default=3)
    p.add_argument("--wafer_positive_label", type=str, default="wafer_present")
    p.add_argument("--wafer_conf_threshold", type=float, default=0.5)
    p.add_argument("--color_min_confidence", type=float, default=0.01)
    p.add_argument("--codec", default="mp4v")
    p.add_argument("--overlay_video_name", default="inference_overlay.mp4")
    p.add_argument("--save_csv_name", default="frame_results.csv")
    p.add_argument("--save_event_name", default="event_summary.csv")
    return p

def main() -> None:
    """ Main 실행 함수
    """
    # 1) 파성 생성 및 초기화
    args = build_argparser().parse_args()
    output_dir = ensure_dir(args.output_dir)    
    left_roi = parse_roi(args.left_roi)
    right_roi = parse_roi(args.right_roi)

    # 2) Empty 판정 레퍼런스 이미지 로드
    empty_refs = load_reference_images(args.empty_reference_dir)
    if len(empty_refs["left"]) == 0:
        raise FileNotFoundError(f"Right Empty Reference 이미지가 없습니다: {Path(args.empty_reference_dir) / 'left'}")
    if len(empty_refs["right"]) == 0:
        raise FileNotFoundError(f"Right Empty Reference 이미지가 없습니다: {Path(args.empty_reference_dir) / 'right'}")

    # 3) Left/Right threshold 로드
    thresholds = normarlize_threshold_structure(safe_json_load(args.threshold_file))

    # 4) Wafer 존재 판정 모델 로드
    wafer_classifier = WaferClassifier(args.wafer_model_path, device=get_device())

    # 5) 비디오 영상 로드
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {args.video_path}")

    # 6) 영상 프레임 설정
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # 7) 오버레이 영상 초기 설정
    video_out = output_dir / args.overlay_video_name
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"출력 영상을 생성할 수 없습니다: {video_out}")

    # 8) HoldState 초기화 : n번 연속으로 변경된 상태 유지 시 현재 상태 변경하는 메소드
    left_empty_hold = BoolHoldState(False, False, 0)
    right_empty_hold = BoolHoldState(False, False, 0)
    wafer_hold = LabelHoldState("no_cst", "no_cst", 0)
    left_color_hold = LabelHoldState("unknown", "unknown", 0)
    right_color_hold = LabelHoldState("unknown", "unknown", 0)
    final_hold = LabelHoldState("INCOMPLETE", "INCOMPLETE", 0)

    # 9) 프레임 단위 판정 + Overlay 이미지 생성
    results = []
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1
        if frame_idx % args.sample_every == 0:
            t_infer_start = time.perf_counter() # 추론 시간 측정: 시작

            # 1. Left / Right 영역 크롭
            left_img = crop_roi(frame, left_roi)
            right_img = crop_roi(frame, right_roi)

            # 2. Empty 여부 판정
            le_raw, le_score = predict_empty_min_score(
                left_img, empty_refs["left"], threshold=args.left_empty_threshold,
                center_crop_ratio=args.empty_center_crop_ratio, blur_ksize=args.blur_ksize)
            re_raw, re_score = predict_empty_min_score(
                right_img, empty_refs["right"], threshold=args.left_empty_threshold,
                center_crop_ratio=args.empty_center_crop_ratio, blur_ksize=args.blur_ksize)
            left_empty_hold = update_temporal_hold_bool(left_empty_hold, le_raw, args.temporal_hold)
            right_empty_hold = update_temporal_hold_bool(right_empty_hold, re_raw, args.temporal_hold)
            left_empty = left_empty_hold.stable_label
            right_empty = right_empty_hold.stable_label

            # 3. RIGHT 영역 empty 아닐 경우 Wafer 존재 여부 판정
            if not right_empty:
                _, _, probs = wafer_classifier.predict(right_img)
                wafer_prob = float(probs.get(args.wafer_positive_label, 0.0))
                wafer_candidate = args.wafer_positive_label if wafer_prob >= args.wafer_conf_threshold else "no_wafer"
                wafer_hold = update_temporal_hold_label(wafer_hold, wafer_candidate, args.wafer_hold_frames)
                wafer_label = wafer_hold.stable_label
                wafer_conf = wafer_prob
            else:
                wafer_hold = update_temporal_hold_label(wafer_hold, "no_cst", args.wafer_hold_frames)
                wafer_label = wafer_hold.stable_label
                wafer_conf = 0.0

            # 4. 대표 색상 판정
            left_color = right_color = "empty"
            lc_score = rc_score = np.nan
            if not left_empty:
                lc_raw, lc_score = predict_color_rect(left_img, thresholds["left"], args.color_center_crop_ratio,
                                                    args.blur_ksize, args.color_min_confidence)
                left_color_hold = update_temporal_hold_label(left_color_hold, lc_raw, args.temporal_hold)
                left_color = left_color_hold.stable_label
            if not right_empty:
                rc_raw, rc_score = predict_color_rect(right_img, thresholds["right"], args.color_center_crop_ratio,
                                                    args.blur_ksize, args.color_min_confidence)
                right_color_hold = update_temporal_hold_label(right_color_hold, rc_raw, args.temporal_hold)
                right_color = right_color_hold.stable_label

            # 5. 종합 Status 판정
            final_raw, final_msg = derive_final_status(left_empty, right_empty, wafer_label, left_color, right_color, args.wafer_positive_label)
            if final_raw == "INCOMPLETE":
                final_hold = LabelHoldState("INCOMPLETE", "INCOMPLETE", 0)
            else:
                final_hold = update_temporal_hold_label(final_hold, final_raw, args.final_hold_frames)
            final_status = final_hold.stable_label

            # 6. 프레임 단위 결과 append
            t_infer = time.perf_counter() - t_infer_start
            results.append(
                {
                    "frame_idx": frame_idx,
                    "time_sec": round(frame_idx / fps, 6),
                    "inference_time_ms": round(float(t_infer), 3),
                    "left_empty": left_empty,
                    "right_empty": right_empty,
                    "left_empty_score": round(le_score, 6),
                    "right_empty_score": round(re_score, 6),
                    "right_wafer_label": wafer_label,
                    "right_wafer_conf": round(float(wafer_conf), 6), 
                    "left_color": left_color,
                    "right_color": right_color,
                    "left_color_score": round(lc_score, 6),
                    "right_color_score": round(rc_score, 6),
                    "final_status": final_status,
                    "final_message": final_msg,
                }
            )
            annotated = draw_overlay(frame, left_roi, right_roi,
                                     left_empty, right_empty, le_score, re_score,
                                     wafer_label, round(float(wafer_conf), 6),
                                     left_color, right_color, lc_score, rc_score,
                                     final_status, final_msg)
            writer.write(annotated)
            # cv2.imwrite(f"./{output_dir}/frame_{str(frame_idx).zfill(6)}.png", annotated)                            
    csv_path = output_dir / args.save_csv_name
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("=== DONE ===")
    
# endregion


# region 파일 실행

if __name__ == "__main__":
    main()

# endregion
