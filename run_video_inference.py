from __future__ import annotations

import argparse
import json
import math
import sys
import time

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.anomaly.patchcore import (
    PatchCoreModel,
)

from src.anomaly.preprocessing import (
    AspectPadPreprocessor,
)


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}


# ============================================================
# Colors - BGR
# ============================================================

COLOR_WHITE = (
    240,
    240,
    240,
)

COLOR_GRAY = (
    130,
    130,
    130,
)

COLOR_DARK = (
    24,
    24,
    24,
)

COLOR_GREEN = (
    80,
    210,
    90,
)

COLOR_YELLOW = (
    0,
    220,
    255,
)

COLOR_RED = (
    60,
    60,
    245,
)

COLOR_ORANGE = (
    0,
    150,
    255,
)

COLOR_BLUE = (
    230,
    150,
    70,
)


# ============================================================
# Path
# ============================================================

def resolve_path(
    value: str | Path,
) -> Path:

    path = Path(
        value
    )

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


# ============================================================
# Image IO
# ============================================================

def imread_safe(
    path: Path,
) -> np.ndarray | None:

    try:

        data = np.fromfile(
            str(path),
            dtype=np.uint8,
        )

        if data.size == 0:
            return None

        return cv2.imdecode(
            data,
            cv2.IMREAD_COLOR,
        )

    except Exception:
        return None


# ============================================================
# ROI config
# ============================================================

def normalize_roi(
    value: Any,
) -> tuple[int, int, int, int] | None:

    if value is None:
        return None

    if (
        isinstance(value, (list, tuple))
        and len(value) == 4
    ):

        return tuple(
            int(v)
            for v in value
        )

    if isinstance(
        value,
        dict,
    ):

        keys = {
            "x",
            "y",
            "width",
            "height",
        }

        if keys.issubset(
            value.keys()
        ):

            return (
                int(value["x"]),
                int(value["y"]),
                int(value["width"]),
                int(value["height"]),
            )

    return None


def find_search_roi(
    *,
    roi_name: str,
    video_config: dict,
    roi_config: dict,
) -> tuple[int, int, int, int]:

    # --------------------------------------------------------
    # video_inference.yaml에서 직접 지정한 경우
    # --------------------------------------------------------

    direct = normalize_roi(
        video_config[
            "gate"
        ][
            roi_name
        ].get(
            "search_roi"
        )
    )

    if direct is not None:
        return direct

    # --------------------------------------------------------
    # 기존 roi.yaml 여러 구조 지원
    # --------------------------------------------------------

    candidates = []

    candidates.append(
        roi_config.get(
            "rois",
            {},
        ).get(
            roi_name,
            {},
        ).get(
            "search_roi"
        )
    )

    roi_section = roi_config.get(
        roi_name
    )

    if isinstance(
        roi_section,
        dict,
    ):
        candidates.append(
            roi_section.get(
                "search_roi"
            )
        )

    candidates.append(
        roi_config.get(
            "search_rois",
            {},
        ).get(
            roi_name
        )
    )

    # 혹시 left/right 자체가 ROI list인 경우
    candidates.append(
        roi_section
    )

    for candidate in candidates:

        roi = normalize_roi(
            candidate
        )

        if roi is not None:
            return roi

    raise ValueError(
        f"Could not find search_roi "
        f"for '{roi_name}'. "
        f"Set gate.{roi_name}.search_roi "
        f"in config/video_inference.yaml."
    )


def load_anomaly_roi(
    config: dict,
    roi_name: str,
) -> tuple[int, int, int, int]:

    value = (
        config[
            "rois"
        ][
            roi_name
        ][
            "anomaly_roi"
        ]
    )

    roi = normalize_roi(
        value
    )

    if roi is None:

        raise ValueError(
            f"Invalid anomaly_roi: "
            f"{roi_name}"
        )

    return roi


# ============================================================
# Gate result
# ============================================================

@dataclass
class GateResult:

    matched: bool = False

    status: str = "not_run"

    homography: (
        np.ndarray
        | None
    ) = None

    polygon: (
        np.ndarray
        | None
    ) = None

    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0

    area_ratio: float = float(
        "nan"
    )

    rotation_deg: float = float(
        "nan"
    )


# ============================================================
# Localize Planar Object
# ============================================================

class GateLocalizer:

    """
    ImageProcessingAPI의 localize_planar_object와
    같은 기본 흐름:

    Feature
      -> Lowe ratio
      -> Homography RANSAC
      -> polygon
      -> inliers
    """

    def __init__(
        self,
        template: np.ndarray,
        config: dict,
    ) -> None:

        self.template = template

        self.config = config

        self.algorithm = str(
            config.get(
                "algorithm",
                "sift",
            )
        ).lower()

        self.matcher_name = str(
            config.get(
                "matcher",
                "flann",
            )
        ).lower()

        self.max_features = int(
            config.get(
                "max_features",
                3000,
            )
        )

        self.ratio_threshold = float(
            config.get(
                "ratio_threshold",
                0.7,
            )
        )

        self.min_matches = int(
            config.get(
                "min_matches",
                4,
            )
        )

        self.max_matches = int(
            config.get(
                "max_matches",
                500,
            )
        )

        self.reprojection_threshold = float(
            config.get(
                "reprojection_threshold",
                5.0,
            )
        )

        # ----------------------------------------------------
        # Detector
        # ----------------------------------------------------

        if self.algorithm == "sift":

            self.detector = (
                cv2.SIFT_create(
                    nfeatures=(
                        self.max_features
                    )
                )
            )

            self.norm = (
                cv2.NORM_L2
            )

        elif self.algorithm == "orb":

            self.detector = (
                cv2.ORB_create(
                    nfeatures=(
                        self.max_features
                    )
                )
            )

            self.norm = (
                cv2.NORM_HAMMING
            )

        else:

            raise ValueError(
                f"Unsupported algorithm: "
                f"{self.algorithm}"
            )

        # ----------------------------------------------------
        # Template descriptor는 한 번만 계산
        # ----------------------------------------------------

        template_gray = (
            cv2.cvtColor(
                template,
                cv2.COLOR_BGR2GRAY,
            )
        )

        (
            self.template_keypoints,
            self.template_descriptors,
        ) = self.detector.detectAndCompute(
            template_gray,
            None,
        )

        if (
            self.template_descriptors
            is None
            or len(
                self.template_keypoints
            ) < 4
        ):

            raise RuntimeError(
                "Could not extract enough "
                "template features."
            )

    # ========================================================
    # Matcher
    # ========================================================

    def _create_matcher(
        self,
    ):

        if self.matcher_name == "bf":

            return cv2.BFMatcher(
                self.norm,
                crossCheck=False,
            )

        if self.algorithm == "sift":

            return cv2.FlannBasedMatcher(
                {
                    "algorithm": 1,
                    "trees": 5,
                },
                {
                    "checks": 64,
                },
            )

        # ORB + FLANN LSH
        return cv2.FlannBasedMatcher(
            {
                "algorithm": 6,
                "table_number": 6,
                "key_size": 12,
                "multi_probe_level": 1,
            },
            {
                "checks": 64,
            },
        )

    # ========================================================
    # Localize
    # ========================================================

    def localize(
        self,
        scene: np.ndarray,
    ) -> GateResult:

        gray = cv2.cvtColor(
            scene,
            cv2.COLOR_BGR2GRAY,
        )

        (
            scene_keypoints,
            scene_descriptors,
        ) = self.detector.detectAndCompute(
            gray,
            None,
        )

        if (
            scene_descriptors is None
            or len(
                scene_keypoints
            ) < 4
        ):

            return GateResult(
                status="no_features"
            )

        matcher = (
            self._create_matcher()
        )

        try:

            pairs = matcher.knnMatch(
                self.template_descriptors,
                scene_descriptors,
                k=2,
            )

        except cv2.error:

            return GateResult(
                status="match_failed"
            )

        good = []

        for pair in pairs:

            if len(pair) < 2:
                continue

            first, second = (
                pair[0],
                pair[1],
            )

            if (
                first.distance
                < self.ratio_threshold
                * second.distance
            ):
                good.append(
                    first
                )

        good.sort(
            key=lambda x: x.distance
        )

        good = good[
            :self.max_matches
        ]

        if len(
            good
        ) < self.min_matches:

            return GateResult(
                status=(
                    "insufficient_matches"
                ),
                good_matches=len(
                    good
                ),
            )

        source = np.float32(
            [
                self.template_keypoints[
                    match.queryIdx
                ].pt
                for match in good
            ]
        )

        destination = np.float32(
            [
                scene_keypoints[
                    match.trainIdx
                ].pt
                for match in good
            ]
        )

        (
            homography,
            mask,
        ) = cv2.findHomography(
            source.reshape(
                -1,
                1,
                2,
            ),
            destination.reshape(
                -1,
                1,
                2,
            ),
            cv2.RANSAC,
            self.reprojection_threshold,
        )

        if homography is None:

            return GateResult(
                status=(
                    "homography_failed"
                ),
                good_matches=len(
                    good
                ),
            )

        template_h, template_w = (
            self.template.shape[:2]
        )

        corners = np.float32(
            [
                [0, 0],
                [
                    template_w - 1,
                    0,
                ],
                [
                    template_w - 1,
                    template_h - 1,
                ],
                [
                    0,
                    template_h - 1,
                ],
            ]
        ).reshape(
            -1,
            1,
            2,
        )

        polygon = (
            cv2.perspectiveTransform(
                corners,
                homography,
            )
            .reshape(
                -1,
                2,
            )
        )

        if mask is None:

            inliers = len(
                good
            )

        else:

            inliers = int(
                mask.sum()
            )

        inlier_ratio = (
            inliers
            / max(
                1,
                len(good),
            )
        )

        # ----------------------------------------------------
        # Polygon area ratio
        # ----------------------------------------------------

        projected_area = abs(
            cv2.contourArea(
                polygon.astype(
                    np.float32
                )
            )
        )

        template_area = max(
            1.0,
            float(
                (
                    template_w - 1
                )
                * (
                    template_h - 1
                )
            ),
        )

        area_ratio = (
            projected_area
            / template_area
        )

        # ----------------------------------------------------
        # Rotation
        # ----------------------------------------------------

        dx = (
            polygon[1, 0]
            - polygon[0, 0]
        )

        dy = (
            polygon[1, 1]
            - polygon[0, 1]
        )

        rotation_deg = (
            math.degrees(
                math.atan2(
                    float(dy),
                    float(dx),
                )
            )
        )

        return GateResult(
            matched=True,
            status="matched",
            homography=(
                homography.astype(
                    np.float64
                )
            ),
            polygon=polygon,
            good_matches=len(
                good
            ),
            inliers=inliers,
            inlier_ratio=float(
                inlier_ratio
            ),
            area_ratio=float(
                area_ratio
            ),
            rotation_deg=float(
                rotation_deg
            ),
        )


# ============================================================
# Gate threshold
# ============================================================

def apply_gate_threshold(
    result: GateResult,
    config: dict,
) -> bool:

    if not result.matched:
        return False

    checks = [
        (
            "min_good_matches",
            result.good_matches,
            lambda v, t: v >= t,
        ),
        (
            "min_inliers",
            result.inliers,
            lambda v, t: v >= t,
        ),
        (
            "min_inlier_ratio",
            result.inlier_ratio,
            lambda v, t: v >= t,
        ),
        (
            "min_area_ratio",
            result.area_ratio,
            lambda v, t: v >= t,
        ),
        (
            "max_area_ratio",
            result.area_ratio,
            lambda v, t: v <= t,
        ),
        (
            "max_abs_rotation",
            abs(
                result.rotation_deg
            ),
            lambda v, t: v <= t,
        ),
    ]

    for (
        key,
        value,
        predicate,
    ) in checks:

        threshold = config.get(
            key
        )

        if threshold is None:
            continue

        if not predicate(
            value,
            float(
                threshold
            ),
        ):
            return False

    return True


# ============================================================
# Temporal Filter
# ============================================================

class TemporalFilter:

    def __init__(
        self,
        *,
        window_size: int,
        required_ng: int,
        reset_after_gate_miss: int,
    ) -> None:

        self.window_size = (
            window_size
        )

        self.required_ng = (
            required_ng
        )

        self.reset_after_gate_miss = (
            reset_after_gate_miss
        )

        self.history = deque(
            maxlen=window_size
        )

        self.gate_miss_count = 0

        self.alarm = False

    def reset(
        self,
    ) -> None:

        self.history.clear()

        self.gate_miss_count = 0

        self.alarm = False

    def update(
        self,
        raw_ng: bool,
    ) -> bool:

        self.gate_miss_count = 0

        self.history.append(
            bool(
                raw_ng
            )
        )

        self.alarm = (
            len(
                self.history
            )
            >= self.required_ng
            and sum(
                self.history
            )
            >= self.required_ng
        )

        return self.alarm

    def gate_miss(
        self,
    ) -> bool:

        self.gate_miss_count += 1

        if (
            self.gate_miss_count
            >= self.reset_after_gate_miss
        ):

            self.reset()

        return self.alarm

    @property
    def vote_count(
        self,
    ) -> int:

        return int(
            sum(
                self.history
            )
        )


# ============================================================
# Letterbox anomaly map reverse
# ============================================================

def restore_anomaly_map(
    anomaly_map: np.ndarray,
    *,
    original_width: int,
    original_height: int,
    input_size: int,
) -> np.ndarray:

    scale = min(
        input_size
        / original_width,
        input_size
        / original_height,
    )

    resized_width = max(
        1,
        int(
            round(
                original_width
                * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                original_height
                * scale
            )
        ),
    )

    offset_x = (
        input_size
        - resized_width
    ) // 2

    offset_y = (
        input_size
        - resized_height
    ) // 2

    content = anomaly_map[
        offset_y:
        offset_y + resized_height,

        offset_x:
        offset_x + resized_width,
    ]

    return cv2.resize(
        content,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    ).astype(
        np.float32
    )


# ============================================================
# ROI Runtime
# ============================================================

@dataclass
class RoiRuntime:

    name: str

    search_roi: tuple[
        int,
        int,
        int,
        int,
    ]

    anomaly_roi: tuple[
        int,
        int,
        int,
        int,
    ]

    template: np.ndarray

    localizer: GateLocalizer

    gate_config: dict

    patchcore: PatchCoreModel

    preprocessor: AspectPadPreprocessor

    threshold: float

    temporal: TemporalFilter


# ============================================================
# Result state
# ============================================================

@dataclass
class RoiState:

    roi: str

    gate_status: str = (
        "NOT RUN"
    )

    gate_pass: bool = False

    good_matches: int = 0
    inliers: int = 0

    inlier_ratio: float = float(
        "nan"
    )

    area_ratio: float = float(
        "nan"
    )

    rotation_deg: float = float(
        "nan"
    )

    valid_ratio: float = float(
        "nan"
    )

    score: float = float(
        "nan"
    )

    raw_ng: bool = False
    alarm: bool = False

    vote_count: int = 0
    history: list[bool] = field(
        default_factory=list
    )

    polygon_global: (
        np.ndarray
        | None
    ) = None

    crop: (
        np.ndarray
        | None
    ) = None

    anomaly_map: (
        np.ndarray
        | None
    ) = None

    gate_ms: float = 0.0
    patchcore_ms: float = 0.0


# ============================================================
# Alignment
# ============================================================

def align_and_crop(
    *,
    scene: np.ndarray,
    homography: np.ndarray,
    template_shape: tuple[
        int,
        int,
    ],
    anomaly_roi: tuple[
        int,
        int,
        int,
        int,
    ],
) -> tuple[
    np.ndarray | None,
    float,
]:

    template_h, template_w = (
        template_shape
    )

    try:

        inverse = np.linalg.inv(
            homography
        )

    except np.linalg.LinAlgError:

        return (
            None,
            0.0,
        )

    aligned = cv2.warpPerspective(
        scene,
        inverse,
        (
            template_w,
            template_h,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=(
            cv2.BORDER_CONSTANT
        ),
    )

    source_mask = np.full(
        scene.shape[:2],
        255,
        dtype=np.uint8,
    )

    valid_mask = cv2.warpPerspective(
        source_mask,
        inverse,
        (
            template_w,
            template_h,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=(
            cv2.BORDER_CONSTANT
        ),
        borderValue=0,
    )

    x, y, w, h = (
        anomaly_roi
    )

    if (
        x < 0
        or y < 0
        or x + w > template_w
        or y + h > template_h
    ):

        return (
            None,
            0.0,
        )

    crop = aligned[
        y:y + h,
        x:x + w,
    ]

    valid_crop = valid_mask[
        y:y + h,
        x:x + w,
    ]

    valid_ratio = float(
        np.mean(
            valid_crop > 0
        )
    )

    return (
        crop,
        valid_ratio,
    )


# ============================================================
# ROI inference
# ============================================================

@torch.no_grad()
def infer_roi(
    *,
    frame: np.ndarray,
    runtime: RoiRuntime,
    device: torch.device,
    min_valid_ratio: float,
) -> RoiState:

    state = RoiState(
        roi=runtime.name
    )

    sx, sy, sw, sh = (
        runtime.search_roi
    )

    frame_h, frame_w = (
        frame.shape[:2]
    )

    if (
        sx < 0
        or sy < 0
        or sx + sw > frame_w
        or sy + sh > frame_h
    ):

        state.gate_status = (
            "SEARCH ROI INVALID"
        )

        runtime.temporal.gate_miss()

        return state

    scene = frame[
        sy:sy + sh,
        sx:sx + sw,
    ]

    # --------------------------------------------------------
    # Gate
    # --------------------------------------------------------

    start = time.perf_counter()

    gate_result = (
        runtime.localizer.localize(
            scene
        )
    )

    state.gate_ms = (
        (
            time.perf_counter()
            - start
        )
        * 1000.0
    )

    state.gate_status = (
        gate_result.status
    )

    state.good_matches = (
        gate_result.good_matches
    )

    state.inliers = (
        gate_result.inliers
    )

    state.inlier_ratio = (
        gate_result.inlier_ratio
    )

    state.area_ratio = (
        gate_result.area_ratio
    )

    state.rotation_deg = (
        gate_result.rotation_deg
    )

    state.gate_pass = (
        apply_gate_threshold(
            gate_result,
            runtime.gate_config,
        )
    )

    # --------------------------------------------------------
    # Polygon global coordinate
    # --------------------------------------------------------

    if (
        gate_result.polygon
        is not None
    ):

        polygon = (
            gate_result
            .polygon
            .copy()
        )

        polygon[:, 0] += sx
        polygon[:, 1] += sy

        state.polygon_global = (
            polygon
        )

    if not state.gate_pass:

        runtime.temporal.gate_miss()

        state.alarm = (
            runtime.temporal.alarm
        )

        state.vote_count = (
            runtime.temporal.vote_count
        )

        state.history = list(
            runtime.temporal.history
        )

        return state

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    (
        crop,
        valid_ratio,
    ) = align_and_crop(
        scene=scene,
        homography=(
            gate_result.homography
        ),
        template_shape=(
            runtime.template.shape[:2]
        ),
        anomaly_roi=(
            runtime.anomaly_roi
        ),
    )

    state.valid_ratio = (
        valid_ratio
    )

    if (
        crop is None
        or valid_ratio
        < min_valid_ratio
    ):

        state.gate_status = (
            "ALIGN INVALID"
        )

        state.gate_pass = False

        runtime.temporal.gate_miss()

        state.alarm = (
            runtime.temporal.alarm
        )

        state.vote_count = (
            runtime.temporal.vote_count
        )

        state.history = list(
            runtime.temporal.history
        )

        return state

    state.crop = crop

    # --------------------------------------------------------
    # PatchCore
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )

    tensor = (
        runtime.preprocessor(
            image_rgb
        )
        .unsqueeze(0)
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    (
        scores,
        anomaly_maps,
    ) = runtime.patchcore.predict(
        tensor
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    state.patchcore_ms = (
        (
            time.perf_counter()
            - start
        )
        * 1000.0
    )

    score_array = (
        scores
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    state.score = float(
        score_array[0]
    )

    anomaly_array = (
        anomaly_maps
        .detach()
        .cpu()
        .numpy()
    )

    anomaly_map = np.squeeze(
        anomaly_array[0]
    )

    crop_h, crop_w = (
        crop.shape[:2]
    )

    state.anomaly_map = (
        restore_anomaly_map(
            anomaly_map,
            original_width=(
                crop_w
            ),
            original_height=(
                crop_h
            ),
            input_size=(
                runtime.patchcore
                .config
                .input_size
            ),
        )
    )

    # --------------------------------------------------------
    # Raw NG
    # --------------------------------------------------------

    state.raw_ng = (
        state.score
        >= runtime.threshold
    )

    state.alarm = (
        runtime.temporal.update(
            state.raw_ng
        )
    )

    state.vote_count = (
        runtime.temporal.vote_count
    )

    state.history = list(
        runtime.temporal.history
    )

    return state


# ============================================================
# Visualization helpers
# ============================================================

def put_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.65,
    color: tuple[int, int, int] = (
        230,
        230,
        230,
    ),
    thickness: int = 2,
) -> None:

    cv2.putText(
        image,
        text,
        (
            x,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fit_thumbnail(
    image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:

    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    src_h, src_w = (
        image.shape[:2]
    )

    scale = min(
        width / src_w,
        height / src_h,
    )

    new_w = max(
        1,
        int(
            src_w * scale
        ),
    )

    new_h = max(
        1,
        int(
            src_h * scale
        ),
    )

    resized = cv2.resize(
        image,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_CUBIC,
    )

    x = (
        width
        - new_w
    ) // 2

    y = (
        height
        - new_h
    ) // 2

    canvas[
        y:y + new_h,
        x:x + new_w,
    ] = resized

    return canvas


def make_heatmap(
    state: RoiState,
) -> np.ndarray | None:

    if (
        state.crop is None
        or state.anomaly_map
        is None
    ):
        return None

    anomaly = (
        state.anomaly_map
    )

    minimum = float(
        anomaly.min()
    )

    maximum = float(
        anomaly.max()
    )

    if maximum <= minimum:

        normalized = np.zeros(
            anomaly.shape,
            dtype=np.uint8,
        )

    else:

        normalized = (
            (
                anomaly
                - minimum
            )
            / (
                maximum
                - minimum
            )
            * 255
        ).astype(
            np.uint8
        )

    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET,
    )

    return cv2.addWeighted(
        state.crop,
        0.55,
        heatmap,
        0.45,
        0,
    )


def status_color(
    state: RoiState,
) -> tuple[int, int, int]:

    if state.alarm:
        return COLOR_RED

    if state.raw_ng:
        return COLOR_YELLOW

    if state.gate_pass:
        return COLOR_GREEN

    return COLOR_GRAY


# ============================================================
# GT Event
# ============================================================

def active_events(
    events: list[dict],
    timestamp: float,
) -> list[dict]:

    return [
        event
        for event in events
        if (
            float(
                event[
                    "start_sec"
                ]
            )
            <= timestamp
            <= float(
                event[
                    "end_sec"
                ]
            )
        )
    ]


# ============================================================
# Render
# ============================================================

def render_annotated_frame(
    *,
    frame: np.ndarray,
    states: dict[
        str,
        RoiState,
    ],
    runtimes: dict[
        str,
        RoiRuntime,
    ],
    timestamp: float,
    inference_fps: float,
    panel_width: int,
    events: list[dict],
) -> np.ndarray:

    display = frame.copy()

    frame_h, frame_w = (
        display.shape[:2]
    )

    # ========================================================
    # Search ROI + Polygon
    # ========================================================

    for roi_name in [
        "left",
        "right",
    ]:

        runtime = runtimes[
            roi_name
        ]

        state = states[
            roi_name
        ]

        color = status_color(
            state
        )

        sx, sy, sw, sh = (
            runtime.search_roi
        )

        cv2.rectangle(
            display,
            (
                sx,
                sy,
            ),
            (
                sx + sw,
                sy + sh,
            ),
            color,
            3,
            cv2.LINE_AA,
        )

        put_text(
            display,
            roi_name.upper(),
            sx,
            max(
                30,
                sy - 12,
            ),
            scale=0.8,
            color=color,
            thickness=2,
        )

        if (
            state.polygon_global
            is not None
        ):

            polygon = (
                np.round(
                    state.polygon_global
                )
                .astype(
                    np.int32
                )
                .reshape(
                    -1,
                    1,
                    2,
                )
            )

            cv2.polylines(
                display,
                [
                    polygon
                ],
                True,
                color,
                3,
                cv2.LINE_AA,
            )

    # ========================================================
    # Alarm banner
    # ========================================================

    alarm_rois = [
        name.upper()
        for name, state
        in states.items()
        if state.alarm
    ]

    if alarm_rois:

        banner_height = 82

        overlay = (
            display.copy()
        )

        cv2.rectangle(
            overlay,
            (
                0,
                0,
            ),
            (
                frame_w,
                banner_height,
            ),
            COLOR_RED,
            -1,
        )

        display = cv2.addWeighted(
            overlay,
            0.85,
            display,
            0.15,
            0,
        )

        put_text(
            display,
            (
                "LEAK ANOMALY DETECTED"
                " | ROI: "
                + ", ".join(
                    alarm_rois
                )
            ),
            35,
            55,
            scale=1.15,
            color=COLOR_WHITE,
            thickness=3,
        )

    # ========================================================
    # GT banner
    # ========================================================

    gt = active_events(
        events,
        timestamp,
    )

    if gt:

        y1 = (
            92
            if alarm_rois
            else 10
        )

        y2 = (
            y1 + 62
        )

        cv2.rectangle(
            display,
            (
                10,
                y1,
            ),
            (
                frame_w - 10,
                y2,
            ),
            COLOR_ORANGE,
            -1,
        )

        gt_text = " | ".join(
            (
                f"GT: "
                f"{event['roi'].upper()} "
                f"{event['defect_type'].upper()}"
            )
            for event in gt
        )

        put_text(
            display,
            gt_text,
            30,
            y1 + 42,
            scale=0.9,
            color=COLOR_WHITE,
            thickness=2,
        )

    # ========================================================
    # Right information panel
    # ========================================================

    panel = np.full(
        (
            frame_h,
            panel_width,
            3,
        ),
        COLOR_DARK,
        dtype=np.uint8,
    )

    put_text(
        panel,
        "HOSE LEAK INSPECTION",
        28,
        45,
        scale=0.85,
        color=COLOR_WHITE,
        thickness=2,
    )

    put_text(
        panel,
        (
            f"Time      : "
            f"{timestamp:7.2f} s"
        ),
        28,
        82,
        scale=0.62,
    )

    put_text(
        panel,
        (
            f"Inference : "
            f"{inference_fps:.1f} FPS"
        ),
        28,
        112,
        scale=0.62,
    )

    # --------------------------------------------------------
    # LEFT / RIGHT section
    # --------------------------------------------------------

    section_top = {
        "left": 160,
        "right": (
            frame_h // 2
            + 55
        ),
    }

    for roi_name in [
        "left",
        "right",
    ]:

        state = states[
            roi_name
        ]

        runtime = runtimes[
            roi_name
        ]

        y = section_top[
            roi_name
        ]

        color = status_color(
            state
        )

        # separator
        cv2.line(
            panel,
            (
                20,
                y - 25,
            ),
            (
                panel_width - 20,
                y - 25,
            ),
            (
                70,
                70,
                70,
            ),
            1,
        )

        put_text(
            panel,
            roi_name.upper(),
            28,
            y,
            scale=0.92,
            color=color,
            thickness=2,
        )

        gate_text = (
            "PASS"
            if state.gate_pass
            else "SKIP"
        )

        put_text(
            panel,
            (
                f"Gate       : "
                f"{gate_text} "
                f"({state.gate_status})"
            ),
            28,
            y + 42,
            scale=0.58,
            color=(
                COLOR_GREEN
                if state.gate_pass
                else COLOR_GRAY
            ),
        )

        put_text(
            panel,
            (
                f"Match      : "
                f"{state.good_matches} / "
                f"Inlier {state.inliers}"
            ),
            28,
            y + 76,
            scale=0.58,
        )

        if math.isfinite(
            state.area_ratio
        ):

            put_text(
                panel,
                (
                    f"Area/Rot   : "
                    f"{state.area_ratio:.3f}"
                    f" / "
                    f"{state.rotation_deg:.1f} deg"
                ),
                28,
                y + 110,
                scale=0.58,
            )

        if math.isfinite(
            state.score
        ):

            score_color = (
                COLOR_YELLOW
                if state.raw_ng
                else COLOR_GREEN
            )

            put_text(
                panel,
                (
                    f"Score      : "
                    f"{state.score:.3f}"
                    f" / "
                    f"{runtime.threshold:.3f}"
                ),
                28,
                y + 148,
                scale=0.65,
                color=score_color,
            )

        else:

            put_text(
                panel,
                (
                    "Score      : -- "
                    f"/ {runtime.threshold:.3f}"
                ),
                28,
                y + 148,
                scale=0.65,
                color=COLOR_GRAY,
            )

        raw_text = (
            "NG"
            if state.raw_ng
            else "OK"
        )

        put_text(
            panel,
            (
                f"Raw        : "
                f"{raw_text}"
            ),
            28,
            y + 184,
            scale=0.62,
            color=(
                COLOR_YELLOW
                if state.raw_ng
                else COLOR_GREEN
            ),
        )

        put_text(
            panel,
            (
                f"Temporal   : "
                f"{state.vote_count}"
                f" / "
                f"{runtime.temporal.window_size}"
            ),
            220,
            y + 184,
            scale=0.62,
        )

        alarm_text = (
            "ALARM"
            if state.alarm
            else "OFF"
        )

        put_text(
            panel,
            (
                f"Final      : "
                f"{alarm_text}"
            ),
            460,
            y + 184,
            scale=0.64,
            color=(
                COLOR_RED
                if state.alarm
                else COLOR_GREEN
            ),
            thickness=2,
        )

        # ----------------------------------------------------
        # Temporal history circles
        # ----------------------------------------------------

        history_y = (
            y + 220
        )

        put_text(
            panel,
            "History",
            28,
            history_y + 7,
            scale=0.50,
            color=COLOR_GRAY,
        )

        circle_x = 135

        for value in state.history:

            cv2.circle(
                panel,
                (
                    circle_x,
                    history_y,
                ),
                10,
                (
                    COLOR_RED
                    if value
                    else COLOR_GREEN
                ),
                -1,
                cv2.LINE_AA,
            )

            circle_x += 32

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        put_text(
            panel,
            (
                f"Gate "
                f"{state.gate_ms:.1f}ms"
                f" | PatchCore "
                f"{state.patchcore_ms:.1f}ms"
            ),
            360,
            history_y + 7,
            scale=0.46,
            color=COLOR_GRAY,
        )

        # ----------------------------------------------------
        # Crop + heatmap
        # ----------------------------------------------------

        thumbnail_y = (
            y + 255
        )

        thumbnail_h = 185
        thumbnail_w = (
            panel_width
            - 75
        ) // 2

        if (
            state.crop is not None
        ):

            roi_thumb = (
                fit_thumbnail(
                    state.crop,
                    thumbnail_w,
                    thumbnail_h,
                )
            )

            panel[
                thumbnail_y:
                thumbnail_y + thumbnail_h,
                25:
                25 + thumbnail_w,
            ] = roi_thumb

            put_text(
                panel,
                "Aligned ROI",
                28,
                thumbnail_y + thumbnail_h + 25,
                scale=0.48,
                color=COLOR_GRAY,
            )

        heatmap = (
            make_heatmap(
                state
            )
        )

        if heatmap is not None:

            heat_thumb = (
                fit_thumbnail(
                    heatmap,
                    thumbnail_w,
                    thumbnail_h,
                )
            )

            x2 = (
                50
                + thumbnail_w
            )

            panel[
                thumbnail_y:
                thumbnail_y + thumbnail_h,
                x2:
                x2 + thumbnail_w,
            ] = heat_thumb

            put_text(
                panel,
                "Anomaly Map",
                x2 + 3,
                thumbnail_y + thumbnail_h + 25,
                scale=0.48,
                color=COLOR_GRAY,
            )

    return np.hstack(
        [
            display,
            panel,
        ]
    )


# ============================================================
# Event loading
# ============================================================

def load_events(
    video_path: Path,
    explicit_path: Path | None,
) -> list[dict]:

    if explicit_path is not None:

        path = explicit_path

    else:

        path = (
            video_path
            .with_suffix(
                ".events.json"
            )
        )

    if not path.exists():
        return []

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return list(
        data.get(
            "events",
            []
        )
    )


# ============================================================
# Result row
# ============================================================

def state_to_columns(
    prefix: str,
    state: RoiState,
) -> dict:

    return {
        f"{prefix}_gate_status":
            state.gate_status,

        f"{prefix}_gate_pass":
            state.gate_pass,

        f"{prefix}_good_matches":
            state.good_matches,

        f"{prefix}_inliers":
            state.inliers,

        f"{prefix}_inlier_ratio":
            state.inlier_ratio,

        f"{prefix}_area_ratio":
            state.area_ratio,

        f"{prefix}_rotation_deg":
            state.rotation_deg,

        f"{prefix}_valid_ratio":
            state.valid_ratio,

        f"{prefix}_score":
            state.score,

        f"{prefix}_raw_ng":
            state.raw_ng,

        f"{prefix}_vote_count":
            state.vote_count,

        f"{prefix}_alarm":
            state.alarm,

        f"{prefix}_gate_ms":
            state.gate_ms,

        f"{prefix}_patchcore_ms":
            state.patchcore_ms,
    }


# ============================================================
# Statistics
# ============================================================

def score_statistics(
    values: pd.Series,
) -> dict:

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:

        return {
            "count": 0,
        }

    return {
        "count":
            int(
                len(values)
            ),

        "p50":
            float(
                values.quantile(
                    0.50
                )
            ),

        "p95":
            float(
                values.quantile(
                    0.95
                )
            ),

        "p99":
            float(
                values.quantile(
                    0.99
                )
            ),

        "max":
            float(
                values.max()
            ),

        "mean":
            float(
                values.mean()
            ),
    }


# ============================================================
# Alarm rising edges
# ============================================================

def count_alarm_events(
    values: pd.Series,
) -> int:

    current = (
        values
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    if len(current) == 0:
        return 0

    previous = np.concatenate(
        [
            np.array(
                [False]
            ),
            current[:-1],
        ]
    )

    return int(
        np.sum(
            current
            & ~previous
        )
    )


# ============================================================
# GT evaluation
# ============================================================

def evaluate_events(
    *,
    result: pd.DataFrame,
    events: list[dict],
    grace_sec: float,
) -> list[dict]:

    output = []

    for event in events:

        roi = str(
            event[
                "roi"
            ]
        ).lower()

        start = float(
            event[
                "start_sec"
            ]
        )

        end = float(
            event[
                "end_sec"
            ]
        )

        alarm_column = (
            f"{roi}_alarm"
        )

        candidates = result[
            (
                result[
                    "timestamp_sec"
                ]
                >= start
            )
            & (
                result[
                    "timestamp_sec"
                ]
                <= end
                + grace_sec
            )
            & (
                result[
                    alarm_column
                ]
                == True
            )
        ]

        if candidates.empty:

            detected = False

            first_alarm = None

            time_to_detect = None

        else:

            detected = True

            first_alarm = float(
                candidates.iloc[
                    0
                ][
                    "timestamp_sec"
                ]
            )

            time_to_detect = (
                first_alarm
                - start
            )

        output.append(
            {
                "id":
                    event.get(
                        "id"
                    ),

                "roi":
                    roi,

                "defect_type":
                    event.get(
                        "defect_type"
                    ),

                "start_sec":
                    start,

                "end_sec":
                    end,

                "detected":
                    detected,

                "first_alarm_sec":
                    first_alarm,

                "time_to_detect_sec":
                    time_to_detect,
            }
        )

    return output


# ============================================================
# Runtime builder
# ============================================================

def build_runtimes(
    config: dict,
    device: torch.device,
) -> dict[str, RoiRuntime]:

    paths = config[
        "paths"
    ]

    roi_config_path = resolve_path(
        paths[
            "roi_config"
        ]
    )

    normal_roi_path = resolve_path(
        paths[
            "normal_roi_config"
        ]
    )

    patchcore_config_path = (
        resolve_path(
            paths[
                "patchcore_config"
            ]
        )
    )

    with roi_config_path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        roi_config = yaml.safe_load(
            stream
        )

    with normal_roi_path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        normal_roi_config = (
            yaml.safe_load(
                stream
            )
        )

    with patchcore_config_path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        patchcore_config = (
            yaml.safe_load(
                stream
            )
        )

    temporal_config = config[
        "temporal"
    ]

    runtimes = {}

    for roi_name in [
        "left",
        "right",
    ]:

        template_path = (
            resolve_path(
                paths[
                    "templates"
                ][
                    roi_name
                ]
            )
        )

        template = imread_safe(
            template_path
        )

        if template is None:

            raise FileNotFoundError(
                template_path
            )

        search_roi = (
            find_search_roi(
                roi_name=roi_name,
                video_config=config,
                roi_config=roi_config,
            )
        )

        anomaly_roi = (
            load_anomaly_roi(
                normal_roi_config,
                roi_name,
            )
        )

        localizer = (
            GateLocalizer(
                template,
                config[
                    "matching"
                ],
            )
        )

        model_dir = resolve_path(
            patchcore_config[
                "rois"
            ][
                roi_name
            ][
                "model_dir"
            ]
        )

        model_path = (
            model_dir
            / "patchcore.pt"
        )

        patchcore = (
            PatchCoreModel.load(
                model_path,
                device,
            )
        )

        preprocessor = (
            AspectPadPreprocessor(
                patchcore
                .config
                .input_size
            )
        )

        runtimes[
            roi_name
        ] = RoiRuntime(
            name=roi_name,

            search_roi=(
                search_roi
            ),

            anomaly_roi=(
                anomaly_roi
            ),

            template=template,

            localizer=localizer,

            gate_config=(
                config[
                    "gate"
                ][
                    roi_name
                ]
            ),

            patchcore=patchcore,

            preprocessor=(
                preprocessor
            ),

            threshold=float(
                config[
                    "patchcore"
                ][
                    roi_name
                ][
                    "threshold"
                ]
            ),

            temporal=(
                TemporalFilter(
                    window_size=int(
                        temporal_config[
                            "window_size"
                        ]
                    ),
                    required_ng=int(
                        temporal_config[
                            "required_ng"
                        ]
                    ),
                    reset_after_gate_miss=int(
                        temporal_config[
                            "reset_after_gate_miss"
                        ]
                    ),
                )
            ),
        )

    return runtimes


# ============================================================
# Video inference
# ============================================================

def run_video(
    *,
    video_path: Path,
    output_root: Path,
    config: dict,
    runtimes: dict[
        str,
        RoiRuntime,
    ],
    device: torch.device,
    explicit_events: Path | None = None,
    save_annotated: bool = True,
) -> dict:

    # Temporal state reset
    for runtime in (
        runtimes.values()
    ):
        runtime.temporal.reset()

    events = load_events(
        video_path,
        explicit_events,
    )

    output_dir = (
        output_root
        / video_path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open: "
            f"{video_path}"
        )

    source_fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    target_fps = float(
        config[
            "runtime"
        ][
            "target_fps"
        ]
    )

    inference_period = (
        1.0
        / target_fps
    )

    min_valid_ratio = float(
        config[
            "runtime"
        ][
            "min_valid_ratio"
        ]
    )

    panel_width = int(
        config[
            "runtime"
        ][
            "panel_width"
        ]
    )

    grace_sec = float(
        config[
            "runtime"
        ][
            "event_grace_sec"
        ]
    )

    # --------------------------------------------------------
    # Writer
    # --------------------------------------------------------

    writer = None

    annotated_path = (
        output_dir
        / "annotated.mp4"
    )

    if save_annotated:

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        writer = cv2.VideoWriter(
            str(
                annotated_path
            ),
            fourcc,
            source_fps,
            (
                frame_width
                + panel_width,
                frame_height,
            ),
        )

        if not writer.isOpened():

            raise RuntimeError(
                "Could not create "
                "annotated video."
            )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    states = {
        "left":
            RoiState(
                roi="left"
            ),

        "right":
            RoiState(
                roi="right"
            ),
    }

    rows = []

    frame_index = 0

    next_inference_sec = 0.0

    # ========================================================
    # Frame loop
    # ========================================================

    while True:

        ok, frame = (
            capture.read()
        )

        if not ok:
            break

        timestamp = (
            frame_index
            / source_fps
        )

        # ----------------------------------------------------
        # 2 FPS inference
        # ----------------------------------------------------

        if (
            timestamp
            + 1e-9
            >= next_inference_sec
        ):

            inference_start = (
                time.perf_counter()
            )

            for roi_name in [
                "left",
                "right",
            ]:

                states[
                    roi_name
                ] = infer_roi(
                    frame=frame,
                    runtime=(
                        runtimes[
                            roi_name
                        ]
                    ),
                    device=device,
                    min_valid_ratio=(
                        min_valid_ratio
                    ),
                )

            total_ms = (
                (
                    time.perf_counter()
                    - inference_start
                )
                * 1000.0
            )

            row = {
                "frame_idx":
                    frame_index,

                "timestamp_sec":
                    timestamp,

                "total_inference_ms":
                    total_ms,
            }

            row.update(
                state_to_columns(
                    "left",
                    states[
                        "left"
                    ],
                )
            )

            row.update(
                state_to_columns(
                    "right",
                    states[
                        "right"
                    ],
                )
            )

            rows.append(
                row
            )

            while (
                next_inference_sec
                <= timestamp
            ):

                next_inference_sec += (
                    inference_period
                )

        # ----------------------------------------------------
        # Render every source frame
        # ----------------------------------------------------

        if writer is not None:

            annotated = (
                render_annotated_frame(
                    frame=frame,
                    states=states,
                    runtimes=runtimes,
                    timestamp=timestamp,
                    inference_fps=(
                        target_fps
                    ),
                    panel_width=(
                        panel_width
                    ),
                    events=events,
                )
            )

            writer.write(
                annotated
            )

        frame_index += 1

    # ========================================================
    # Finish
    # ========================================================

    capture.release()

    if writer is not None:
        writer.release()

    result = pd.DataFrame(
        rows
    )

    result_path = (
        output_dir
        / "result.csv"
    )

    result.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    duration_sec = (
        frame_index
        / source_fps
        if source_fps > 0
        else 0.0
    )

    duration_hours = (
        duration_sec
        / 3600.0
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "video":
            str(
                video_path
            ),

        "source_fps":
            source_fps,

        "target_inference_fps":
            target_fps,

        "source_frames":
            frame_index,

        "inference_frames":
            len(
                result
            ),

        "duration_sec":
            duration_sec,

        "rois":
            {},
    }

    for roi_name in [
        "left",
        "right",
    ]:

        gate_col = (
            f"{roi_name}_gate_pass"
        )

        score_col = (
            f"{roi_name}_score"
        )

        raw_col = (
            f"{roi_name}_raw_ng"
        )

        alarm_col = (
            f"{roi_name}_alarm"
        )

        alarm_events = (
            count_alarm_events(
                result[
                    alarm_col
                ]
            )
            if not result.empty
            else 0
        )

        summary[
            "rois"
        ][
            roi_name
        ] = {
            "gate_pass_count":
                int(
                    result[
                        gate_col
                    ].sum()
                ),

            "gate_pass_rate":
                float(
                    result[
                        gate_col
                    ].mean()
                )
                if not result.empty
                else 0.0,

            "scores":
                score_statistics(
                    result[
                        score_col
                    ]
                ),

            "raw_ng_count":
                int(
                    result[
                        raw_col
                    ].sum()
                ),

            "alarm_inference_count":
                int(
                    result[
                        alarm_col
                    ].sum()
                ),

            "alarm_events":
                alarm_events,

            "alarm_events_per_hour":
                (
                    float(
                        alarm_events
                        / duration_hours
                    )
                    if duration_hours > 0
                    else None
                ),
        }

    if not result.empty:

        summary[
            "latency"
        ] = {
            "mean_ms":
                float(
                    result[
                        "total_inference_ms"
                    ].mean()
                ),

            "p95_ms":
                float(
                    result[
                        "total_inference_ms"
                    ].quantile(
                        0.95
                    )
                ),

            "max_ms":
                float(
                    result[
                        "total_inference_ms"
                    ].max()
                ),
        }

    # ========================================================
    # Synthetic GT evaluation
    # ========================================================

    if events:

        event_results = (
            evaluate_events(
                result=result,
                events=events,
                grace_sec=(
                    grace_sec
                ),
            )
        )

        detected_count = sum(
            1
            for item in event_results
            if item[
                "detected"
            ]
        )

        summary[
            "ground_truth"
        ] = {
            "event_count":
                len(
                    event_results
                ),

            "detected_count":
                detected_count,

            "event_recall":
                (
                    detected_count
                    / len(
                        event_results
                    )
                    if event_results
                    else None
                ),

            "events":
                event_results,
        }

        pd.DataFrame(
            event_results
        ).to_csv(
            output_dir
            / "event_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # ========================================================
    # Save
    # ========================================================

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 75
    )

    print(
        f"VIDEO : {video_path.name}"
    )

    print(
        f"OUT   : {output_dir}"
    )

    print(
        f"TIME  : {duration_sec:.1f} sec"
    )

    print(
        f"INFER : {len(result)} frames"
    )

    for roi_name in [
        "left",
        "right",
    ]:

        info = (
            summary[
                "rois"
            ][
                roi_name
            ]
        )

        print(
            f"{roi_name.upper():5s}"
            f" | Gate "
            f"{info['gate_pass_rate']:.3f}"
            f" | Raw NG "
            f"{info['raw_ng_count']}"
            f" | Alarm Events "
            f"{info['alarm_events']}"
        )

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    source = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    source.add_argument(
        "--video",
        type=Path,
    )

    source.add_argument(
        "--input-dir",
        type=Path,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/video_inference.yaml"
        ),
    )

    parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help=(
            "Optional events.json. "
            "If omitted, <video>.events.json "
            "is detected automatically."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/video_inference"
        ),
    )

    parser.add_argument(
        "--no-annotated",
        action="store_true",
    )

    args = parser.parse_args()

    config_path = resolve_path(
        args.config
    )

    output_root = resolve_path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        config = yaml.safe_load(
            stream
        )

    device_name = str(
        config[
            "runtime"
        ].get(
            "device",
            "cpu",
        )
    )

    if (
        device_name == "auto"
    ):

        device_name = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        device_name
    )

    print(
        f"[DEVICE] {device}"
    )

    # --------------------------------------------------------
    # Model / template는 한 번만 load
    # --------------------------------------------------------

    runtimes = build_runtimes(
        config,
        device,
    )

    # --------------------------------------------------------
    # Source videos
    # --------------------------------------------------------

    if args.video is not None:

        videos = [
            resolve_path(
                args.video
            )
        ]

    else:

        input_dir = resolve_path(
            args.input_dir
        )

        videos = sorted(
            path
            for path in input_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in VIDEO_EXTENSIONS
            )
        )

    if not videos:

        raise RuntimeError(
            "No video files found."
        )

    summaries = {}

    for video_path in videos:

        # explicit --events는 단일 영상일 때만 적용
        explicit_events = None

        if (
            args.events is not None
            and len(videos) == 1
        ):

            explicit_events = (
                resolve_path(
                    args.events
                )
            )

        summary = run_video(
            video_path=video_path,
            output_root=(
                output_root
            ),
            config=config,
            runtimes=runtimes,
            device=device,
            explicit_events=(
                explicit_events
            ),
            save_annotated=(
                not args.no_annotated
            ),
        )

        summaries[
            video_path.name
        ] = summary

    (
        output_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summaries,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 75
    )

    print(
        "[DONE] Video inference completed"
    )

    print(
        f"Output: {output_root}"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()
