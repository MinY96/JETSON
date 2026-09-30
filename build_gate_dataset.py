from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import yaml


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
}

CSV_FIELDS = [
    "video_name",
    "video_path",
    "frame_idx",
    "timestamp_sec",
    "source_fps",
    "frame_width",
    "frame_height",

    "roi_name",
    "search_x",
    "search_y",
    "search_width",
    "search_height",
    "image_path",

    # API 결과
    "match_status",
    "http_status",
    "error_code",
    "error_message",

    # Matching metrics
    "good_matches",
    "inliers",
    "inlier_ratio",

    # Homography로 계산한 Geometry
    "projected_width",
    "projected_height",
    "projected_area",
    "area_ratio",
    "rotation_deg",

    "center_x_local",
    "center_y_local",
    "center_x_global",
    "center_y_global",

    # JSON 문자열
    "polygon_local",
    "polygon_global",
    "homography",

    "api_duration_ms",

    # 이후 수동 검증용
    "manual_label",
    "note",
]


@dataclass(frozen=True)
class RoiConfig:
    name: str
    template_path: Path
    search_roi: tuple[int, int, int, int]


@dataclass
class LocalizeResult:
    status: str

    http_status: int | None = None

    error_code: str = ""
    error_message: str = ""

    good_matches: int | None = None
    inliers: int | None = None
    inlier_ratio: float | None = None

    homography: list[list[float]] | None = None
    polygon: list[list[float]] | None = None

    duration_ms: float | None = None


class ImageProcessingApi:
    def __init__(
        self,
        base_url: str,
        timeout_sec: float,
        matching_params: dict[str, Any],
    ) -> None:
        if not base_url.endswith("/"):
            base_url += "/"

        self.client = httpx.Client(
            base_url=base_url,
            timeout=timeout_sec,
        )

        self.matching_params = matching_params

    def close(self) -> None:
        self.client.close()

    def health_check(self) -> None:
        response = self.client.get("health")
        response.raise_for_status()

    def localize(
        self,
        template_name: str,
        template_bytes: bytes,
        scene_name: str,
        scene_bytes: bytes,
    ) -> LocalizeResult:

        payload = {
            "params": self.matching_params,
            "image_inputs": [
                {
                    "input_name": "query_image",
                    "file_index": 0,
                },
                {
                    "input_name": "scene_image",
                    "file_index": 1,
                },
            ],
        }

        files = [
            (
                "files",
                (
                    template_name,
                    template_bytes,
                    "image/png",
                ),
            ),
            (
                "files",
                (
                    scene_name,
                    scene_bytes,
                    "image/jpeg",
                ),
            ),
        ]

        start = time.perf_counter()

        try:
            response = self.client.post(
                "operations/localize_planar_object/execute",
                data={
                    "payload": json.dumps(payload),
                },
                files=files,
            )
        except httpx.HTTPError as exc:
            duration_ms = (
                time.perf_counter() - start
            ) * 1000.0

            return LocalizeResult(
                status="http_error",
                error_code="http_error",
                error_message=str(exc),
                duration_ms=duration_ms,
            )

        duration_ms = (
            time.perf_counter() - start
        ) * 1000.0

        try:
            body = response.json()
        except Exception:
            return LocalizeResult(
                status="invalid_response",
                http_status=response.status_code,
                error_code="invalid_response",
                error_message=response.text[:500],
                duration_ms=duration_ms,
            )

        # 정상 Localize
        if response.is_success and body.get("success") is True:
            output = body.get("output", {})
            data = output.get("data", {})

            metrics = data.get("metrics", {})

            return LocalizeResult(
                status="matched",
                http_status=response.status_code,

                good_matches=_safe_int(
                    metrics.get("good_matches")
                ),
                inliers=_safe_int(
                    metrics.get("inliers")
                ),
                inlier_ratio=_safe_float(
                    metrics.get("inlier_ratio")
                ),

                homography=data.get("homography"),
                polygon=data.get("polygon"),

                duration_ms=duration_ms,
            )

        # insufficient_matches 등의 실패도 Dataset에 기록
        error = body.get("error", {}) or {}
        details = error.get("details", {}) or {}

        error_code = str(
            error.get("code", "api_error")
        )

        good_matches = _safe_int(
            details.get("matches")
        )

        return LocalizeResult(
            status=error_code,
            http_status=response.status_code,

            error_code=error_code,
            error_message=str(
                error.get("message", "")
            ),

            good_matches=good_matches,

            duration_ms=duration_ms,
        )


def _safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def _safe_int(
    value: Any,
) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_yaml(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid config file: {path}"
        )

    return config


def load_roi_configs(
    config: dict[str, Any],
) -> list[RoiConfig]:

    roi_section = config.get("rois")

    if not isinstance(roi_section, dict):
        raise ValueError(
            "'rois' section is missing in gate.yaml"
        )

    result: list[RoiConfig] = []

    for name, value in roi_section.items():

        if not isinstance(value, dict):
            raise ValueError(
                f"Invalid ROI config: {name}"
            )

        template_path = Path(
            value["template"]
        ).resolve()

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}"
            )

        search_roi = value.get("search_roi")

        if (
            not isinstance(search_roi, list)
            or len(search_roi) != 4
        ):
            raise ValueError(
                f"{name}.search_roi must be "
                "[x, y, width, height]"
            )

        x, y, w, h = map(
            int,
            search_roi,
        )

        if w <= 0 or h <= 0:
            raise ValueError(
                f"Invalid search ROI size: {name}"
            )

        result.append(
            RoiConfig(
                name=name,
                template_path=template_path,
                search_roi=(x, y, w, h),
            )
        )

    return result


def load_template_png(
    path: Path,
) -> tuple[np.ndarray, bytes]:

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Could not read template: {path}"
        )

    ok, encoded = cv2.imencode(
        ".png",
        image,
    )

    if not ok:
        raise RuntimeError(
            f"Could not encode template: {path}"
        )

    return image, encoded.tobytes()


def crop_search_roi(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:

    x, y, w, h = roi

    frame_h, frame_w = frame.shape[:2]

    if x < 0 or y < 0:
        raise ValueError(
            f"ROI start outside frame: {roi}"
        )

    if x + w > frame_w:
        raise ValueError(
            f"ROI width outside frame: "
            f"{roi}, frame={frame_w}x{frame_h}"
        )

    if y + h > frame_h:
        raise ValueError(
            f"ROI height outside frame: "
            f"{roi}, frame={frame_w}x{frame_h}"
        )

    return frame[
        y:y + h,
        x:x + w,
    ].copy()


def encode_jpeg(
    image: np.ndarray,
    quality: int,
) -> bytes:

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
        ],
    )

    if not ok:
        raise RuntimeError(
            "Failed to encode JPEG"
        )

    return encoded.tobytes()


def calculate_geometry(
    polygon: list[list[float]] | None,
    template_shape: tuple[int, int],
    search_roi: tuple[int, int, int, int],
) -> dict[str, Any]:

    empty = {
        "projected_width": None,
        "projected_height": None,
        "projected_area": None,
        "area_ratio": None,
        "rotation_deg": None,

        "center_x_local": None,
        "center_y_local": None,
        "center_x_global": None,
        "center_y_global": None,

        "polygon_local": "",
        "polygon_global": "",
    }

    if polygon is None:
        return empty

    try:
        points = np.asarray(
            polygon,
            dtype=np.float32,
        ).reshape(4, 2)
    except Exception:
        return empty

    if not np.isfinite(points).all():
        return empty

    # local polygon
    p0, p1, p2, p3 = points

    top_width = np.linalg.norm(
        p1 - p0
    )
    bottom_width = np.linalg.norm(
        p2 - p3
    )

    left_height = np.linalg.norm(
        p3 - p0
    )
    right_height = np.linalg.norm(
        p2 - p1
    )

    projected_width = float(
        (top_width + bottom_width) / 2.0
    )

    projected_height = float(
        (left_height + right_height) / 2.0
    )

    projected_area = float(
        abs(
            cv2.contourArea(
                points.reshape(-1, 1, 2)
            )
        )
    )

    template_h, template_w = template_shape

    template_area = float(
        max(
            1,
            (template_w - 1)
            * (template_h - 1),
        )
    )

    area_ratio = (
        projected_area
        / template_area
    )

    # Template top edge 기준 회전각
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])

    rotation_deg = math.degrees(
        math.atan2(dy, dx)
    )

    center = points.mean(axis=0)

    center_x_local = float(
        center[0]
    )
    center_y_local = float(
        center[1]
    )

    x, y, _, _ = search_roi

    global_points = points.copy()
    global_points[:, 0] += x
    global_points[:, 1] += y

    center_x_global = (
        center_x_local + x
    )
    center_y_global = (
        center_y_local + y
    )

    return {
        "projected_width":
            projected_width,

        "projected_height":
            projected_height,

        "projected_area":
            projected_area,

        "area_ratio":
            area_ratio,

        "rotation_deg":
            rotation_deg,

        "center_x_local":
            center_x_local,

        "center_y_local":
            center_y_local,

        "center_x_global":
            center_x_global,

        "center_y_global":
            center_y_global,

        "polygon_local":
            json.dumps(
                points.tolist(),
                ensure_ascii=False,
            ),

        "polygon_global":
            json.dumps(
                global_points.tolist(),
                ensure_ascii=False,
            ),
    }


def load_existing_keys(
    csv_path: Path,
) -> set[tuple[str, int]]:

    if not csv_path.exists():
        return set()

    keys: set[tuple[str, int]] = set()

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:

        reader = csv.DictReader(stream)

        for row in reader:
            try:
                keys.add(
                    (
                        row["video_name"],
                        int(row["frame_idx"]),
                    )
                )
            except Exception:
                continue

    return keys


def create_csv_writer(
    csv_path: Path,
) -> tuple[Any, csv.DictWriter]:

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = csv_path.exists()

    stream = csv_path.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    )

    writer = csv.DictWriter(
        stream,
        fieldnames=CSV_FIELDS,
    )

    if not exists:
        writer.writeheader()
        stream.flush()

    return stream, writer


def discover_videos(
    video_dir: Path,
) -> list[Path]:

    videos = [
        path
        for path in video_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in VIDEO_EXTENSIONS
        )
    ]

    return sorted(videos)


def create_frame_name(
    video_stem: str,
    frame_idx: int,
    timestamp_sec: float,
) -> str:

    timestamp_ms = int(
        round(timestamp_sec * 1000)
    )

    return (
        f"{video_stem}"
        f"_f{frame_idx:07d}"
        f"_t{timestamp_ms:09d}.jpg"
    )


def process_video(
    *,
    video_path: Path,
    output_dir: Path,
    roi_configs: list[RoiConfig],
    template_images: dict[str, np.ndarray],
    template_bytes: dict[str, bytes],
    api: ImageProcessingApi,
    writers: dict[str, csv.DictWriter],
    streams: dict[str, Any],
    existing_keys: dict[
        str,
        set[tuple[str, int]]
    ],
    target_fps: float,
    jpeg_quality: int,
    save_full_frame: bool,
) -> dict[str, int]:

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        print(
            f"[WARN] Cannot open video: "
            f"{video_path}"
        )

        return {}

    source_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
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

    if source_fps <= 0:
        source_fps = 30.0

    # 예:
    # 30 FPS → target 2 FPS
    # 15 frame에 한 번 처리
    frame_step = max(
        1,
        int(
            round(
                source_fps
                / target_fps
            )
        ),
    )

    print()
    print(
        f"[VIDEO] {video_path.name}"
    )
    print(
        f"  resolution : "
        f"{frame_width}x{frame_height}"
    )
    print(
        f"  source fps : "
        f"{source_fps:.3f}"
    )
    print(
        f"  frames     : "
        f"{frame_count}"
    )
    print(
        f"  sample step: "
        f"{frame_step}"
    )

    counters = {
        roi.name: 0
        for roi in roi_configs
    }

    frame_idx = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        timestamp_sec = (
            frame_idx / source_fps
        )

        frame_name = create_frame_name(
            video_path.stem,
            frame_idx,
            timestamp_sec,
        )

        # 필요할 경우 전체 Frame도 저장
        if save_full_frame:
            full_frame_dir = (
                output_dir
                / "frames"
                / video_path.stem
            )

            full_frame_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            cv2.imwrite(
                str(
                    full_frame_dir
                    / frame_name
                ),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    jpeg_quality,
                ],
            )

        for roi_config in roi_configs:

            key = (
                video_path.name,
                frame_idx,
            )

            # 재실행 시 기존 데이터 Skip
            if (
                key
                in existing_keys[
                    roi_config.name
                ]
            ):
                continue

            search_roi = crop_search_roi(
                frame,
                roi_config.search_roi,
            )

            roi_image_dir = (
                output_dir
                / roi_config.name
                / "images"
            )

            roi_image_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            image_path = (
                roi_image_dir
                / frame_name
            )

            # API에 보내는 것과 동일한
            # JPEG bytes를 저장
            scene_bytes = encode_jpeg(
                search_roi,
                jpeg_quality,
            )

            image_path.write_bytes(
                scene_bytes
            )

            result = api.localize(
                template_name=(
                    f"{roi_config.name}_template.png"
                ),
                template_bytes=(
                    template_bytes[
                        roi_config.name
                    ]
                ),
                scene_name=frame_name,
                scene_bytes=scene_bytes,
            )

            template_image = (
                template_images[
                    roi_config.name
                ]
            )

            geometry = calculate_geometry(
                polygon=result.polygon,
                template_shape=(
                    template_image.shape[:2]
                ),
                search_roi=(
                    roi_config.search_roi
                ),
            )

            x, y, w, h = (
                roi_config.search_roi
            )

            row = {
                "video_name":
                    video_path.name,

                "video_path":
                    str(video_path),

                "frame_idx":
                    frame_idx,

                "timestamp_sec":
                    round(
                        timestamp_sec,
                        6,
                    ),

                "source_fps":
                    source_fps,

                "frame_width":
                    frame_width,

                "frame_height":
                    frame_height,

                "roi_name":
                    roi_config.name,

                "search_x":
                    x,

                "search_y":
                    y,

                "search_width":
                    w,

                "search_height":
                    h,

                "image_path":
                    str(image_path),

                "match_status":
                    result.status,

                "http_status":
                    result.http_status,

                "error_code":
                    result.error_code,

                "error_message":
                    result.error_message,

                "good_matches":
                    result.good_matches,

                "inliers":
                    result.inliers,

                "inlier_ratio":
                    result.inlier_ratio,

                "projected_width":
                    geometry[
                        "projected_width"
                    ],

                "projected_height":
                    geometry[
                        "projected_height"
                    ],

                "projected_area":
                    geometry[
                        "projected_area"
                    ],

                "area_ratio":
                    geometry[
                        "area_ratio"
                    ],

                "rotation_deg":
                    geometry[
                        "rotation_deg"
                    ],

                "center_x_local":
                    geometry[
                        "center_x_local"
                    ],

                "center_y_local":
                    geometry[
                        "center_y_local"
                    ],

                "center_x_global":
                    geometry[
                        "center_x_global"
                    ],

                "center_y_global":
                    geometry[
                        "center_y_global"
                    ],

                "polygon_local":
                    geometry[
                        "polygon_local"
                    ],

                "polygon_global":
                    geometry[
                        "polygon_global"
                    ],

                "homography":
                    (
                        json.dumps(
                            result.homography,
                            ensure_ascii=False,
                        )
                        if result.homography
                        is not None
                        else ""
                    ),

                "api_duration_ms":
                    (
                        round(
                            result.duration_ms,
                            3,
                        )
                        if result.duration_ms
                        is not None
                        else None
                    ),

                # 이후 실제 영상 보고
                # visible / not_visible 입력
                "manual_label": "",

                "note": "",
            }

            writers[
                roi_config.name
            ].writerow(row)

            streams[
                roi_config.name
            ].flush()

            existing_keys[
                roi_config.name
            ].add(key)

            counters[
                roi_config.name
            ] += 1

        if (
            frame_idx
            // frame_step
        ) % 50 == 0:

            print(
                f"  frame={frame_idx:7d} "
                f"time={timestamp_sec:7.2f}s"
            )

        frame_idx += 1

    cap.release()

    return counters


def build_summary(
    output_dir: Path,
    roi_configs: list[RoiConfig],
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "rois": {},
    }

    for roi_config in roi_configs:

        csv_path = (
            output_dir
            / roi_config.name
            / "metadata.csv"
        )

        counts: dict[str, int] = {}

        total = 0

        if csv_path.exists():
            with csv_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as stream:

                reader = csv.DictReader(
                    stream
                )

                for row in reader:
                    total += 1

                    status = (
                        row.get(
                            "match_status"
                        )
                        or "unknown"
                    )

                    counts[status] = (
                        counts.get(
                            status,
                            0,
                        )
                        + 1
                    )

        result["rois"][
            roi_config.name
        ] = {
            "total": total,
            "status_counts": counts,
        }

    return result


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/gate.yaml"
        ),
    )

    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(
            "data/videos/train"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/gate_dataset/train"
        ),
    )

    args = parser.parse_args()

    config_path = (
        args.config.resolve()
    )

    video_dir = (
        args.videos_dir.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    if not config_path.exists():
        raise FileNotFoundError(
            config_path
        )

    if not video_dir.exists():
        raise FileNotFoundError(
            video_dir
        )

    config = load_yaml(
        config_path
    )

    roi_configs = (
        load_roi_configs(
            config
        )
    )

    api_config = (
        config.get(
            "api",
            {},
        )
    )

    sampling_config = (
        config.get(
            "sampling",
            {},
        )
    )

    matching_config = (
        config.get(
            "matching",
            {},
        )
    )

    base_url = api_config.get(
        "base_url",
        "http://127.0.0.1:8000/api/v1/",
    )

    timeout_sec = float(
        api_config.get(
            "timeout_sec",
            30.0,
        )
    )

    target_fps = float(
        sampling_config.get(
            "target_fps",
            2.0,
        )
    )

    jpeg_quality = int(
        sampling_config.get(
            "jpeg_quality",
            95,
        )
    )

    save_full_frame = bool(
        sampling_config.get(
            "save_full_frame",
            False,
        )
    )

    matching_params = {
        "algorithm":
            matching_config.get(
                "algorithm",
                "sift",
            ),

        "matcher":
            matching_config.get(
                "matcher",
                "flann",
            ),

        "max_features":
            int(
                matching_config.get(
                    "max_features",
                    3000,
                )
            ),

        "ratio_threshold":
            float(
                matching_config.get(
                    "ratio_threshold",
                    0.70,
                )
            ),

        "min_matches":
            int(
                matching_config.get(
                    "min_matches",
                    4,
                )
            ),

        "max_matches":
            int(
                matching_config.get(
                    "max_matches",
                    500,
                )
            ),

        "reprojection_threshold":
            float(
                matching_config.get(
                    "reprojection_threshold",
                    5.0,
                )
            ),
    }

    videos = discover_videos(
        video_dir
    )

    if not videos:
        raise RuntimeError(
            f"No videos found: {video_dir}"
        )

    print(
        f"[INFO] Found "
        f"{len(videos)} videos"
    )

    # Template는 한 번만 읽음
    template_images: dict[
        str,
        np.ndarray
    ] = {}

    template_bytes: dict[
        str,
        bytes
    ] = {}

    for roi_config in roi_configs:

        image, encoded = (
            load_template_png(
                roi_config.template_path
            )
        )

        template_images[
            roi_config.name
        ] = image

        template_bytes[
            roi_config.name
        ] = encoded

        print(
            f"[TEMPLATE] "
            f"{roi_config.name}: "
            f"{roi_config.template_path} "
            f"{image.shape[1]}x"
            f"{image.shape[0]}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    streams: dict[str, Any] = {}
    writers: dict[
        str,
        csv.DictWriter
    ] = {}

    existing_keys: dict[
        str,
        set[tuple[str, int]]
    ] = {}

    for roi_config in roi_configs:

        csv_path = (
            output_dir
            / roi_config.name
            / "metadata.csv"
        )

        existing_keys[
            roi_config.name
        ] = load_existing_keys(
            csv_path
        )

        stream, writer = (
            create_csv_writer(
                csv_path
            )
        )

        streams[
            roi_config.name
        ] = stream

        writers[
            roi_config.name
        ] = writer

    api = ImageProcessingApi(
        base_url=base_url,
        timeout_sec=timeout_sec,
        matching_params=matching_params,
    )

    try:
        print(
            "[INFO] Checking "
            "ImageProcessingAPI..."
        )

        api.health_check()

        print(
            "[INFO] API health OK"
        )

        for index, video_path in enumerate(
            videos,
            start=1,
        ):

            print(
                f"\n"
                f"=============================="
            )

            print(
                f"[{index}/{len(videos)}] "
                f"{video_path.name}"
            )

            process_video(
                video_path=video_path,
                output_dir=output_dir,

                roi_configs=roi_configs,

                template_images=(
                    template_images
                ),

                template_bytes=(
                    template_bytes
                ),

                api=api,

                writers=writers,
                streams=streams,

                existing_keys=(
                    existing_keys
                ),

                target_fps=target_fps,

                jpeg_quality=(
                    jpeg_quality
                ),

                save_full_frame=(
                    save_full_frame
                ),
            )

    finally:
        api.close()

        for stream in (
            streams.values()
        ):
            stream.close()

    summary = build_summary(
        output_dir,
        roi_configs,
    )

    summary.update(
        {
            "video_dir":
                str(video_dir),

            "video_count":
                len(videos),

            "target_fps":
                target_fps,

            "matching_params":
                matching_params,
        }
    )

    summary_path = (
        output_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=============================="
    )
    print(
        "[DONE] Gate dataset created"
    )

    for name, info in (
        summary["rois"].items()
    ):
        print(
            f"  {name}: "
            f"{info['total']} images "
            f"{info['status_counts']}"
        )

    print(
        f"\nSummary: {summary_path}"
    )


if __name__ == "__main__":
    main()
