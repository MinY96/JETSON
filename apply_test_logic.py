from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


# -----------------------------------------------------------------------------
# Path / API configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "template" / "hose"
INPUT_DIR = PROJECT_ROOT / "data" / "frames" / "test_case_1"
RESULT_DIR = PROJECT_ROOT / "result" / "test_case_1"

API_BASE_URL = os.getenv(
    "IMAGE_PROCESSING_API_URL",
    "http://127.0.0.1:8000/api/v1",
).rstrip("/")

BOTTOM_TEMPLATE = TEMPLATE_DIR / "bottom.png"
TOP_TEMPLATE = TEMPLATE_DIR / "top.png"

SUPPORTED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

REQUEST_TIMEOUT = 120

# Backend localize_planar_object parameters.
LOCALIZE_PARAMS = {
    "algorithm": "sift",
    "matcher": "flann",
    "max_features": 3000,
    "ratio_threshold": 0.70,
    "min_matches": 10,
    "max_matches": 500,
    "reprojection_threshold": 5.0,
}

# Shape verification in this project.
# The backend estimates the homography; this layer rejects geometrically implausible
# matches for the current use case where the target size should stay almost the same.
MIN_INLIER_RATIO = 0.35
MIN_SCALE = 0.75
MAX_SCALE = 1.30
MAX_SCALE_ANISOTROPY = 0.25

# Secondary texture verification.
# HOG is calculated by the backend API after rectifying the detected region to the
# template coordinate system. This is a texture/gradient similarity, not a physical
# material classifier. Tune this value after observing real samples.
ENABLE_TEXTURE_CHECK = True
MIN_TEXTURE_SIMILARITY = 0.55
HOG_PARAMS = {
    "width": 128,
    "height": 128,
    "cell_size": 8,
    "block_cells": 2,
    "bins": 9,
}

# Annotation colors (BGR)
TEMPLATE_STYLES = {
    "bottom": {"color": (0, 0, 255)},
    "top": {"color": (0, 255, 0)},
}

NO_MATCH_ERROR_CODES = {
    "insufficient_features",
    "insufficient_matches",
    "homography_estimation_failed",
}


@dataclass(slots=True)
class TemplateInfo:
    name: str
    path: Path
    content: bytes
    media_type: str
    width: int
    height: int
    hog_descriptor: np.ndarray | None = None


@dataclass(slots=True)
class MatchResult:
    template: str
    found: bool
    reason: str
    polygon: np.ndarray | None = None
    homography: np.ndarray | None = None
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    scale_x: float | None = None
    scale_y: float | None = None
    texture_similarity: float | None = None


# -----------------------------------------------------------------------------
# Basic image / HTTP helpers
# -----------------------------------------------------------------------------
def imread_unicode(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        suffix = ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image as PNG")
    return encoded.tobytes()


def media_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


def response_json(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Backend returned a non-JSON response: HTTP {response.status_code}\n"
            f"{response.text[:500]}"
        ) from exc


def check_backend(session: requests.Session) -> None:
    url = f"{API_BASE_URL}/health"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Image Processing API is not reachable.\n"
            f"Expected health endpoint: {url}\n"
            "Start the backend first, for example:\n"
            "  python -m uvicorn src.main:app --host 0.0.0.0 --port 8000"
        ) from exc


# -----------------------------------------------------------------------------
# 1) Select one source image -> select bottom/top ROI -> crop by backend Workflow
# -----------------------------------------------------------------------------
def choose_template_source_image() -> Path:
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filename = filedialog.askopenfilename(
            title="Select source image for hose templates",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if filename:
            return Path(filename)
    except Exception:
        pass

    value = input("Template source image path: ").strip().strip('"')
    if not value:
        raise RuntimeError("Template source image was not selected")
    return Path(value)


def select_two_rois(image: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
    selections: dict[str, tuple[int, int, int, int]] = {}

    try:
        for name in ("bottom", "top"):
            window_name = f"Select {name.upper()} ROI - ENTER/SPACE confirm, C cancel"
            roi = cv2.selectROI(
                window_name,
                image,
                showCrosshair=True,
                fromCenter=False,
            )
            cv2.destroyWindow(window_name)

            x, y, width, height = map(int, roi)
            if width <= 0 or height <= 0:
                raise RuntimeError(f"{name} ROI selection was cancelled")
            selections[name] = (x, y, width, height)
    finally:
        cv2.destroyAllWindows()

    return selections


def extract_templates_via_backend(
    session: requests.Session,
    source_path: Path,
    rois: dict[str, tuple[int, int, int, int]],
) -> None:
    source_bytes = source_path.read_bytes()

    nodes = []
    outputs: dict[str, dict[str, str]] = {}

    for name in ("bottom", "top"):
        x, y, width, height = rois[name]
        node_id = f"{name}_roi"
        nodes.append(
            {
                "id": node_id,
                "node_type": "roi_crop",
                "inputs": {
                    "image": {
                        "type": "graph_input",
                        "input_name": "image",
                    }
                },
                "params": {
                    "coordinate_mode": "pixels",
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "clamp": False,
                },
            }
        )
        outputs[name] = {
            "type": "node_output",
            "node_id": node_id,
            "output_name": "image",
        }

    payload = {
        "graph": {
            "name": "extract_hose_templates",
            "display_name": "Extract Hose Templates",
            "version": "1.0.0",
            "inputs": [
                {
                    "name": "image",
                    "kind": "image",
                    "required": True,
                }
            ],
            "nodes": nodes,
            "outputs": outputs,
        },
        "image_inputs": [
            {
                "input_name": "image",
                "file_index": 0,
            }
        ],
        "retain_intermediates": False,
        "analyze_intermediates": False,
    }

    response = session.post(
        f"{API_BASE_URL}/workflow/execute",
        data={"payload": json.dumps(payload)},
        files=[
            (
                "files",
                (
                    source_path.name,
                    source_bytes,
                    media_type_for(source_path),
                ),
            )
        ],
        timeout=REQUEST_TIMEOUT,
    )
    body = response_json(response)

    if not response.ok or not body.get("success", False):
        raise RuntimeError(
            "ROI crop workflow failed:\n"
            + json.dumps(body.get("error", body), indent=2, ensure_ascii=False)
        )

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    path_by_name = {
        "bottom": BOTTOM_TEMPLATE,
        "top": TOP_TEMPLATE,
    }

    for name, output_path in path_by_name.items():
        image_info = body["output"]["images"][name]
        raw = base64.b64decode(image_info["data"])
        output_path.write_bytes(raw)
        print(f"[template] {name}: {output_path}")


# -----------------------------------------------------------------------------
# Backend operations used during matching
# -----------------------------------------------------------------------------
def execute_single_image_operation(
    session: requests.Session,
    operation: str,
    image_bytes: bytes,
    filename: str,
    media_type: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "params": params,
        "image_inputs": [
            {
                "input_name": "image",
                "file_index": 0,
            }
        ],
    }
    response = session.post(
        f"{API_BASE_URL}/operations/{operation}/execute",
        data={"payload": json.dumps(payload)},
        files=[("files", (filename, image_bytes, media_type))],
        timeout=REQUEST_TIMEOUT,
    )
    body = response_json(response)
    if not response.ok or not body.get("success", False):
        raise RuntimeError(
            f"Backend operation failed: {operation}\n"
            + json.dumps(body.get("error", body), indent=2, ensure_ascii=False)
        )
    return body


def get_hog_descriptor(
    session: requests.Session,
    image_bytes: bytes,
    filename: str,
    media_type: str = "image/png",
) -> np.ndarray:
    body = execute_single_image_operation(
        session=session,
        operation="hog_descriptor",
        image_bytes=image_bytes,
        filename=filename,
        media_type=media_type,
        params=HOG_PARAMS,
    )
    descriptor = np.asarray(
        body["output"]["data"]["descriptor"],
        dtype=np.float32,
    ).reshape(-1)
    return descriptor


def localize_template(
    session: requests.Session,
    template: TemplateInfo,
    scene_path: Path,
    scene_bytes: bytes,
) -> MatchResult:
    payload = {
        "params": LOCALIZE_PARAMS,
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

    response = session.post(
        f"{API_BASE_URL}/operations/localize_planar_object/execute",
        data={"payload": json.dumps(payload)},
        files=[
            (
                "files",
                (
                    template.path.name,
                    template.content,
                    template.media_type,
                ),
            ),
            (
                "files",
                (
                    scene_path.name,
                    scene_bytes,
                    media_type_for(scene_path),
                ),
            ),
        ],
        timeout=REQUEST_TIMEOUT,
    )
    body = response_json(response)

    if not response.ok or not body.get("success", False):
        error = body.get("error") or {}
        code = str(error.get("code", "unknown_error"))
        if code in NO_MATCH_ERROR_CODES:
            return MatchResult(
                template=template.name,
                found=False,
                reason=code,
            )
        raise RuntimeError(
            f"Localization failed for {template.name} / {scene_path.name}:\n"
            + json.dumps(error or body, indent=2, ensure_ascii=False)
        )

    data = body["output"]["data"]
    metrics = data["metrics"]
    polygon = np.asarray(data["polygon"], dtype=np.float32).reshape(4, 2)
    homography = np.asarray(data["homography"], dtype=np.float64).reshape(3, 3)

    result = MatchResult(
        template=template.name,
        found=True,
        reason="localized",
        polygon=polygon,
        homography=homography,
        good_matches=int(metrics["good_matches"]),
        inliers=int(metrics["inliers"]),
        inlier_ratio=float(metrics["inlier_ratio"]),
    )
    return result


# -----------------------------------------------------------------------------
# Project-side verification: geometry first, texture second
# -----------------------------------------------------------------------------
def verify_geometry(result: MatchResult, template: TemplateInfo) -> MatchResult:
    if not result.found or result.polygon is None:
        return result

    if result.inlier_ratio < MIN_INLIER_RATIO:
        result.found = False
        result.reason = f"low_inlier_ratio:{result.inlier_ratio:.3f}"
        return result

    p = result.polygon.astype(np.float32)
    width_top = float(np.linalg.norm(p[1] - p[0]))
    width_bottom = float(np.linalg.norm(p[2] - p[3]))
    height_right = float(np.linalg.norm(p[2] - p[1]))
    height_left = float(np.linalg.norm(p[3] - p[0]))

    detected_width = 0.5 * (width_top + width_bottom)
    detected_height = 0.5 * (height_right + height_left)

    scale_x = detected_width / max(1.0, float(template.width))
    scale_y = detected_height / max(1.0, float(template.height))
    result.scale_x = scale_x
    result.scale_y = scale_y

    if not (MIN_SCALE <= scale_x <= MAX_SCALE and MIN_SCALE <= scale_y <= MAX_SCALE):
        result.found = False
        result.reason = f"scale_out_of_range:sx={scale_x:.3f},sy={scale_y:.3f}"
        return result

    anisotropy = abs(scale_x - scale_y) / max(scale_x, scale_y, 1e-6)
    if anisotropy > MAX_SCALE_ANISOTROPY:
        result.found = False
        result.reason = f"shape_distortion:{anisotropy:.3f}"
        return result

    area = abs(float(cv2.contourArea(p.reshape(-1, 1, 2))))
    if area <= 1.0:
        result.found = False
        result.reason = "degenerate_polygon"
        return result

    result.reason = "shape_ok"
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def verify_texture(
    session: requests.Session,
    result: MatchResult,
    template: TemplateInfo,
    scene_image: np.ndarray,
) -> MatchResult:
    if not ENABLE_TEXTURE_CHECK or not result.found:
        return result
    if result.homography is None or template.hog_descriptor is None:
        return result

    try:
        inverse_h = np.linalg.inv(result.homography)
    except np.linalg.LinAlgError:
        result.found = False
        result.reason = "singular_homography"
        return result

    # H maps template -> scene, therefore inv(H) rectifies scene -> template.
    rectified = cv2.warpPerspective(
        scene_image,
        inverse_h,
        (template.width, template.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    patch_descriptor = get_hog_descriptor(
        session=session,
        image_bytes=encode_png(rectified),
        filename=f"{template.name}_matched_patch.png",
        media_type="image/png",
    )
    similarity = cosine_similarity(template.hog_descriptor, patch_descriptor)
    result.texture_similarity = similarity

    if similarity < MIN_TEXTURE_SIMILARITY:
        result.found = False
        result.reason = f"low_texture_similarity:{similarity:.3f}"
        return result

    result.reason = "matched"
    return result


# -----------------------------------------------------------------------------
# Drawing / batch processing
# -----------------------------------------------------------------------------
def draw_match(image: np.ndarray, result: MatchResult) -> None:
    if not result.found or result.polygon is None:
        return

    color = TEMPLATE_STYLES[result.template]["color"]
    polygon_i = np.round(result.polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polygon_i], True, color, 3, cv2.LINE_AA)

    x = int(np.min(result.polygon[:, 0]))
    y = int(np.min(result.polygon[:, 1]))
    y = max(25, y - 8)

    label = (
        f"{result.template} "
        f"m={result.good_matches} "
        f"inlier={result.inlier_ratio:.2f}"
    )
    if result.texture_similarity is not None:
        label += f" tex={result.texture_similarity:.2f}"

    cv2.putText(
        image,
        label,
        (max(0, x), y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def load_template(
    session: requests.Session,
    name: str,
    path: Path,
) -> TemplateInfo:
    if not path.exists():
        raise FileNotFoundError(f"Template does not exist: {path}")

    image = imread_unicode(path)
    info = TemplateInfo(
        name=name,
        path=path,
        content=path.read_bytes(),
        media_type=media_type_for(path),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
    )

    if ENABLE_TEXTURE_CHECK:
        info.hog_descriptor = get_hog_descriptor(
            session=session,
            image_bytes=info.content,
            filename=path.name,
            media_type=info.media_type,
        )

    return info


def iter_input_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def process_images(session: requests.Session) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    templates = [
        load_template(session, "bottom", BOTTOM_TEMPLATE),
        load_template(session, "top", TOP_TEMPLATE),
    ]
    image_paths = iter_input_images(INPUT_DIR)
    if not image_paths:
        print(f"No images found: {INPUT_DIR}")
        return

    summary_rows: list[dict[str, Any]] = []

    for index, scene_path in enumerate(image_paths, start=1):
        print(f"[{index:04d}/{len(image_paths):04d}] {scene_path.name}")

        # One image at a time: bounded client memory usage.
        scene_image = imread_unicode(scene_path)
        scene_bytes = scene_path.read_bytes()
        annotated = scene_image.copy()

        for template in templates:
            result = localize_template(
                session=session,
                template=template,
                scene_path=scene_path,
                scene_bytes=scene_bytes,
            )
            result = verify_geometry(result, template)
            result = verify_texture(session, result, template, scene_image)
            draw_match(annotated, result)

            summary_rows.append(
                {
                    "image": scene_path.name,
                    "template": template.name,
                    "found": result.found,
                    "reason": result.reason,
                    "good_matches": result.good_matches,
                    "inliers": result.inliers,
                    "inlier_ratio": result.inlier_ratio,
                    "scale_x": result.scale_x,
                    "scale_y": result.scale_y,
                    "texture_similarity": result.texture_similarity,
                }
            )

            status = "FOUND" if result.found else "NOT_FOUND"
            texture_text = (
                f", texture={result.texture_similarity:.3f}"
                if result.texture_similarity is not None
                else ""
            )
            print(
                f"    {template.name:<6} {status:<9} "
                f"matches={result.good_matches}, "
                f"inlier={result.inlier_ratio:.3f}{texture_text} "
                f"({result.reason})"
            )

        # Save every input image, including images where neither template was found.
        output_path = RESULT_DIR / scene_path.name
        imwrite_unicode(output_path, annotated)

    summary_path = RESULT_DIR / "_match_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "template",
                "found",
                "reason",
                "good_matches",
                "inliers",
                "inlier_ratio",
                "scale_x",
                "scale_y",
                "texture_similarity",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nResult images : {RESULT_DIR}")
    print(f"Summary CSV   : {summary_path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract two hose templates and apply backend localization to a frame folder."
    )
    parser.add_argument(
        "--select-template",
        action="store_true",
        help="Select a new source image and recreate bottom/top templates.",
    )
    parser.add_argument(
        "--template-source",
        type=Path,
        default=None,
        help="Source image used with OpenCV ROI selector. Implies --select-template.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        check_backend(session)
        print(f"Backend OK: {API_BASE_URL}")

        templates_missing = not BOTTOM_TEMPLATE.exists() or not TOP_TEMPLATE.exists()
        should_extract = args.select_template or args.template_source is not None or templates_missing

        if should_extract:
            source_path = args.template_source or choose_template_source_image()
            if not source_path.exists():
                raise FileNotFoundError(f"Template source image does not exist: {source_path}")

            source_image = imread_unicode(source_path)
            print("Select BOTTOM ROI first, then TOP ROI.")
            rois = select_two_rois(source_image)
            extract_templates_via_backend(session, source_path, rois)
        else:
            print(f"Using existing template: {BOTTOM_TEMPLATE}")
            print(f"Using existing template: {TOP_TEMPLATE}")

        process_images(session)


if __name__ == "__main__":
    main()
