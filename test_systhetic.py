from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import httpx
import numpy as np


# ============================================================
# Settings
# ============================================================

BASE_URL = "http://127.0.0.1:8000/api/v1"

IMAGE_PATH = Path(r"images\normal.png")

OUTPUT_DIR = Path(r"output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_INPUT_SIZE = 512

# ROI 선택용 Preview 최대 크기
PREVIEW_MAX_WIDTH = 1200
PREVIEW_MAX_HEIGHT = 800


# ============================================================
# ROI Selector
# ============================================================

def select_roi_scaled(
    image: np.ndarray,
) -> tuple[int, int, int, int]:

    h, w = image.shape[:2]

    scale = min(
        PREVIEW_MAX_WIDTH / w,
        PREVIEW_MAX_HEIGHT / h,
        1.0,
    )

    preview = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    x, y, rw, rh = cv2.selectROI(
        "Select Hose ROI",
        preview,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyAllWindows()

    if rw == 0 or rh == 0:
        raise RuntimeError("ROI selection cancelled.")

    x = int(round(x / scale))
    y = int(round(y / scale))
    rw = int(round(rw / scale))
    rh = int(round(rh / scale))

    return x, y, rw, rh


# ============================================================
# Image Encoding
# ============================================================

def encode_png(
    image: np.ndarray,
) -> bytes:

    ok, encoded = cv2.imencode(
        ".png",
        image,
    )

    if not ok:
        raise RuntimeError(
            "Failed to encode image."
        )

    return encoded.tobytes()


# ============================================================
# Synthetic API
# ============================================================

def generate_synthetic(
    image: np.ndarray,
    payload: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:

    image_bytes = encode_png(image)

    response = httpx.post(
        f"{BASE_URL}/synthetic/generate",
        params={
            "response_format": "zip",
        },
        data={
            "payload": json.dumps(payload),
        },
        files={
            "source_image": (
                "source.png",
                image_bytes,
                "image/png",
            ),
        },
        timeout=120.0,
    )

    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()

    with zipfile.ZipFile(
        io.BytesIO(response.content)
    ) as archive:

        manifest = json.loads(
            archive.read(
                "manifest.json"
            )
        )

        result_image = cv2.imdecode(
            np.frombuffer(
                archive.read(
                    "candidates/00/image.png"
                ),
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )

        result_mask = cv2.imdecode(
            np.frombuffer(
                archive.read(
                    "candidates/00/mask.png"
                ),
                dtype=np.uint8,
            ),
            cv2.IMREAD_GRAYSCALE,
        )

        difference = cv2.imdecode(
            np.frombuffer(
                archive.read(
                    "candidates/00/difference.png"
                ),
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )

    return (
        result_image,
        result_mask,
        difference,
        manifest,
    )


# ============================================================
# Load CCTV Frame
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        IMAGE_PATH
    )

original = image.copy()


# ============================================================
# Select Hose Joint ROI
# ============================================================

x, y, w, h = select_roi_scaled(
    original
)

roi_original = original[
    y:y + h,
    x:x + w,
].copy()

cv2.imwrite(
    str(
        OUTPUT_DIR /
        "01_roi_original.png"
    ),
    roi_original,
)


# ============================================================
# Resize ROI -> 512x512
# ============================================================

roi = cv2.resize(
    roi_original,
    (
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
    ),
    interpolation=cv2.INTER_AREA,
)


# ============================================================
# STEP 1
# Small Hose Tear
# ============================================================

tear_payload = {

    "method": "procedural",

    "defect_type": "hose_small_tear",

    "severity": 0.45,

    "count": 1,

    "seed": 100,

    "mask": {

        "type": "crack",

        # ROI에서 찢어진 부분의 중심
        "center_x": 0.52,
        "center_y": 0.48,

        # crack 길이
        "width_ratio": 0.07,

        # crack 흔들림 정도
        "height_ratio": 0.015,

        # 호스 방향에 맞춰 조정
        "rotation_deg": 25,

        "feather_px": 1,

        "parameters": {
            "segments": 5,
            "thickness": 2,
            "branches": False,
        },
    },

    "parameters": {

        "pattern": "crack",

        # BGR
        "color": [
            20,
            20,
            20,
        ],

        "opacity": 0.65,
    },
}


(
    roi_tear,
    tear_mask,
    tear_diff,
    tear_manifest,
) = generate_synthetic(
    roi,
    tear_payload,
)


cv2.imwrite(
    str(
        OUTPUT_DIR /
        "02_roi_tear.png"
    ),
    roi_tear,
)

cv2.imwrite(
    str(
        OUTPUT_DIR /
        "02_tear_mask.png"
    ),
    tear_mask,
)

cv2.imwrite(
    str(
        OUTPUT_DIR /
        "02_tear_difference.png"
    ),
    tear_diff,
)


# ============================================================
# STEP 2
# Liquid Pooling
# ============================================================

pool_payload = {

    "method": "procedural",

    "defect_type": "liquid_pooling",

    "severity": 0.50,

    "count": 1,

    "seed": 200,

    "mask": {

        "type": "random_blob",

        # crack보다 조금 아래쪽
        "center_x": 0.53,
        "center_y": 0.53,

        "width_ratio": 0.13,
        "height_ratio": 0.075,

        "feather_px": 5,

        "parameters": {
            "points": 14,
            "irregularity": 0.30,
            "smooth": 9,
        },
    },

    "parameters": {

        "pattern": "pooling",

        # 살짝 밝은 회색/투명 액체 느낌
        # OpenCV 기준 BGR
        "color": [
            185,
            190,
            195,
        ],

        "opacity": 0.23,
    },
}


(
    roi_pool,
    pool_mask,
    pool_diff,
    pool_manifest,
) = generate_synthetic(
    roi_tear,
    pool_payload,
)


cv2.imwrite(
    str(
        OUTPUT_DIR /
        "03_roi_pooling.png"
    ),
    roi_pool,
)

cv2.imwrite(
    str(
        OUTPUT_DIR /
        "03_pool_mask.png"
    ),
    pool_mask,
)


# ============================================================
# STEP 3
# Small Droplet
# ============================================================

droplet_payload = {

    "method": "procedural",

    "defect_type": "liquid_droplet",

    "severity": 0.40,

    "count": 1,

    "seed": 300,

    "mask": {

        "type": "ellipse",

        "center_x": 0.54,
        "center_y": 0.61,

        "width_ratio": 0.025,
        "height_ratio": 0.05,

        "rotation_deg": 0,

        "feather_px": 3,
    },

    "parameters": {

        "pattern": "droplet",

        "color": [
            195,
            200,
            205,
        ],

        "opacity": 0.32,
    },
}


(
    roi_final,
    droplet_mask,
    droplet_diff,
    droplet_manifest,
) = generate_synthetic(
    roi_pool,
    droplet_payload,
)


cv2.imwrite(
    str(
        OUTPUT_DIR /
        "04_roi_final.png"
    ),
    roi_final,
)


# ============================================================
# Resize back to original ROI size
# ============================================================

roi_final_original_size = cv2.resize(
    roi_final,
    (
        w,
        h,
    ),
    interpolation=cv2.INTER_LINEAR,
)


# ============================================================
# Composite back to 1920x1080
# ============================================================

result = original.copy()

result[
    y:y + h,
    x:x + w,
] = roi_final_original_size


cv2.imwrite(
    str(
        OUTPUT_DIR /
        "05_full_synthetic.png"
    ),
    result,
)


print()
print(
    "Synthetic image saved:"
)
print(
    OUTPUT_DIR /
    "05_full_synthetic.png"
)
