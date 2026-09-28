from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "http://127.0.0.1:8000/api/v1"

IMAGE_PATH = Path(
    r"images\normal_2560x1440.png"
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# 현재 구조 기준
NUM_MAJOR_ROIS = 2
HOSES_PER_MAJOR_ROI = 2

DEVICE = "GPU"

# OpenVINO SD1.5
INPUT_SIZE = 512

# 2560x1440 화면 preview
PREVIEW_MAX_WIDTH = 1400
PREVIEW_MAX_HEIGHT = 850

# 큰 ROI 선택 후 주변 context 추가
MAJOR_ROI_PADDING = 0.10

# 개별 Hose ROI
HOSE_ROI_PADDING = 0.15

# API timeout
API_TIMEOUT_SEC = 600.0


# ============================================================
# DIFFUSION PARAMETERS
# ============================================================

DIFFUSION_CONFIGS = {

    # --------------------------------------------------------
    # 1. 볼트 / 너트 연결부 누수
    # --------------------------------------------------------
    "connection_leak": {
        "defect_type": "connection_leak",

        "prompt": (
            "close-up of an industrial flexible hose "
            "connected to a metal bolt fitting, "
            "a localized leak at the hose-to-metal fitting "
            "connection, "
            "clear water leaking directly from the joint "
            "with a visible thin pressurized stream, "
            "small transparent droplets and glossy wet "
            "reflections around the metal fitting, "
            "the hose and connector remain intact, "
            "realistic industrial CCTV image, "
            "same hose, same connector, same equipment, "
            "same lighting and same camera perspective, "
            "localized defect only, photorealistic"
        ),

        "negative_prompt": (
            "extra hoses, extra connectors, extra pipes, "
            "different equipment, changed background, "
            "large rupture, exploded hose, "
            "completely detached hose, severed hose, "
            "heavy flooding, huge splash, "
            "water covering the whole scene, "
            "hands, people, tools, labels, text, "
            "smoke, fire, cartoon, CGI, illustration, "
            "blurry, low quality"
        ),

        "steps": 24,
        "guidance_scale": 7.5,
        "strength": 0.78,
    },

    # --------------------------------------------------------
    # 2. 테이프 부위 찢어짐 + 누수
    # --------------------------------------------------------
    "tape_tear_leak": {
        "defect_type": "tape_tear_leak",

        "prompt": (
            "close-up of a flexible industrial hose "
            "wrapped with sealing tape, "
            "a small realistic tear in the taped section "
            "near the hose connection, "
            "clear water leaking from the torn taped area, "
            "visible wet tape texture, "
            "a thin continuous water stream, "
            "small transparent droplets and glossy wet "
            "reflections around the damaged tape, "
            "realistic industrial CCTV image, "
            "same hose, same tape wrapping, "
            "same equipment, same lighting and perspective, "
            "localized damage only, photorealistic"
        ),

        "negative_prompt": (
            "extra hoses, changed hose shape, "
            "different tape style, missing connector, "
            "completely broken hose, large rupture, "
            "severed hose, exploded hose, "
            "heavy flooding, huge splash, "
            "different equipment, changed background, "
            "hands, tools, text, labels, "
            "cartoon, painting, CGI, blurry, low quality"
        ),

        "steps": 26,
        "guidance_scale": 7.5,
        "strength": 0.82,
    },

    # --------------------------------------------------------
    # 3. 강한 물 분사
    # --------------------------------------------------------
    "strong_spray": {
        "defect_type": "strong_water_spray",

        "prompt": (
            "close-up of an industrial flexible hose "
            "near a metal fitting, "
            "a small damaged area at the hose connection "
            "causing a strong narrow stream of clear water "
            "to spray outward, "
            "a clearly visible pressurized transparent "
            "water jet with bright specular highlights, "
            "small airborne water droplets, "
            "fine spray particles and glossy wet reflections "
            "around the leaking point, "
            "the hose remains connected to the fitting, "
            "realistic industrial CCTV image, "
            "same hose, same fitting, same equipment, "
            "same lighting and same camera perspective, "
            "localized leak only, photorealistic"
        ),

        "negative_prompt": (
            "exploded hose, large torn opening, "
            "completely severed hose, detached hose, "
            "extreme flooding, huge splash, "
            "water covering entire image, "
            "extra hoses, extra connectors, "
            "different equipment, changed background, "
            "hands, tools, text, labels, "
            "cartoon, CGI, illustration, "
            "unrealistic water, blurry, low quality"
        ),

        "steps": 26,
        "guidance_scale": 7.5,
        "strength": 0.80,
    },
}


# ============================================================
# OPTIONAL PROCEDURAL ENHANCEMENT
# ============================================================

# diffusion 후 젖은 영역 추가
ENABLE_POOLING = True

# diffusion 후 작은 물방울 추가
ENABLE_DROPLET = True

POOLING_OPACITY = 0.16
DROPLET_OPACITY = 0.25


# ============================================================
# UTIL
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
            "PNG encode failed."
        )

    return encoded.tobytes()


def save_json(
    path: Path,
    data: Any,
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# BACKEND CHECK
# ============================================================

def check_backend() -> None:

    print()
    print(
        "========================================"
    )
    print(
        "Checking Synthetic NG backend..."
    )
    print(
        "========================================"
    )

    response = httpx.get(
        f"{BASE_URL}/synthetic/diffusion/models",
        timeout=30.0,
    )

    response.raise_for_status()

    models = response.json()

    target = None

    for model in models:

        if (
            model.get("model")
            == "sd15_openvino_inpaint"
        ):
            target = model
            break

    if target is None:
        raise RuntimeError(
            "sd15_openvino_inpaint "
            "provider not found."
        )

    print(
        "Model        :",
        target.get("model"),
    )

    print(
        "Dependency   :",
        target.get(
            "dependency_available"
        ),
    )

    print(
        "Loaded       :",
        target.get("loaded"),
    )

    print(
        "Device       :",
        target.get("device"),
    )

    if not target.get(
        "dependency_available",
        False,
    ):
        raise RuntimeError(
            "OpenVINO diffusion dependency "
            "is unavailable."
        )


# ============================================================
# ROI UI
# ============================================================

def select_roi_scaled(
    image: np.ndarray,
    window_name: str,
    max_width: int = PREVIEW_MAX_WIDTH,
    max_height: int = PREVIEW_MAX_HEIGHT,
) -> tuple[int, int, int, int]:

    image_h, image_w = image.shape[:2]

    scale = min(
        max_width / image_w,
        max_height / image_h,
        1.0,
    )

    preview = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    x, y, w, h = cv2.selectROI(
        window_name,
        preview,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        window_name
    )

    if (
        w <= 0
        or h <= 0
    ):
        raise RuntimeError(
            f"ROI selection cancelled: "
            f"{window_name}"
        )

    # preview coordinate
    # -> original coordinate
    x = int(
        round(x / scale)
    )
    y = int(
        round(y / scale)
    )
    w = int(
        round(w / scale)
    )
    h = int(
        round(h / scale)
    )

    return (
        x,
        y,
        w,
        h,
    )


def expand_to_square(
    x: int,
    y: int,
    w: int,
    h: int,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[
    int,
    int,
    int,
    int,
]:

    cx = x + w / 2
    cy = y + h / 2

    side = max(
        w,
        h,
    )

    side = int(
        round(
            side
            * (
                1
                + padding_ratio * 2
            )
        )
    )

    side = min(
        side,
        image_width,
        image_height,
    )

    x1 = int(
        round(
            cx - side / 2
        )
    )

    y1 = int(
        round(
            cy - side / 2
        )
    )

    x1 = max(
        0,
        min(
            x1,
            image_width - side,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            image_height - side,
        ),
    )

    return (
        x1,
        y1,
        side,
        side,
    )


# ============================================================
# DEFECT MASK
# ============================================================

def select_defect_mask(
    hose_roi: np.ndarray,
    title: str,
) -> dict:

    """
    사용자가 선택한 rectangle을
    normalized rectangle mask로 변환.

    강한 물 분사의 경우
    연결부 + 물이 분사될 공간까지
    길게 선택하는 것을 권장.
    """

    h, w = hose_roi.shape[:2]

    x, y, rw, rh = select_roi_scaled(
        hose_roi,
        title,
        max_width=900,
        max_height=900,
    )

    center_x = (
        x + rw / 2
    ) / w

    center_y = (
        y + rh / 2
    ) / h

    width_ratio = (
        rw / w
    )

    height_ratio = (
        rh / h
    )

    return {
        "type": "rectangle",

        "center_x": float(
            center_x
        ),

        "center_y": float(
            center_y
        ),

        "width_ratio": float(
            width_ratio
        ),

        "height_ratio": float(
            height_ratio
        ),

        "rotation_deg": 0,

        "feather_px": 5,
    }


# ============================================================
# API CALL
# ============================================================

def call_generate_api(
    image: np.ndarray,
    payload: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:

    start = time.perf_counter()

    response = httpx.post(
        f"{BASE_URL}/synthetic/generate",

        params={
            "response_format": "zip",
        },

        data={
            "payload": json.dumps(
                payload
            ),
        },

        files={
            "source_image": (
                "source.png",
                encode_png(image),
                "image/png",
            ),
        },

        timeout=API_TIMEOUT_SEC,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"API elapsed: "
        f"{elapsed:.2f} sec"
    )

    if (
        response.status_code
        != 200
    ):

        print()
        print(
            "===== Backend Error ====="
        )

        print(
            response.text
        )

        response.raise_for_status()

    with zipfile.ZipFile(
        io.BytesIO(
            response.content
        )
    ) as zf:

        manifest = json.loads(
            zf.read(
                "manifest.json"
            )
        )

        image_result = cv2.imdecode(
            np.frombuffer(
                zf.read(
                    "candidates/00/image.png"
                ),
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imdecode(
            np.frombuffer(
                zf.read(
                    "candidates/00/mask.png"
                ),
                dtype=np.uint8,
            ),
            cv2.IMREAD_GRAYSCALE,
        )

        difference = cv2.imdecode(
            np.frombuffer(
                zf.read(
                    "candidates/00/difference.png"
                ),
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )

    return (
        image_result,
        mask,
        difference,
        manifest,
    )


# ============================================================
# PAYLOAD
# ============================================================

def build_diffusion_payload(
    mask_spec: dict,
    mode: str,
    seed: int,
) -> dict:

    config = (
        DIFFUSION_CONFIGS[
            mode
        ]
    )

    return {
        "method":
            "diffusion_inpaint",

        "defect_type":
            config[
                "defect_type"
            ],

        "severity":
            0.8,

        "count":
            1,

        "seed":
            seed,

        "mask":
            mask_spec,

        "parameters": {

            "model":
                "sd15_openvino_inpaint",

            "prompt":
                config[
                    "prompt"
                ],

            "negative_prompt":
                config[
                    "negative_prompt"
                ],

            "device":
                DEVICE,

            "input_size":
                INPUT_SIZE,

            "steps":
                config[
                    "steps"
                ],

            "guidance_scale":
                config[
                    "guidance_scale"
                ],

            "strength":
                config[
                    "strength"
                ],
        },
    }


def build_pooling_payload(
    mask_spec: dict,
    seed: int,
) -> dict:

    cx = mask_spec[
        "center_x"
    ]

    cy = mask_spec[
        "center_y"
    ]

    wr = mask_spec[
        "width_ratio"
    ]

    hr = mask_spec[
        "height_ratio"
    ]

    return {
        "method": "procedural",

        "defect_type":
            "liquid_pooling",

        "severity": 0.4,

        "count": 1,

        "seed": seed,

        "mask": {
            "type":
                "random_blob",

            "center_x":
                cx,

            # 살짝 아래쪽으로
            "center_y":
                min(
                    cy + 0.025,
                    0.95,
                ),

            "width_ratio":
                min(
                    wr * 0.9,
                    0.30,
                ),

            "height_ratio":
                min(
                    hr * 0.75,
                    0.20,
                ),

            "feather_px": 5,

            "parameters": {
                "points": 14,
                "irregularity": 0.25,
                "smooth": 9,
            },
        },

        "parameters": {
            "pattern": "pooling",

            # BGR
            "color": [
                190,
                195,
                200,
            ],

            "opacity":
                POOLING_OPACITY,
        },
    }


def build_droplet_payload(
    mask_spec: dict,
    seed: int,
) -> dict:

    cx = mask_spec[
        "center_x"
    ]

    cy = mask_spec[
        "center_y"
    ]

    return {
        "method": "procedural",

        "defect_type":
            "liquid_droplet",

        "severity":
            0.4,

        "count":
            1,

        "seed":
            seed,

        "mask": {
            "type":
                "ellipse",

            "center_x":
                min(
                    cx + 0.02,
                    0.95,
                ),

            "center_y":
                min(
                    cy + 0.07,
                    0.95,
                ),

            "width_ratio":
                0.025,

            "height_ratio":
                0.050,

            "rotation_deg":
                0,

            "feather_px":
                3,
        },

        "parameters": {
            "pattern":
                "droplet",

            "color": [
                195,
                200,
                205,
            ],

            "opacity":
                DROPLET_OPACITY,
        },
    }


# ============================================================
# MODE SELECT
# ============================================================

def select_defect_mode(
    major_index: int,
    hose_index: int,
) -> str | None:

    print()
    print(
        "================================"
    )

    print(
        f"Major ROI {major_index} / "
        f"Hose {hose_index}"
    )

    print(
        "================================"
    )

    print(
        "1 : 볼트/너트 연결부 누수"
    )

    print(
        "2 : 테이프 찢어짐 + 누수"
    )

    print(
        "3 : 강한 물 분사"
    )

    print(
        "0 : 건너뛰기"
    )

    while True:

        value = input(
            "Select mode: "
        ).strip()

        if value == "0":
            return None

        if value == "1":
            return "connection_leak"

        if value == "2":
            return "tape_tear_leak"

        if value == "3":
            return "strong_spray"

        print(
            "0, 1, 2, 3 중 하나를 "
            "입력하세요."
        )


# ============================================================
# IMAGE SAVE
# ============================================================

def ensure_size(
    image: np.ndarray,
    width: int,
    height: int,
    interpolation: int,
) -> np.ndarray:

    if (
        image.shape[1] == width
        and image.shape[0] == height
    ):
        return image

    return cv2.resize(
        image,
        (
            width,
            height,
        ),
        interpolation=interpolation,
    )


# ============================================================
# PROCESS ONE HOSE
# ============================================================

def process_hose(
    hose_roi: np.ndarray,
    major_index: int,
    hose_index: int,
    mode: str,
) -> tuple[
    np.ndarray,
    dict,
]:

    folder = (
        OUTPUT_DIR
        / f"major_{major_index:02d}"
        / f"hose_{hose_index:02d}"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            folder
            / "00_original.png"
        ),
        hose_roi,
    )

    # --------------------------------------------------------
    # defect mask
    # --------------------------------------------------------

    print()
    print(
        "Defect가 생길 영역을 "
        "선택하세요."
    )

    if mode == "strong_spray":

        print(
            "강한 분사는 연결부뿐 아니라 "
            "물이 뻗어갈 방향의 공간까지 "
            "길게 선택하세요."
        )

    mask_spec = (
        select_defect_mask(
            hose_roi,
            (
                f"Major {major_index} "
                f"Hose {hose_index} "
                "- Select Defect Area"
            ),
        )
    )

    # --------------------------------------------------------
    # diffusion
    # --------------------------------------------------------

    seed = (
        1000
        + major_index * 100
        + hose_index * 10
    )

    diffusion_payload = (
        build_diffusion_payload(
            mask_spec,
            mode,
            seed,
        )
    )

    save_json(
        folder
        / "01_diffusion_request.json",
        diffusion_payload,
    )

    print()
    print(
        f"[Diffusion] {mode}"
    )

    (
        result,
        diffusion_mask,
        diffusion_diff,
        diffusion_manifest,
    ) = call_generate_api(
        hose_roi,
        diffusion_payload,
    )

    h, w = (
        hose_roi.shape[:2]
    )

    result = ensure_size(
        result,
        w,
        h,
        cv2.INTER_LANCZOS4,
    )

    diffusion_mask = ensure_size(
        diffusion_mask,
        w,
        h,
        cv2.INTER_NEAREST,
    )

    diffusion_diff = ensure_size(
        diffusion_diff,
        w,
        h,
        cv2.INTER_LINEAR,
    )

    cv2.imwrite(
        str(
            folder
            / "02_diffusion.png"
        ),
        result,
    )

    cv2.imwrite(
        str(
            folder
            / "02_diffusion_mask.png"
        ),
        diffusion_mask,
    )

    cv2.imwrite(
        str(
            folder
            / "02_diffusion_difference.png"
        ),
        diffusion_diff,
    )

    # --------------------------------------------------------
    # Optional Pooling
    # --------------------------------------------------------

    if ENABLE_POOLING:

        pooling_payload = (
            build_pooling_payload(
                mask_spec,
                seed + 1,
            )
        )

        print(
            "[Procedural] pooling"
        )

        (
            result,
            pool_mask,
            pool_diff,
            pool_manifest,
        ) = call_generate_api(
            result,
            pooling_payload,
        )

        result = ensure_size(
            result,
            w,
            h,
            cv2.INTER_LANCZOS4,
        )

        cv2.imwrite(
            str(
                folder
                / "03_pooling.png"
            ),
            result,
        )

    # --------------------------------------------------------
    # Optional Droplet
    # --------------------------------------------------------

    if ENABLE_DROPLET:

        droplet_payload = (
            build_droplet_payload(
                mask_spec,
                seed + 2,
            )
        )

        print(
            "[Procedural] droplet"
        )

        (
            result,
            drop_mask,
            drop_diff,
            drop_manifest,
        ) = call_generate_api(
            result,
            droplet_payload,
        )

        result = ensure_size(
            result,
            w,
            h,
            cv2.INTER_LANCZOS4,
        )

        cv2.imwrite(
            str(
                folder
                / "04_droplet.png"
            ),
            result,
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            folder
            / "05_final.png"
        ),
        result,
    )

    metadata = {
        "major_roi":
            major_index,

        "hose_index":
            hose_index,

        "mode":
            mode,

        "seed":
            seed,

        "mask":
            mask_spec,

        "diffusion_manifest":
            diffusion_manifest,
    }

    save_json(
        folder
        / "metadata.json",
        metadata,
    )

    return (
        result,
        metadata,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    check_backend()

    original = cv2.imread(
        str(
            IMAGE_PATH
        )
    )

    if original is None:
        raise FileNotFoundError(
            f"Image not found: "
            f"{IMAGE_PATH}"
        )

    image_h, image_w = (
        original.shape[:2]
    )

    print()
    print(
        f"Input image: "
        f"{image_w} x {image_h}"
    )

    final_image = (
        original.copy()
    )

    all_metadata = []

    # ========================================================
    # MAJOR ROI LOOP
    # ========================================================

    for major_index in range(
        1,
        NUM_MAJOR_ROIS + 1,
    ):

        print()
        print(
            "################################"
        )

        print(
            f"Select Major ROI "
            f"#{major_index}"
        )

        print(
            "################################"
        )

        mx, my, mw, mh = (
            select_roi_scaled(
                final_image,
                (
                    f"Select Major ROI "
                    f"#{major_index}"
                ),
            )
        )

        mx, my, mw, mh = (
            expand_to_square(
                mx,
                my,
                mw,
                mh,
                image_width=image_w,
                image_height=image_h,
                padding_ratio=(
                    MAJOR_ROI_PADDING
                ),
            )
        )

        major_roi = (
            final_image[
                my:my + mh,
                mx:mx + mw,
            ].copy()
        )

        major_folder = (
            OUTPUT_DIR
            / f"major_{major_index:02d}"
        )

        major_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(
                major_folder
                / "major_original.png"
            ),
            major_roi,
        )

        # ====================================================
        # HOSE LOOP
        # ====================================================

        for hose_index in range(
            1,
            HOSES_PER_MAJOR_ROI + 1,
        ):

            print()
            print(
                f"Major {major_index}: "
                f"Hose {hose_index} "
                f"세부 ROI를 선택하세요."
            )

            hx, hy, hw, hh = (
                select_roi_scaled(
                    major_roi,
                    (
                        f"Major {major_index} "
                        f"- Hose {hose_index}"
                    ),
                    max_width=900,
                    max_height=900,
                )
            )

            hx, hy, hw, hh = (
                expand_to_square(
                    hx,
                    hy,
                    hw,
                    hh,
                    image_width=(
                        major_roi.shape[1]
                    ),
                    image_height=(
                        major_roi.shape[0]
                    ),
                    padding_ratio=(
                        HOSE_ROI_PADDING
                    ),
                )
            )

            hose_roi = (
                major_roi[
                    hy:hy + hh,
                    hx:hx + hw,
                ].copy()
            )

            mode = (
                select_defect_mode(
                    major_index,
                    hose_index,
                )
            )

            if mode is None:

                print(
                    "Skipped."
                )

                continue

            (
                processed_hose,
                hose_metadata,
            ) = process_hose(
                hose_roi,
                major_index,
                hose_index,
                mode,
            )

            processed_hose = (
                ensure_size(
                    processed_hose,
                    hw,
                    hh,
                    cv2.INTER_LANCZOS4,
                )
            )

            # 세부 ROI를 major ROI에 반영
            major_roi[
                hy:hy + hh,
                hx:hx + hw,
            ] = processed_hose

            hose_metadata[
                "hose_roi"
            ] = {
                "x": hx,
                "y": hy,
                "width": hw,
                "height": hh,
            }

            all_metadata.append(
                hose_metadata
            )

        # ====================================================
        # MAJOR ROI -> FULL IMAGE
        # ====================================================

        cv2.imwrite(
            str(
                major_folder
                / "major_final.png"
            ),
            major_roi,
        )

        final_image[
            my:my + mh,
            mx:mx + mw,
        ] = major_roi

        all_metadata.append({
            "major_roi_index":
                major_index,

            "major_roi": {
                "x": mx,
                "y": my,
                "width": mw,
                "height": mh,
            },
        })

    # ========================================================
    # FINAL
    # ========================================================

    final_path = (
        OUTPUT_DIR
        / "FINAL_SYNTHETIC_CCTV.png"
    )

    cv2.imwrite(
        str(final_path),
        final_image,
    )

    save_json(
        OUTPUT_DIR
        / "FINAL_METADATA.json",
        {
            "source":
                str(IMAGE_PATH),

            "source_width":
                image_w,

            "source_height":
                image_h,

            "device":
                DEVICE,

            "major_rois":
                NUM_MAJOR_ROIS,

            "hoses_per_major_roi":
                HOSES_PER_MAJOR_ROI,

            "results":
                all_metadata,
        },
    )

    print()
    print(
        "========================================"
    )

    print(
        "Synthetic NG generation completed."
    )

    print(
        "========================================"
    )

    print(
        "Final image:"
    )

    print(
        final_path
    )


if __name__ == "__main__":
    main()
