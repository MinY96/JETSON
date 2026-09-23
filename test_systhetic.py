from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import httpx
import numpy as np


# ============================================================
# Configuration
# ============================================================

BASE_URL = "http://127.0.0.1:8000/api/v1"

IMAGE_PATH = Path(r"images\normal.png")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 첫 테스트 권장
PROMPT_PRESET = "tear_with_seepage"

DEVICE = "GPU"

SEED = 100

STEPS = 20
GUIDANCE_SCALE = 7.0
STRENGTH = 0.85

# 원본 2560x1440을 화면에 띄울 때 최대 크기
PREVIEW_MAX_WIDTH = 1400
PREVIEW_MAX_HEIGHT = 850

# 선택한 Hose ROI 주변에 조금 더 context 확보
ROI_PADDING_RATIO = 0.20


# ============================================================
# Basic utilities
# ============================================================

def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)

    if not ok:
        raise RuntimeError("PNG encoding failed.")

    return encoded.tobytes()


def save_json(path: Path, data: dict) -> None:
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
# Backend preflight
# ============================================================

def check_backend() -> None:

    print("Checking backend...")

    response = httpx.get(
        f"{BASE_URL}/synthetic/diffusion/models",
        timeout=30.0,
    )
    response.raise_for_status()

    models = response.json()

    target = None

    for model in models:
        if model["model"] == "sd15_openvino_inpaint":
            target = model
            break

    if target is None:
        raise RuntimeError(
            "sd15_openvino_inpaint provider "
            "was not found in backend."
        )

    print()
    print("=== OpenVINO Provider ===")
    print("model              :", target["model"])
    print("provider           :", target["provider"])
    print(
        "dependency_available:",
        target["dependency_available"],
    )
    print("loaded             :", target["loaded"])
    print("device             :", target["device"])
    print("model_id           :", target["model_id"])

    if not target["dependency_available"]:
        raise RuntimeError(
            "OpenVINO diffusion dependencies "
            "are not available in backend."
        )


# ============================================================
# ROI selection
# ============================================================

def select_roi_scaled(
    image: np.ndarray,
    window_name: str,
) -> tuple[int, int, int, int]:

    image_h, image_w = image.shape[:2]

    scale = min(
        PREVIEW_MAX_WIDTH / image_w,
        PREVIEW_MAX_HEIGHT / image_h,
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

    cv2.destroyWindow(window_name)

    if w <= 0 or h <= 0:
        raise RuntimeError(
            f"ROI selection cancelled: {window_name}"
        )

    # preview 좌표 -> 원본 좌표
    x = int(round(x / scale))
    y = int(round(y / scale))
    w = int(round(w / scale))
    h = int(round(h / scale))

    return x, y, w, h


def expand_to_square(
    x: int,
    y: int,
    w: int,
    h: int,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.20,
) -> tuple[int, int, int, int]:

    cx = x + (w / 2.0)
    cy = y + (h / 2.0)

    side = max(w, h)

    # 양쪽에 padding 추가
    side = int(
        round(
            side
            * (
                1.0
                + padding_ratio * 2.0
            )
        )
    )

    side = min(
        side,
        image_width,
        image_height,
    )

    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))

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

    return x1, y1, side, side


# ============================================================
# Defect area selection
# ============================================================

def select_defect_area(
    roi: np.ndarray,
) -> dict:

    display = roi.copy()

    h, w = display.shape[:2]

    max_size = 800

    scale = min(
        max_size / w,
        max_size / h,
        1.0,
    )

    preview = cv2.resize(
        display,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    dx, dy, dw, dh = cv2.selectROI(
        "Select defect area inside Hose ROI",
        preview,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        "Select defect area inside Hose ROI"
    )

    if dw <= 0 or dh <= 0:
        raise RuntimeError(
            "Defect area selection cancelled."
        )

    # preview -> ROI original coordinate
    dx = dx / scale
    dy = dy / scale
    dw = dw / scale
    dh = dh / scale

    # defect rectangle -> normalized ellipse parameters
    center_x = (dx + dw / 2.0) / w
    center_y = (dy + dh / 2.0) / h

    width_ratio = dw / w
    height_ratio = dh / h

    return {
        "type": "ellipse",
        "center_x": float(center_x),
        "center_y": float(center_y),
        "width_ratio": float(width_ratio),
        "height_ratio": float(height_ratio),
        "rotation_deg": 0.0,
        "feather_px": 5,
    }


# ============================================================
# API
# ============================================================

def generate_synthetic(
    roi: np.ndarray,
    mask_spec: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:

    payload = {
        "method": "diffusion_inpaint",

        "defect_type": PROMPT_PRESET,

        "severity": 0.5,

        "count": 1,

        "seed": SEED,

        "mask": mask_spec,

        "parameters": {
            "model": "sd15_openvino_inpaint",

            "prompt_preset": PROMPT_PRESET,

            "device": DEVICE,

            "input_size": 512,

            "steps": STEPS,

            "guidance_scale": GUIDANCE_SCALE,

            "strength": STRENGTH,
        },
    }

    print()
    print("=== Request ===")
    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
    )

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
                "hose_roi.png",
                encode_png(roi),
                "image/png",
            ),
        },

        # Intel UHD 최초 inference는
        # 충분히 길게 허용
        timeout=600.0,
    )

    if response.status_code != 200:

        print()
        print("=== Backend Error ===")
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

        generated = cv2.imdecode(
            np.frombuffer(
                archive.read(
                    "candidates/00/image.png"
                ),
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imdecode(
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
        generated,
        mask,
        difference,
        manifest,
    )


# ============================================================
# Visualization
# ============================================================

def draw_rect(
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
) -> np.ndarray:

    result = image.copy()

    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 255, 255),
        3,
    )

    cv2.putText(
        result,
        label,
        (
            x,
            max(30, y - 10),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


def mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:

    result = image.copy()

    overlay = np.zeros_like(result)

    overlay[:, :, 2] = mask

    result = cv2.addWeighted(
        result,
        1.0,
        overlay,
        0.4,
        0.0,
    )

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:

    check_backend()

    # --------------------------------------------------------
    # Load original 2560x1440 CCTV image
    # --------------------------------------------------------

    original = cv2.imread(
        str(IMAGE_PATH)
    )

    if original is None:
        raise FileNotFoundError(
            IMAGE_PATH
        )

    image_h, image_w = original.shape[:2]

    print()
    print(
        f"Input image: "
        f"{image_w} x {image_h}"
    )

    # --------------------------------------------------------
    # 1. Select hose connection
    # --------------------------------------------------------

    print()
    print(
        "1) 호스 시작부/연결부를 "
        "마우스로 대략 선택하세요."
    )

    x, y, w, h = select_roi_scaled(
        original,
        "Select Hose ROI",
    )

    # --------------------------------------------------------
    # 2. Square ROI + context
    # --------------------------------------------------------

    x, y, w, h = expand_to_square(
        x=x,
        y=y,
        w=w,
        h=h,
        image_width=image_w,
        image_height=image_h,
        padding_ratio=ROI_PADDING_RATIO,
    )

    hose_roi = original[
        y:y + h,
        x:x + w,
    ].copy()

    print(
        f"Hose ROI: "
        f"x={x}, y={y}, "
        f"w={w}, h={h}"
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "01_hose_roi.png"
        ),
        hose_roi,
    )

    full_roi_preview = draw_rect(
        original,
        x,
        y,
        w,
        h,
        "Hose ROI",
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "01_full_roi_preview.png"
        ),
        full_roi_preview,
    )

    # --------------------------------------------------------
    # 3. Select exact defect region
    # --------------------------------------------------------

    print()
    print(
        "2) ROI 안에서 "
        "누수/찢어짐을 생성할 위치를 "
        "작게 선택하세요."
    )

    mask_spec = select_defect_area(
        hose_roi
    )

    print()
    print("Defect mask:")
    print(
        json.dumps(
            mask_spec,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # 4. Generate
    # --------------------------------------------------------

    print()
    print(
        f"3) OpenVINO SD1.5 inference "
        f"({DEVICE})..."
    )

    (
        generated_roi,
        generated_mask,
        difference,
        manifest,
    ) = generate_synthetic(
        hose_roi,
        mask_spec,
    )

    # backend가 원본 ROI size로 다시 복원하므로
    # 보통 resize가 필요하지 않음
    if (
        generated_roi.shape[:2]
        != hose_roi.shape[:2]
    ):
        generated_roi = cv2.resize(
            generated_roi,
            (w, h),
            interpolation=cv2.INTER_LANCZOS4,
        )

    if (
        generated_mask.shape[:2]
        != hose_roi.shape[:2]
    ):
        generated_mask = cv2.resize(
            generated_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    # --------------------------------------------------------
    # 5. Save ROI results
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "02_generated_roi.png"
        ),
        generated_roi,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "03_mask.png"
        ),
        generated_mask,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "04_difference.png"
        ),
        difference,
    )

    overlay = mask_overlay(
        generated_roi,
        generated_mask,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "05_generated_mask_overlay.png"
        ),
        overlay,
    )

    # --------------------------------------------------------
    # 6. Composite into original CCTV
    # --------------------------------------------------------

    final = original.copy()

    final[
        y:y + h,
        x:x + w,
    ] = generated_roi

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "06_full_composite.png"
        ),
        final,
    )

    # --------------------------------------------------------
    # 7. Metadata
    # --------------------------------------------------------

    metadata = {
        "source_image": str(
            IMAGE_PATH
        ),

        "source_size": [
            image_w,
            image_h,
        ],

        "hose_roi": {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
        },

        "mask": mask_spec,

        "prompt_preset": (
            PROMPT_PRESET
        ),

        "device": DEVICE,

        "seed": SEED,

        "steps": STEPS,

        "guidance_scale": (
            GUIDANCE_SCALE
        ),

        "strength": STRENGTH,

        "manifest": manifest,
    }

    save_json(
        OUTPUT_DIR
        / "07_metadata.json",
        metadata,
    )

    print()
    print("==========================")
    print("Finished")
    print("==========================")

    print(
        OUTPUT_DIR
        / "01_hose_roi.png"
    )

    print(
        OUTPUT_DIR
        / "02_generated_roi.png"
    )

    print(
        OUTPUT_DIR
        / "03_mask.png"
    )

    print(
        OUTPUT_DIR
        / "04_difference.png"
    )

    print(
        OUTPUT_DIR
        / "06_full_composite.png"
    )


if __name__ == "__main__":
    main()
