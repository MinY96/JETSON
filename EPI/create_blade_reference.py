
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def select_and_crop(
    image_path: str,
    output_path: str,
    display_width: int = 1200,
    display_height: int = 800,
):
    image_path = Path(image_path)
    output_path = Path(output_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    original_h, original_w = image.shape[:2]

    # 화면 크기에 맞추되 원본보다 확대하지 않음
    scale = min(
        display_width / original_w,
        display_height / original_h,
        1.0,
    )

    preview_w = max(1, round(original_w * scale))
    preview_h = max(1, round(original_h * scale))

    preview = cv2.resize(
        image,
        (preview_w, preview_h),
        interpolation=cv2.INTER_AREA,
    )

    window_name = "Select Blade Reference ROI"

    print("=" * 60)
    print("Blade Reference Crop Tool")
    print("=" * 60)
    print(f"Input image : {image_path}")
    print(f"Image size  : {original_w} x {original_h}")
    print(f"Preview     : {preview_w} x {preview_h}")
    print(f"Scale       : {scale:.4f}")
    print()
    print("Mouse drag  : Select ROI")
    print("Enter/Space : Confirm")
    print("C           : Reset selection")
    print("ESC         : Cancel")
    print()

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_AUTOSIZE,
    )

    # showCrosshair=True
    # fromCenter=False: 일반적인 좌상단 -> 우하단 선택
    x, y, w, h = cv2.selectROI(
        window_name,
        preview,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyAllWindows()

    if w <= 0 or h <= 0:
        print("[CANCEL] No ROI selected.")
        return None

    # 축소된 좌표를 원본 이미지 좌표로 변환
    x1 = max(0, round(x / scale))
    y1 = max(0, round(y / scale))

    x2 = min(
        original_w,
        round((x + w) / scale),
    )
    y2 = min(
        original_h,
        round((y + h) / scale),
    )

    original_roi = (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )

    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        raise ValueError(
            f"Invalid crop: {original_roi}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        cropped,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save image: {output_path}"
        )

    print("=" * 60)
    print("[SUCCESS] Blade reference saved")
    print("=" * 60)
    print(f"Output      : {output_path}")
    print(f"Original ROI: {original_roi}")
    print(f"Crop size   : {cropped.shape[1]} x {cropped.shape[0]}")

    return original_roi


def main():
    parser = argparse.ArgumentParser(
        description="Create Blade template by mouse ROI selection"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Source measurement image",
    )

    parser.add_argument(
        "--output",
        default="references/blade_left.png",
        help="Output reference PNG",
    )

    parser.add_argument(
        "--display-width",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--display-height",
        type=int,
        default=800,
    )

    args = parser.parse_args()

    select_and_crop(
        image_path=args.image,
        output_path=args.output,
        display_width=args.display_width,
        display_height=args.display_height,
    )


if __name__ == "__main__":
    main()
