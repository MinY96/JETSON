from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WINDOW_NAME = "Anomaly ROI Selector"


# ============================================================
# Utility
# ============================================================

def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def imread_safe(path: Path) -> np.ndarray | None:
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


def resize_for_screen(
    image: np.ndarray,
    max_width: int = 1600,
    max_height: int = 900,
) -> np.ndarray:

    height, width = image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    if scale >= 1.0:
        return image

    return cv2.resize(
        image,
        (
            int(width * scale),
            int(height * scale),
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# Image sampling
# ============================================================

def sample_images(
    image_paths: list[Path],
    sample_count: int,
) -> list[Path]:

    if len(image_paths) <= sample_count:
        return image_paths

    indices = np.linspace(
        0,
        len(image_paths) - 1,
        sample_count,
        dtype=int,
    )

    return [
        image_paths[index]
        for index in indices
    ]


# ============================================================
# Config
# ============================================================

def load_config(
    config_path: Path,
) -> dict:

    if not config_path.exists():
        raise FileNotFoundError(
            config_path
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        data = yaml.safe_load(stream)

    if not data:
        raise ValueError(
            f"Empty config: {config_path}"
        )

    return data


def get_existing_roi(
    config: dict,
    roi_name: str,
) -> tuple[int, int, int, int] | None:

    value = (
        config
        .get("rois", {})
        .get(roi_name, {})
        .get("anomaly_roi")
    )

    if value is None:
        return None

    if len(value) != 4:
        return None

    return tuple(
        int(v)
        for v in value
    )


def save_roi_to_config(
    config_path: Path,
    config: dict,
    roi_name: str,
    roi: tuple[int, int, int, int],
) -> None:

    if "rois" not in config:
        config["rois"] = {}

    if roi_name not in config["rois"]:
        config["rois"][roi_name] = {}

    config["rois"][
        roi_name
    ][
        "anomaly_roi"
    ] = list(roi)

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as stream:

        yaml.safe_dump(
            config,
            stream,
            allow_unicode=True,
            sort_keys=False,
        )


# ============================================================
# Drawing
# ============================================================

def draw_overlay(
    image: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    *,
    index: int,
    total: int,
    filename: str,
) -> np.ndarray:

    output = image.copy()

    if roi is not None:

        x, y, width, height = roi

        cv2.rectangle(
            output,
            (x, y),
            (
                x + width,
                y + height,
            ),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            (
                f"ROI: "
                f"[{x}, {y}, {width}, {height}]"
            ),
            (
                10,
                30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        (
            f"{index + 1}/{total}  "
            f"{filename}"
        ),
        (
            10,
            output.shape[0] - 45,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        (
            "R: Select ROI | "
            "A/D: Prev/Next | "
            "S: Save | Q: Quit"
        ),
        (
            10,
            output.shape[0] - 15,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# ROI selection
# ============================================================

def select_roi(
    image: np.ndarray,
) -> tuple[int, int, int, int] | None:

    roi = cv2.selectROI(
        "Select Anomaly ROI",
        image,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        "Select Anomaly ROI"
    )

    x, y, width, height = (
        int(v)
        for v in roi
    )

    if width <= 0 or height <= 0:
        return None

    return (
        x,
        y,
        width,
        height,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--roi",
        choices=[
            "left",
            "right",
        ],
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
        ],
        default="train",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/normal_roi.yaml"
        ),
    )

    parser.add_argument(
        "--normal-root",
        type=Path,
        default=Path(
            "data/normal_roi"
        ),
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    config_path = resolve_project_path(
        args.config
    )

    normal_root = resolve_project_path(
        args.normal_root
    )

    aligned_dir = (
        normal_root
        / args.split
        / args.roi
        / "aligned"
    )

    if not aligned_dir.exists():
        raise FileNotFoundError(
            f"Aligned directory not found: "
            f"{aligned_dir}"
        )

    image_paths = sorted(
        aligned_dir.glob("*.png")
    )

    if not image_paths:
        raise RuntimeError(
            f"No aligned images: "
            f"{aligned_dir}"
        )

    image_paths = sample_images(
        image_paths,
        args.sample_count,
    )

    config = load_config(
        config_path
    )

    roi = get_existing_roi(
        config,
        args.roi,
    )

    current = 0

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    print()
    print(
        f"ROI        : {args.roi}"
    )

    print(
        f"Images     : {len(image_paths)}"
    )

    print(
        f"Current ROI: {roi}"
    )

    print()
    print(
        "R : ROI 선택"
    )

    print(
        "A : 이전"
    )

    print(
        "D : 다음"
    )

    print(
        "S : YAML 저장"
    )

    print(
        "Q : 종료"
    )

    while True:

        image_path = image_paths[
            current
        ]

        image = imread_safe(
            image_path
        )

        if image is None:
            current = (
                current + 1
            ) % len(
                image_paths
            )

            continue

        display = draw_overlay(
            image,
            roi,
            index=current,
            total=len(
                image_paths
            ),
            filename=image_path.name,
        )

        display = resize_for_screen(
            display
        )

        cv2.imshow(
            WINDOW_NAME,
            display,
        )

        key = cv2.waitKeyEx(
            0
        )

        ascii_key = (
            key
            & 0xFF
        )

        # ----------------------------------------
        # Select
        # ----------------------------------------

        if ascii_key in (
            ord("r"),
            ord("R"),
        ):

            selected = select_roi(
                image
            )

            if selected is not None:
                roi = selected

                print(
                    f"[ROI] {roi}"
                )

        # ----------------------------------------
        # Previous
        # ----------------------------------------

        elif ascii_key in (
            ord("a"),
            ord("A"),
        ):

            current = max(
                0,
                current - 1,
            )

        # Windows Left Arrow
        elif key == 2424832:

            current = max(
                0,
                current - 1,
            )

        # ----------------------------------------
        # Next
        # ----------------------------------------

        elif ascii_key in (
            ord("d"),
            ord("D"),
            ord(" "),
        ):

            current = min(
                len(image_paths) - 1,
                current + 1,
            )

        # Windows Right Arrow
        elif key == 2555904:

            current = min(
                len(image_paths) - 1,
                current + 1,
            )

        # ----------------------------------------
        # Save
        # ----------------------------------------

        elif ascii_key in (
            ord("s"),
            ord("S"),
        ):

            if roi is None:

                print(
                    "[WARN] ROI가 선택되지 않았습니다."
                )

                continue

            save_roi_to_config(
                config_path,
                config,
                args.roi,
                roi,
            )

            print()
            print(
                "[SAVED]"
            )

            print(
                f"{args.roi}: "
                f"anomaly_roi: "
                f"{list(roi)}"
            )

            print()

        # ----------------------------------------
        # Quit
        # ----------------------------------------

        elif ascii_key in (
            ord("q"),
            ord("Q"),
            27,
        ):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
