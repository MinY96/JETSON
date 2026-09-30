from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WINDOW_NAME = "Gate Dataset Labeler"

VALID_MANUAL_LABELS = {
    "",
    "visible",
    "not_visible",
    "ignore",
}

VALID_LOCALIZATION_LABELS = {
    "",
    "good",
    "bad",
    "unknown",
}

EXTRA_FIELDS = [
    "localization_label",
]

# OpenCV display
INFO_PANEL_HEIGHT = 220

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
FONT_THICKNESS = 1
LINE_HEIGHT = 27


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result):
        return None

    return result


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_float(
    value: Any,
    digits: int = 3,
) -> str:
    number = safe_float(value)

    if number is None:
        return "-"

    return f"{number:.{digits}f}"


def read_csv(
    csv_path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV header not found: {csv_path}"
            )

        fieldnames = list(reader.fieldnames)

        for field in EXTRA_FIELDS:
            if field not in fieldnames:
                fieldnames.append(field)

        rows: list[dict[str, str]] = []

        for row in reader:
            normalized = {
                field: row.get(field, "") or ""
                for field in fieldnames
            }

            manual_label = normalized.get(
                "manual_label",
                "",
            )

            if manual_label not in VALID_MANUAL_LABELS:
                normalized["manual_label"] = ""

            localization_label = normalized.get(
                "localization_label",
                "",
            )

            if (
                localization_label
                not in VALID_LOCALIZATION_LABELS
            ):
                normalized["localization_label"] = ""

            rows.append(normalized)

    return rows, fieldnames


def atomic_save_csv(
    csv_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    temp_path = csv_path.with_suffix(
        csv_path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

        stream.flush()
        os.fsync(stream.fileno())

    os.replace(
        temp_path,
        csv_path,
    )


def resolve_image_path(
    row: dict[str, str],
    csv_path: Path,
) -> Path | None:
    raw_path = row.get(
        "image_path",
        "",
    )

    if raw_path:
        image_path = Path(raw_path)

        if image_path.exists():
            return image_path

        if not image_path.is_absolute():
            candidate = (
                PROJECT_ROOT
                / image_path
            )

            if candidate.exists():
                return candidate.resolve()

    # Dataset 이동 등에 대비한 fallback
    if raw_path:
        fallback = (
            csv_path.parent
            / "images"
            / Path(raw_path).name
        )

        if fallback.exists():
            return fallback.resolve()

    return None


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


def parse_polygon(
    raw: str,
) -> np.ndarray | None:
    if not raw:
        return None

    try:
        points = np.asarray(
            json.loads(raw),
            dtype=np.float32,
        ).reshape(-1, 2)
    except Exception:
        return None

    if points.shape[0] < 4:
        return None

    if not np.isfinite(points).all():
        return None

    return points


def draw_polygon(
    image: np.ndarray,
    polygon: np.ndarray | None,
) -> np.ndarray:
    output = image.copy()

    if polygon is None:
        return output

    points = np.round(
        polygon
    ).astype(np.int32)

    cv2.polylines(
        output,
        [points.reshape(-1, 1, 2)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    center = np.mean(
        polygon,
        axis=0,
    )

    cv2.circle(
        output,
        tuple(
            np.round(center).astype(int)
        ),
        radius=5,
        color=(0, 0, 255),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    return output


def put_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = FONT_SCALE,
    thickness: int = FONT_THICKNESS,
) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        FONT,
        scale,
        (235, 235, 235),
        thickness,
        cv2.LINE_AA,
    )


def build_canvas(
    image: np.ndarray,
    row: dict[str, str],
    *,
    roi_name: str,
    current: int,
    total: int,
) -> np.ndarray:
    polygon = parse_polygon(
        row.get(
            "polygon_local",
            "",
        )
    )

    rendered = draw_polygon(
        image,
        polygon,
    )

    h, w = rendered.shape[:2]

    canvas = np.zeros(
        (
            h + INFO_PANEL_HEIGHT,
            w,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        INFO_PANEL_HEIGHT:,
        :,
    ] = rendered

    manual_label = (
        row.get(
            "manual_label",
            "",
        )
        or "UNLABELED"
    )

    localization_label = (
        row.get(
            "localization_label",
            "",
        )
        or "-"
    )

    lines = [
        (
            f"[{roi_name.upper()}] "
            f"{current + 1}/{total}"
        ),
        (
            f"Video: {row.get('video_name', '-')}   "
            f"Frame: {row.get('frame_idx', '-')}   "
            f"Time: {format_float(row.get('timestamp_sec'), 2)} sec"
        ),
        (
            f"Status: {row.get('match_status', '-')}   "
            f"good_matches: {row.get('good_matches') or '-'}   "
            f"inliers: {row.get('inliers') or '-'}   "
            f"inlier_ratio: {format_float(row.get('inlier_ratio'))}"
        ),
        (
            f"area_ratio: {format_float(row.get('area_ratio'))}   "
            f"rotation: {format_float(row.get('rotation_deg'), 2)} deg   "
            f"API: {format_float(row.get('api_duration_ms'), 1)} ms"
        ),
        (
            f"Manual Label: {manual_label}   "
            f"Localization: {localization_label}"
        ),
        (
            "1 Visible | 2 Not Visible | 3 Ignore | "
            "G Loc-Good | B Loc-Bad | U Loc-Unknown"
        ),
        (
            "A Previous | D/Space Next | 0 Clear | "
            "W Save | Q Save & Quit"
        ),
    ]

    y = 28

    for line in lines:
        put_text(
            canvas,
            line,
            15,
            y,
        )
        y += LINE_HEIGHT

    return canvas


def resize_for_screen(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    h, w = image.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1.0,
    )

    if scale >= 1.0:
        return image

    new_w = max(
        1,
        int(round(w * scale)),
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def select_indices(
    rows: list[dict[str, str]],
    *,
    statuses: set[str] | None,
    sample_per_status: int,
    random_seed: int,
    unlabeled_only: bool,
) -> list[int]:
    indices = []

    for i, row in enumerate(rows):
        status = row.get(
            "match_status",
            "",
        )

        if (
            statuses is not None
            and status not in statuses
        ):
            continue

        if (
            unlabeled_only
            and row.get(
                "manual_label",
                "",
            )
        ):
            continue

        indices.append(i)

    if sample_per_status <= 0:
        return indices

    rng = random.Random(
        random_seed
    )

    grouped: dict[
        str,
        list[int],
    ] = {}

    for index in indices:
        status = rows[index].get(
            "match_status",
            "unknown",
        )

        grouped.setdefault(
            status,
            [],
        ).append(index)

    selected: list[int] = []

    for status in sorted(grouped):
        values = grouped[status]

        rng.shuffle(values)

        selected.extend(
            values[
                :sample_per_status
            ]
        )

    # 영상 순서 / 시간 순서로 다시 정렬
    selected.sort(
        key=lambda idx: (
            rows[idx].get(
                "video_name",
                "",
            ),
            safe_int(
                rows[idx].get(
                    "frame_idx",
                )
            )
            or 0,
        )
    )

    return selected


def find_first_unlabeled(
    rows: list[dict[str, str]],
    indices: list[int],
) -> int:
    for view_index, row_index in enumerate(
        indices
    ):
        if not rows[row_index].get(
            "manual_label",
            "",
        ):
            return view_index

    return 0


def print_summary(
    rows: list[dict[str, str]],
    indices: list[int],
) -> None:
    status_counter = Counter()
    label_counter = Counter()
    localization_counter = Counter()

    for index in indices:
        row = rows[index]

        status_counter[
            row.get(
                "match_status",
                "unknown",
            )
        ] += 1

        label_counter[
            row.get(
                "manual_label",
                "",
            )
            or "unlabeled"
        ] += 1

        localization_counter[
            row.get(
                "localization_label",
                "",
            )
            or "unlabeled"
        ] += 1

    print()
    print("========== View Summary ==========")
    print(f"Total: {len(indices)}")

    print("\nStatus:")
    for key, value in status_counter.items():
        print(
            f"  {key}: {value}"
        )

    print("\nManual labels:")
    for key, value in label_counter.items():
        print(
            f"  {key}: {value}"
        )

    print("\nLocalization:")
    for key, value in localization_counter.items():
        print(
            f"  {key}: {value}"
        )

    print("==================================")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GUI labeler for Gate dataset"
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "data/gate_dataset/train"
        ),
    )

    parser.add_argument(
        "--roi",
        choices=[
            "left",
            "right",
        ],
        default="left",
    )

    parser.add_argument(
        "--status",
        type=str,
        default="all",
        help=(
            "all or comma-separated statuses, e.g. "
            "matched,insufficient_matches"
        ),
    )

    parser.add_argument(
        "--sample-per-status",
        type=int,
        default=0,
        help=(
            "0 = use all. "
            "Example: 150 = max 150 samples per status."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--unlabeled-only",
        action="store_true",
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
    )

    parser.add_argument(
        "--max-height",
        type=int,
        default=950,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    dataset_dir = resolve_project_path(
        args.dataset_dir
    )

    csv_path = (
        dataset_dir
        / args.roi
        / "metadata.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found: {csv_path}"
        )

    rows, fieldnames = read_csv(
        csv_path
    )

    if args.status.lower() == "all":
        statuses = None
    else:
        statuses = {
            item.strip()
            for item
            in args.status.split(",")
            if item.strip()
        }

    indices = select_indices(
        rows,
        statuses=statuses,
        sample_per_status=args.sample_per_status,
        random_seed=args.seed,
        unlabeled_only=args.unlabeled_only,
    )

    if not indices:
        print(
            "No images matched the current filters."
        )
        return

    print_summary(
        rows,
        indices,
    )

    current = find_first_unlabeled(
        rows,
        indices,
    )

    dirty_changes = 0

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    try:
        while True:
            row_index = indices[
                current
            ]

            row = rows[
                row_index
            ]

            image_path = resolve_image_path(
                row,
                csv_path,
            )

            if image_path is None:
                print(
                    f"[WARN] Image not found: "
                    f"{row.get('image_path')}"
                )

                current = min(
                    current + 1,
                    len(indices) - 1,
                )

                continue

            image = imread_safe(
                image_path
            )

            if image is None:
                print(
                    f"[WARN] Could not read: "
                    f"{image_path}"
                )

                current = min(
                    current + 1,
                    len(indices) - 1,
                )

                continue

            canvas = build_canvas(
                image,
                row,
                roi_name=args.roi,
                current=current,
                total=len(indices),
            )

            display = resize_for_screen(
                canvas,
                max_width=args.max_width,
                max_height=args.max_height,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            key = cv2.waitKeyEx(
                0
            )

            if key < 0:
                continue

            # ASCII 부분
            ascii_key = key & 0xFF

            changed = False

            # --------------------------------
            # Manual label
            # --------------------------------
            if ascii_key == ord("1"):
                row[
                    "manual_label"
                ] = "visible"

                changed = True

                # 라벨 후 자동 다음
                if current < len(indices) - 1:
                    current += 1

            elif ascii_key == ord("2"):
                row[
                    "manual_label"
                ] = "not_visible"

                changed = True

                if current < len(indices) - 1:
                    current += 1

            elif ascii_key == ord("3"):
                row[
                    "manual_label"
                ] = "ignore"

                changed = True

                if current < len(indices) - 1:
                    current += 1

            # --------------------------------
            # Localization 평가
            # --------------------------------
            elif ascii_key in (
                ord("g"),
                ord("G"),
            ):
                row[
                    "localization_label"
                ] = "good"

                changed = True

            elif ascii_key in (
                ord("b"),
                ord("B"),
            ):
                row[
                    "localization_label"
                ] = "bad"

                changed = True

            elif ascii_key in (
                ord("u"),
                ord("U"),
            ):
                row[
                    "localization_label"
                ] = "unknown"

                changed = True

            # --------------------------------
            # Clear
            # --------------------------------
            elif ascii_key == ord("0"):
                row[
                    "manual_label"
                ] = ""

                row[
                    "localization_label"
                ] = ""

                changed = True

            # --------------------------------
            # Navigation
            # --------------------------------
            elif ascii_key in (
                ord("a"),
                ord("A"),
            ):
                current = max(
                    0,
                    current - 1,
                )

            elif ascii_key in (
                ord("d"),
                ord("D"),
                ord(" "),
            ):
                current = min(
                    len(indices) - 1,
                    current + 1,
                )

            # Windows Arrow Left
            elif key == 2424832:
                current = max(
                    0,
                    current - 1,
                )

            # Windows Arrow Right
            elif key == 2555904:
                current = min(
                    len(indices) - 1,
                    current + 1,
                )

            # --------------------------------
            # Save
            # --------------------------------
            elif ascii_key in (
                ord("w"),
                ord("W"),
            ):
                atomic_save_csv(
                    csv_path,
                    rows,
                    fieldnames,
                )

                dirty_changes = 0

                print(
                    "[SAVE] metadata.csv saved"
                )

            # --------------------------------
            # Quit
            # --------------------------------
            elif ascii_key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                atomic_save_csv(
                    csv_path,
                    rows,
                    fieldnames,
                )

                print(
                    "[SAVE] metadata.csv saved"
                )

                break

            if changed:
                dirty_changes += 1

                if (
                    args.save_every > 0
                    and dirty_changes
                    >= args.save_every
                ):
                    atomic_save_csv(
                        csv_path,
                        rows,
                        fieldnames,
                    )

                    dirty_changes = 0

                    print(
                        "[AUTO SAVE]"
                    )

    finally:
        if dirty_changes > 0:
            atomic_save_csv(
                csv_path,
                rows,
                fieldnames,
            )

            print(
                "[SAVE] final changes saved"
            )

        cv2.destroyAllWindows()

    print_summary(
        rows,
        indices,
    )


if __name__ == "__main__":
    main()
