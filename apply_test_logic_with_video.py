from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import cv2
import requests

# Reuse the exact same matching / verification logic and thresholds.
from apply_test_logic import (
    API_BASE_URL,
    BOTTOM_TEMPLATE,
    TOP_TEMPLATE,
    TEMPLATE_DIR,
    MatchResult,
    TemplateInfo,
    check_backend,
    choose_template_source_image,
    draw_match,
    encode_png,
    extract_templates_via_backend,
    imread_unicode,
    load_template,
    localize_template,
    select_two_rois,
    verify_geometry,
    verify_texture,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "result" / "video"

# OpenCV's mp4v codec is broadly available for MP4 output.
VIDEO_CODEC = "mp4v"


def normalize_output_name(name: str) -> str:
    name = name.strip().strip('"')
    if not name:
        raise ValueError("Output file name cannot be empty")

    path = Path(name)
    if path.suffix.lower() != ".mp4":
        path = path.with_suffix(".mp4")
    return path.name


def ask_input_video() -> Path:
    """Open a file picker when possible, otherwise fall back to console input."""
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filename = filedialog.askopenfilename(
            title="Select MP4 video",
            filetypes=[
                ("MP4 video", "*.mp4"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if filename:
            return Path(filename)
    except Exception:
        pass

    value = input("Input video path: ").strip().strip('"')
    if not value:
        raise RuntimeError("Input video was not selected")
    return Path(value)


def resolve_output_dir(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()

    raw = input(
        f"Output directory [{DEFAULT_OUTPUT_DIR}]: "
    ).strip().strip('"')
    if not raw:
        return DEFAULT_OUTPUT_DIR
    return Path(raw).expanduser().resolve()


def resolve_output_name(value: str | None) -> str:
    if value:
        return normalize_output_name(value)

    raw = input("Output video file name (e.g. test_case_1_result.mp4): ")
    return normalize_output_name(raw)


def open_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter: {output_path}\n"
            f"codec={VIDEO_CODEC}, fps={fps}, size={width}x{height}"
        )
    return writer


def infer_template_on_frame(
    session: requests.Session,
    template: TemplateInfo,
    frame: Any,
    frame_path: Path,
    frame_bytes: bytes,
) -> tuple[MatchResult, float]:
    """
    Apply the same matching logic as apply_test_logic.py.

    inference_time_ms includes:
      localize_planar_object API
      -> project-side geometry verification
      -> HOG texture API / verification when enabled

    Video decoding and VideoWriter encoding are intentionally excluded.
    """
    start = time.perf_counter()

    result = localize_template(
        session=session,
        template=template,
        scene_path=frame_path,
        scene_bytes=frame_bytes,
    )
    result = verify_geometry(result, template)
    result = verify_texture(session, result, template, frame)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def process_video(
    session: requests.Session,
    input_video: Path,
    output_dir: Path,
    output_name: str,
) -> None:
    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    if input_video.suffix.lower() != ".mp4":
        print(f"[warning] Input is not .mp4: {input_video.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / output_name
    summary_path = output_dir / f"{Path(output_name).stem}_match_summary.csv"

    templates = [
        load_template(session, "bottom", BOTTOM_TEMPLATE),
        load_template(session, "top", TOP_TEMPLATE),
    ]

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or not (fps < float("inf")):
        fps = 30.0
        print("[warning] Source FPS could not be read. Using 30 FPS.")

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(
            f"Invalid source video size: width={width}, height={height}"
        )

    writer = open_video_writer(
        output_path=output_video_path,
        fps=fps,
        width=width,
        height=height,
    )

    print(f"Input video : {input_video}")
    print(f"Video info  : {width}x{height}, {fps:.3f} FPS, {frame_count} frames")
    print(f"Output video: {output_video_path}")

    summary_rows: list[dict[str, Any]] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            annotated = frame.copy()

            # The backend consumes image files. Encode the current video frame once,
            # then reuse the same bytes for both template localization requests.
            frame_bytes = encode_png(frame)
            virtual_frame_path = Path(f"frame_{frame_index:06d}.png")

            frame_start = time.perf_counter()

            for template in templates:
                result, inference_time_ms = infer_template_on_frame(
                    session=session,
                    template=template,
                    frame=frame,
                    frame_path=virtual_frame_path,
                    frame_bytes=frame_bytes,
                )
                draw_match(annotated, result)

                summary_rows.append(
                    {
                        "image": virtual_frame_path.name,
                        "template": template.name,
                        "found": result.found,
                        "reason": result.reason,
                        "good_matches": result.good_matches,
                        "inliers": result.inliers,
                        "inlier_ratio": result.inlier_ratio,
                        "scale_x": result.scale_x,
                        "scale_y": result.scale_y,
                        "texture_similarity": result.texture_similarity,
                        "inference_time_ms": round(inference_time_ms, 3),
                    }
                )

                status = "FOUND" if result.found else "NOT_FOUND"
                texture_text = (
                    f", texture={result.texture_similarity:.3f}"
                    if result.texture_similarity is not None
                    else ""
                )
                print(
                    f"    frame={frame_index:06d} "
                    f"{template.name:<6} {status:<9} "
                    f"matches={result.good_matches}, "
                    f"inlier={result.inlier_ratio:.3f}{texture_text}, "
                    f"time={inference_time_ms:.1f} ms "
                    f"({result.reason})"
                )

            writer.write(annotated)

            frame_elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
            if frame_count > 0:
                print(
                    f"[{frame_index:06d}/{frame_count:06d}] "
                    f"frame inference={frame_elapsed_ms:.1f} ms"
                )
            else:
                print(
                    f"[{frame_index:06d}] "
                    f"frame inference={frame_elapsed_ms:.1f} ms"
                )

    finally:
        capture.release()
        writer.release()

    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer_csv = csv.DictWriter(
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
                "inference_time_ms",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(summary_rows)

    print("\nDone")
    print(f"Processed frames: {frame_index}")
    print(f"Result video    : {output_video_path}")
    print(f"Summary CSV     : {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the same bottom/top template matching logic to an MP4 video "
            "and save an annotated MP4 plus match summary CSV."
        )
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=None,
        help="Input MP4 path. If omitted, a file picker/console prompt is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the output MP4 and CSV are saved.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output video file name, e.g. test_case_1_result.mp4.",
    )
    parser.add_argument(
        "--select-template",
        action="store_true",
        help="Select a new source image and recreate bottom/top templates before video processing.",
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

    input_video = (
        args.input_video.expanduser().resolve()
        if args.input_video is not None
        else ask_input_video().expanduser().resolve()
    )
    output_dir = resolve_output_dir(args.output_dir)
    output_name = resolve_output_name(args.output_name)

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        check_backend(session)
        print(f"Backend OK: {API_BASE_URL}")

        templates_missing = not BOTTOM_TEMPLATE.exists() or not TOP_TEMPLATE.exists()
        should_extract = (
            args.select_template
            or args.template_source is not None
            or templates_missing
        )

        if should_extract:
            source_path = args.template_source or choose_template_source_image()
            source_path = source_path.expanduser().resolve()
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Template source image does not exist: {source_path}"
                )

            source_image = imread_unicode(source_path)
            print("Select BOTTOM ROI first, then TOP ROI.")
            rois = select_two_rois(source_image)
            extract_templates_via_backend(session, source_path, rois)
        else:
            print(f"Using existing template: {BOTTOM_TEMPLATE}")
            print(f"Using existing template: {TOP_TEMPLATE}")

        process_video(
            session=session,
            input_video=input_video,
            output_dir=output_dir,
            output_name=output_name,
        )


if __name__ == "__main__":
    main()
