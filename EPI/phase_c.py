
"""
Phase C - Integrated Type2 Wafer Gap Analysis
============================================

Input:
    Original CCTV MP4
    Normal / Dark Blade reference images

Pipeline:
    E1 -> E3 -> A2 -> A3 -> B(v5.2) -> Aggregation

Existing validated scripts are executed unchanged.

Outputs:
    output/07_phase_c/
        integrated_results.csv
        unmatched_records.csv
        summary.json
        run_manifest.json
        logs/

Notes:
    - No slot-based wafer inference.
    - Missing wafer gap remains empty (NaN in CSV).
    - Original video time is calculated only if
      the E3 clip-start time is explicitly available.
    - GT timestamps are optional at this runner level;
      E3 itself may require the argument.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import traceback

from datetime import datetime
from pathlib import Path

import pandas as pd


STAGES = ("e1", "e3", "a2", "a3", "b")

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}


# ============================================================
# Paths
# ============================================================

def build_paths(args):
    root = Path(args.work_root)

    return {
        "e1": root / "01_phase_e1",
        "e3": root / "03_extract_clips",
        "a2": root / "04_phase_a_v2",
        "a3": root / "05_phase_a_v3",
        "b": root / "06_phase_b_v5_2",
        "c": Path(args.output),
    }


def ensure_parent(path):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def list_images(directory):
    directory = Path(directory)

    if not directory.is_dir():
        return []

    return sorted(
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
    )


# ============================================================
# Command builder
# ============================================================

def build_commands(args, paths):
    py = sys.executable

    scripts = {
        "e1": args.e1_script,
        "e3": args.e3_script,
        "a2": args.a2_script,
        "a3": args.a3_script,
        "b": args.b_script,
    }

    e1_signals = (
        paths["e1"] / "motion_signals.csv"
    )

    e3_clips = (
        paths["e3"] / "clips"
    )

    a2_results = (
        paths["a2"] / "measurement_frames.csv"
    )

    a2_debug = (
        paths["a2"] / "debug"
    )

    a3_frames = (
        paths["a3"] / "measurement_frames"
    )

    commands = {}

    # --------------------------------------------------------
    # Phase E1
    # --------------------------------------------------------

    commands["e1"] = [
        py,
        str(scripts["e1"]),
        "--video", str(args.video),
        "--output", str(paths["e1"]),
        "--analysis-width", str(args.analysis_width),
        "--frame-step", str(args.frame_step),
    ]

    # --------------------------------------------------------
    # Phase E3
    # --------------------------------------------------------

    commands["e3"] = [
        py,
        str(scripts["e3"]),
        "--signals", str(e1_signals),
        "--video", str(args.video),
        "--output", str(paths["e3"]),
    ]

    if args.timestamps:
        commands["e3"] += [
            "--timestamps",
            str(args.timestamps),
        ]

    # Important:
    # Do not append --no-export.
    # A2/A3 need the extracted Type2 clips.

    # --------------------------------------------------------
    # Phase A v2
    # --------------------------------------------------------

    commands["a2"] = [
        py,
        str(scripts["a2"]),
        "--clips", str(e3_clips),
        "--output", str(paths["a2"]),
    ]

    # --------------------------------------------------------
    # Phase A v3
    # --------------------------------------------------------

    commands["a3"] = [
        py,
        str(scripts["a3"]),
        "--debug", str(a2_debug),
        "--clips", str(e3_clips),
        "--v2-results", str(a2_results),
        "--output", str(paths["a3"]),
    ]

    if args.valley_threshold is not None:
        commands["a3"] += [
            "--valley-threshold",
            str(args.valley_threshold),
        ]

    if args.rise_threshold is not None:
        commands["a3"] += [
            "--rise-threshold",
            str(args.rise_threshold),
        ]

    if args.rise_delta is not None:
        commands["a3"] += [
            "--rise-delta",
            str(args.rise_delta),
        ]

    if args.lead_frames is not None:
        commands["a3"] += [
            "--lead-frames",
            str(args.lead_frames),
        ]

    # --------------------------------------------------------
    # Phase B v5.2
    # --------------------------------------------------------

    commands["b"] = [
        py,
        str(scripts["b"]),
        "--images", str(a3_frames),
        "--template", str(args.template),
        "--dark-template", str(args.dark_template),
        "--output", str(paths["b"]),

        "--search-roi",
        *map(str, args.search_roi),

        "--measure-roi",
        *map(str, args.measure_roi),

        "--normal-x-ratio",
        str(args.normal_x_ratio),

        "--dark-x-ratio",
        str(args.dark_x_ratio),

        "--wafer-edge-threshold",
        str(args.wafer_edge_threshold),

        "--wafer-min-support",
        str(args.wafer_min_support),

        "--wafer-min-span",
        str(args.wafer_min_span),

        "--wafer-min-prominence",
        str(args.wafer_min_prominence),

        "--detail-scale",
        str(args.detail_scale),

        "--jpeg-quality",
        str(args.jpeg_quality),
    ]

    for name in (
        "normal_top_offset",
        "normal_bottom_offset",
        "dark_top_offset",
        "dark_bottom_offset",
    ):
        flag = "--" + name.replace("_", "-")

        commands["b"] += [
            flag,
            str(getattr(args, name)),
        ]

    # Advanced B arguments may be kept in a JSON array:
    #
    # [
    #   "--wafer-support-gradient", "15",
    #   "--min-gap", "5",
    #   "--max-gap", "100"
    # ]
    #
    # Appended arguments override earlier argparse values.
    if args.b_extra_json:
        with Path(args.b_extra_json).open(
            "r", encoding="utf-8"
        ) as f:
            extra = json.load(f)

        if not isinstance(extra, list) or not all(
            isinstance(item, str) for item in extra
        ):
            raise ValueError(
                "--b-extra-json must contain "
                "a JSON array of strings"
            )

        commands["b"].extend(extra)

    return commands


# ============================================================
# Stage execution
# ============================================================

def run_command(stage, command, log_dir):
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = log_dir / f"{stage}.log"

    print()
    print("=" * 75)
    print(f"START: {stage.upper()}")
    print("=" * 75)
    print(subprocess.list2cmdline(command))
    print()

    start = time.perf_counter()

    with log_path.open(
        "w", encoding="utf-8"
    ) as log:

        log.write(
            subprocess.list2cmdline(command) + "\n\n"
        )
        log.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)

        exit_code = process.wait()

    elapsed = time.perf_counter() - start

    print()
    print(
        f"[{stage.upper()}] "
        f"exit={exit_code}, elapsed={elapsed:.1f}s"
    )

    if exit_code != 0:
        raise RuntimeError(
            f"{stage} failed with exit code {exit_code}. "
            f"See {log_path}"
        )

    return {
        "stage": stage,
        "status": "completed",
        "elapsed_sec": round(elapsed, 3),
        "log_path": str(log_path),
        "command": command,
    }


def validate_stage_output(stage, paths):
    checks = {
        "e1": (
            paths["e1"] / "motion_signals.csv",
            "file",
        ),
        "e3": (
            paths["e3"] / "detected_events.csv",
            "file",
        ),
        "a2": (
            paths["a2"] / "measurement_frames.csv",
            "file",
        ),
        "a3": (
            paths["a3"] / "measurement_frames",
            "images",
        ),
        "b": (
            paths["b"] / "geometry_results.csv",
            "file",
        ),
    }

    path, kind = checks[stage]

    if kind == "file" and not path.is_file():
        raise FileNotFoundError(
            f"{stage} output missing: {path}"
        )

    if kind == "images" and not list_images(path):
        raise FileNotFoundError(
            f"{stage} measurement images missing: {path}"
        )

    # Additional prerequisites
    if stage == "e3":
        clips = paths["e3"] / "clips"

        if not clips.is_dir() or not any(
            clips.glob("*.mp4")
        ):
            raise FileNotFoundError(
                f"No Type2 clips found in {clips}"
            )

    if stage == "a2":
        debug_dir = paths["a2"] / "debug"

        if not debug_dir.is_dir() or not any(
            debug_dir.glob("*.csv")
        ):
            raise FileNotFoundError(
                f"A2 debug CSVs missing: {debug_dir}"
            )


# ============================================================
# Event key extraction
# ============================================================

TYPE2_PATTERN = re.compile(
    r"type[_-]?2[_-]?0*(\d+)",
    flags=re.IGNORECASE,
)


def extract_type2_key(value):
    """
    type2_001.mp4
    type2_001_measurement.jpg
    .../type2_001.mp4

        -> type2_001

    Never use Slot numbers as matching keys.
    """
    if value is None or pd.isna(value):
        return None

    text = str(value)
    match = TYPE2_PATTERN.search(text)

    if not match:
        return None

    number = int(match.group(1))
    return f"type2_{number:03d}"


def row_type2_key(row, kind):
    """
    Prefer explicit clip/image identification.
    Do not assume detected_events.csv has one
    row for each Type2 event.
    """
    if kind == "a3":
        identity_columns = [
            "clip",
            "clip_name",
            "clip_path",
            "image",
            "measurement_image",
            "measurement_image_path",
        ]
    elif kind == "e3":
        identity_columns = [
            "clip",
            "clip_name",
            "clip_path",
            "clip_filename",
            "output_clip",
            "saved_clip",
        ]
    else:
        identity_columns = ["image"]

    for column in identity_columns:
        if column in row.index:
            key = extract_type2_key(row[column])
            if key:
                return key

    # Numeric ID is accepted only if the column
    # explicitly represents the Type2 ordinal.
    for column in (
        "type2_index",
        "type2_idx",
        "type2_no",
        "type2_number",
    ):
        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        key = extract_type2_key(value)
        if key:
            return key

        try:
            number = float(value)

            if number.is_integer() and number >= 1:
                return f"type2_{int(number):03d}"
        except (TypeError, ValueError):
            pass

    return None


def read_indexed_csv(path, kind):
    if not Path(path).is_file():
        return {}, [], 0

    df = pd.read_csv(path)

    grouped = {}
    unlinked = []

    for _, row in df.iterrows():
        key = row_type2_key(row, kind)

        if key is None:
            unlinked.append({
                "source": kind,
                "reason": "no_explicit_type2_identifier",
                "record": row.to_dict(),
            })
            continue

        grouped.setdefault(key, []).append(
            row.to_dict()
        )

    index = {}

    for key, records in grouped.items():
        if len(records) == 1:
            index[key] = records[0]
        else:
            # Avoid silently assigning a raw E3
            # movement event to the wrong Type2 clip.
            unlinked.append({
                "source": kind,
                "reason": "duplicate_type2_identifier",
                "type2_key": key,
                "record_count": len(records),
            })

    return index, unlinked, len(df)


# ============================================================
# Merge E3 + A3 + B results
# ============================================================

def first_float(record, columns):
    if record is None:
        return None

    for column in columns:
        value = record.get(column)

        if value is None or pd.isna(value):
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def aggregate_results(paths, video):
    b_path = paths["b"] / "geometry_results.csv"
    a3_path = (
        paths["a3"] / "measurement_frames.csv"
    )
    e3_path = (
        paths["e3"] / "detected_events.csv"
    )

    b_df = pd.read_csv(b_path)

    a3_map, a3_unlinked, a3_count = (
        read_indexed_csv(a3_path, "a3")
    )

    e3_map, e3_unlinked, e3_count = (
        read_indexed_csv(e3_path, "e3")
    )

    output_rows = []
    unmatched = a3_unlinked + e3_unlinked

    b_keys = set()

    for _, b_row in b_df.iterrows():
        merged = b_row.to_dict()

        filename = merged.get("image")
        key = extract_type2_key(filename)

        merged["video"] = str(video)
        merged["type2_key"] = key

        if key is None:
            merged["join_status"] = (
                "missing_type2_identifier"
            )
            merged["source_video_sec"] = None

            unmatched.append({
                "source": "b",
                "reason": "missing_type2_identifier",
                "image": filename,
            })

            output_rows.append(merged)
            continue

        if key in b_keys:
            unmatched.append({
                "source": "b",
                "reason": "duplicate_type2_identifier",
                "type2_key": key,
                "image": filename,
            })

        b_keys.add(key)

        a3_record = a3_map.get(key)
        e3_record = e3_map.get(key)

        # Prefix raw metadata to avoid overwriting
        # Phase B measurement columns.
        if a3_record is not None:
            merged.update({
                f"a3_{column}": value
                for column, value
                in a3_record.items()
            })

        if e3_record is not None:
            merged.update({
                f"e3_{column}": value
                for column, value
                in e3_record.items()
            })

        if a3_record is None and e3_record is None:
            join_status = "b_only"
        elif a3_record is None:
            join_status = "missing_a3_metadata"
        elif e3_record is None:
            join_status = "missing_e3_metadata"
        else:
            join_status = "joined"

        merged["join_status"] = join_status

        # A3 measurement_sec is clip-local.
        # Do NOT treat it as source-video time.
        clip_measure_sec = first_float(
            a3_record,
            ["measurement_sec"],
        )

        clip_start_sec = first_float(
            e3_record,
            [
                "clip_start_sec",
                "clip_start_time_sec",
                "clip_begin_sec",
            ],
        )

        merged["measurement_clip_sec"] = (
            clip_measure_sec
        )

        if (
            clip_measure_sec is not None
            and clip_start_sec is not None
        ):
            merged["source_video_sec"] = (
                clip_start_sec + clip_measure_sec
            )
            merged["source_time_status"] = (
                "derived_from_explicit_clip_start"
            )
        else:
            merged["source_video_sec"] = None
            merged["source_time_status"] = (
                "clip_start_or_measurement_sec_missing"
            )

        output_rows.append(merged)

    # A3 or E3 record without corresponding B result.
    for source, mapping in (
        ("a3", a3_map),
        ("e3", e3_map),
    ):
        for key in mapping:
            if key not in b_keys:
                unmatched.append({
                    "source": source,
                    "reason": "no_phase_b_result",
                    "type2_key": key,
                })

    result_df = pd.DataFrame(output_rows)

    if "type2_key" in result_df.columns:
        result_df = result_df.sort_values(
            "type2_key",
            kind="stable",
            na_position="last",
        )

    unmatched_df = pd.DataFrame(unmatched)

    return result_df, unmatched_df, {
        "e3_csv_rows": e3_count,
        "a3_csv_rows": a3_count,
        "b_csv_rows": len(b_df),
        "e3_linkable_type2_rows": len(e3_map),
        "a3_linkable_type2_rows": len(a3_map),
        "joined_rows": int(
            (result_df["join_status"] == "joined").sum()
        ) if not result_df.empty else 0,
    }


# ============================================================
# Summary
# ============================================================

def make_summary(df, join_stats, paths):
    status_counts = {}

    if "status" in df.columns:
        status_counts = {
            str(k): int(v)
            for k, v in df["status"]
            .value_counts(dropna=False)
            .items()
        }

    result = {
        "total_measurement_rows": int(len(df)),
        "status_counts": status_counts,
        "join_statistics": join_stats,
        "output_paths": {
            "e1_signals": str(
                paths["e1"] / "motion_signals.csv"
            ),
            "e3_events": str(
                paths["e3"] / "detected_events.csv"
            ),
            "a3_frames": str(
                paths["a3"] / "measurement_frames"
            ),
            "b_results": str(
                paths["b"] / "geometry_results.csv"
            ),
            "overlay_dir": str(
                paths["b"] / "overlay"
            ),
            "detail_dir": str(
                paths["b"] / "detail"
            ),
        },
    }

    for column in ("top_gap_px", "bottom_gap_px"):
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        result[column] = {
            "detected_count": int(numeric.notna().sum()),
            "mean_px": (
                float(numeric.mean())
                if numeric.notna().any()
                else None
            ),
        }

    return result


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Phase C: MP4 -> Type2 Events -> "
            "Measurement Frames -> Wafer Gaps"
        )
    )

    parser.add_argument(
        "--video", required=True
    )

    parser.add_argument(
        "--timestamps",
        default=None,
        help="Optional GT Excel passed to E3",
    )

    parser.add_argument(
        "--template", required=True
    )

    parser.add_argument(
        "--dark-template", required=True
    )

    parser.add_argument(
        "--work-root",
        default="output",
    )

    parser.add_argument(
        "--output",
        default="output/07_phase_c",
    )

    parser.add_argument(
        "--start-from",
        choices=(
            "e1", "e3", "a2", "a3",
            "b", "aggregate",
        ),
        default="e1",
        help="Reuse completed earlier stages",
    )

    # Existing script paths
    parser.add_argument(
        "--e1-script",
        default="phase_e1_extract_motion_signals_v2.py",
    )

    parser.add_argument(
        "--e3-script",
        default="phase_e3_extract_type2_clips.py",
    )

    parser.add_argument(
        "--a2-script",
        default="phase_a_select_measurement_frame_v2.py",
    )

    parser.add_argument(
        "--a3-script",
        default="phase_a_select_measurement_frame_v3.py",
    )

    parser.add_argument(
        "--b-script",
        default="phase_b_template_matching_v5_2.py",
    )

    # Phase E
    parser.add_argument(
        "--analysis-width",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
    )

    # Phase A v3
    parser.add_argument(
        "--valley-threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--rise-threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--rise-delta",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--lead-frames",
        type=int,
        default=None,
    )

    # Phase B geometry
    parser.add_argument(
        "--search-roi",
        type=int,
        nargs=4,
        default=[41, 358, 2027, 186],
    )

    parser.add_argument(
        "--measure-roi",
        type=int,
        nargs=4,
        default=[49, 296, 600, 273],
    )

    parser.add_argument(
        "--normal-x-ratio",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--dark-x-ratio",
        type=float,
        default=0.7,
    )

    for name in (
        "normal_top_offset",
        "normal_bottom_offset",
        "dark_top_offset",
        "dark_bottom_offset",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=0.0,
        )

    # Phase B wafer detector
    parser.add_argument(
        "--wafer-edge-threshold",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--wafer-min-support",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--wafer-min-span",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--wafer-min-prominence",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--detail-scale",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=98,
    )

    parser.add_argument(
        "--b-extra-json",
        default=None,
        help="Optional JSON list of additional B CLI args",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser


# ============================================================
# Main
# ============================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    paths = build_paths(args)
    final_dir = paths["c"]
    log_dir = final_dir / "logs"

    video = Path(args.video)

    if not video.is_file():
        parser.error(
            f"Video not found: {video}"
        )

    if not 0 <= args.normal_x_ratio <= 1:
        parser.error("normal-x-ratio must be 0~1")

    if not 0 <= args.dark_x_ratio <= 1:
        parser.error("dark-x-ratio must be 0~1")

    if args.frame_step < 1:
        parser.error("frame-step must be >= 1")

    if args.timestamps and not Path(
        args.timestamps
    ).is_file():
        parser.error(
            f"GT file not found: {args.timestamps}"
        )

    commands = build_commands(args, paths)

    if args.start_from == "aggregate":
        stage_start = len(STAGES)
    else:
        stage_start = STAGES.index(
            args.start_from
        )

    # Check scripts that will actually run.
    if not args.dry_run:
        for stage in STAGES[stage_start:]:
            script = Path(commands[stage][1])

            if not script.is_file():
                parser.error(
                    f"Required script missing: {script}"
                )

        if "b" in STAGES[stage_start:]:
            for reference in (
                args.template,
                args.dark_template,
            ):
                if not Path(reference).is_file():
                    parser.error(
                        f"Reference missing: {reference}"
                    )

    print("=" * 75)
    print("PHASE C - INTEGRATED TYPE2 WAFER ANALYSIS")
    print("=" * 75)
    print(f"Video      : {video}")
    print(f"Work root  : {args.work_root}")
    print(f"Output     : {final_dir}")
    print(f"Start from : {args.start_from}")
    print()

    for stage in STAGES[stage_start:]:
        print(
            f"[{stage.upper()}] "
            + subprocess.list2cmdline(commands[stage])
        )

    if args.dry_run:
        print()
        print("[DRY RUN] No stages executed.")
        return

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "started_at": datetime.now().isoformat(),
        "video": str(video.resolve()),
        "start_from": args.start_from,
        "stages": [],
        "status": "running",
    }

    manifest_path = (
        final_dir / "run_manifest.json"
    )

    def save_manifest():
        with manifest_path.open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                manifest,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    save_manifest()

    total_start = time.perf_counter()

    try:
        for index, stage in enumerate(STAGES):

            if index < stage_start:
                validate_stage_output(
                    stage, paths
                )

                manifest["stages"].append({
                    "stage": stage,
                    "status": "reused",
                })

                print(
                    f"[REUSE] {stage.upper()} "
                    f"existing outputs"
                )

                save_manifest()
                continue

            stage_result = run_command(
                stage,
                commands[stage],
                log_dir,
            )

            validate_stage_output(
                stage, paths
            )

            manifest["stages"].append(
                stage_result
            )

            save_manifest()

        # ----------------------------------------------------
        # Final aggregation
        # ----------------------------------------------------

        print()
        print("=" * 75)
        print("FINAL AGGREGATION")
        print("=" * 75)

        df, unmatched_df, join_stats = (
            aggregate_results(
                paths,
                video,
            )
        )

        result_csv = (
            final_dir / "integrated_results.csv"
        )

        unmatched_csv = (
            final_dir / "unmatched_records.csv"
        )

        summary_path = (
            final_dir / "summary.json"
        )

        df.to_csv(
            result_csv,
            index=False,
            encoding="utf-8-sig",
        )

        unmatched_df.to_csv(
            unmatched_csv,
            index=False,
            encoding="utf-8-sig",
        )

        summary = make_summary(
            df,
            join_stats,
            paths,
        )

        summary["elapsed_sec"] = round(
            time.perf_counter() - total_start,
            3,
        )

        with summary_path.open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                summary,
                f,
                ensure_ascii=False,
                indent=2,
            )

        manifest["status"] = "completed"
        manifest["completed_at"] = (
            datetime.now().isoformat()
        )

        manifest["elapsed_sec"] = (
            summary["elapsed_sec"]
        )

        save_manifest()

        print()
        print("[SUMMARY]")
        print(json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ))

        print()
        print(f"Results : {result_csv}")
        print(f"Unlinked: {unmatched_csv}")
        print(f"Summary : {summary_path}")
        print(f"Manifest: {manifest_path}")

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["failed_at"] = (
            datetime.now().isoformat()
        )

        save_manifest()

        print()
        print(
            f"[PHASE C FAILED] "
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
