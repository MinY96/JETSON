"""
python compare_blade_template_preprocessing.py ^
  --images output\05_phase_a_v3\measurement_frames ^
  --template references\blade_left.png ^
  --output output\07_template_comparison ^
  --search-roi 41 358 2027 186 ^
  --threshold 0.65 ^
  --clahe-clip 2.0 ^
  --clahe-grid 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


METHODS = ("gray", "clahe", "gradient")


def crop_roi(image, roi):
    x, y, w, h = map(int, roi)
    ih, iw = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(iw, x + w)
    y2 = min(ih, y + h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI: {roi}")

    return image[y1:y2, x1:x2], (x1, y1)


def preprocess(image, method, clahe_clip, clahe_grid):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "gray":
        return cv2.GaussianBlur(gray, (3, 3), 0)

    if method == "clahe":
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )
        enhanced = clahe.apply(gray)

        return cv2.GaussianBlur(enhanced, (3, 3), 0)

    if method == "gradient":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        grad_y = cv2.Sobel(
            blurred,
            cv2.CV_32F,
            0, 1,
            ksize=3,
        )

        # 같은 방식으로 처리한 template/search를
        # float32 gradient 공간에서 직접 비교.
        return grad_y

    raise ValueError(method)


def match_one(
    search_image,
    template,
    roi_offset,
    method,
    clahe_clip,
    clahe_grid,
):
    search = preprocess(
        search_image, method, clahe_clip, clahe_grid
    )
    target = preprocess(
        template, method, clahe_clip, clahe_grid
    )

    sh, sw = search.shape
    th, tw = target.shape

    if tw > sw or th > sh:
        raise ValueError(
            f"Template {tw}x{th} > ROI {sw}x{sh}"
        )

    if float(np.std(target)) < 1e-5:
        raise ValueError(
            f"{method}: template variance too small"
        )

    response = cv2.matchTemplate(
        search,
        target,
        cv2.TM_CCOEFF_NORMED,
    )

    response = np.nan_to_num(
        response,
        nan=-1.0,
        posinf=-1.0,
        neginf=-1.0,
    )

    _, score, _, loc = cv2.minMaxLoc(response)

    x = roi_offset[0] + loc[0]
    y = roi_offset[1] + loc[1]

    return {
        "score": float(score),
        "x": int(x),
        "y": int(y),
        "w": int(tw),
        "h": int(th),
    }


def draw_comparison(image, results, threshold):
    h, w = image.shape[:2]

    # 화면 확인을 위해 축소
    display_width = 900
    scale = min(1.0, display_width / w)

    panels = []

    colors = {
        "gray": (0, 255, 255),
        "clahe": (0, 255, 0),
        "gradient": (255, 140, 0),
    }

    for method in METHODS:
        result = results.get(method)

        preview = image.copy()

        if result is not None:
            x = result["x"]
            y = result["y"]
            tw = result["w"]
            th = result["h"]

            color = colors[method]

            cv2.rectangle(
                preview,
                (x, y),
                (x + tw, y + th),
                color,
                3,
            )

            label = (
                f"{method.upper()} "
                f"score={result['score']:.4f}"
            )
        else:
            label = f"{method.upper()} FAILED"

        preview = cv2.resize(
            preview,
            (
                max(1, round(w * scale)),
                max(1, round(h * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

        text_color = (
            (0, 255, 0)
            if result is not None
            and result["score"] >= threshold
            else (0, 0, 255)
        )

        # 제목이 이미지 위에 겹치지 않도록 별도 헤더
        header = np.full(
            (55, preview.shape[1], 3),
            35,
            dtype=np.uint8,
        )

        cv2.putText(
            header,
            label,
            (15, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2,
            cv2.LINE_AA,
        )

        panels.append(np.vstack([header, preview]))

    return np.hstack(panels)


def analyze_image(
    image,
    template,
    search_roi,
    clahe_clip,
    clahe_grid,
):
    search, offset = crop_roi(image, search_roi)

    results = {}

    for method in METHODS:
        results[method] = match_one(
            search,
            template,
            offset,
            method,
            clahe_clip,
            clahe_grid,
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare Blade Template Matching Preprocessing"
    )

    parser.add_argument("--images", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument(
        "--output",
        default="output/07_template_comparison",
    )

    parser.add_argument(
        "--search-roi",
        type=int,
        nargs=4,
        default=[41, 358, 2027, 186],
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--clahe-grid",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--location-tolerance",
        type=float,
        default=10.0,
        help="Allowed match location difference from Gray (px)",
    )

    args = parser.parse_args()

    image_dir = Path(args.images)
    output_dir = Path(args.output)

    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    template = cv2.imread(args.template)

    if template is None:
        raise FileNotFoundError(args.template)

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }
    )

    if not images:
        raise FileNotFoundError(image_dir)

    print("=" * 85)
    print("Blade Template Matching Preprocessing Comparison")
    print("=" * 85)
    print(f"Images       : {len(images)}")
    print(f"Template     : {args.template}")
    print(f"Search ROI   : {args.search_roi}")
    print(f"Threshold    : {args.threshold}")
    print(f"CLAHE clip   : {args.clahe_clip}")
    print(f"CLAHE grid   : {args.clahe_grid}")
    print()

    rows = []

    for i, path in enumerate(images, 1):
        image = cv2.imread(str(path))

        if image is None:
            rows.append({
                "image": path.name,
                "status": "image_read_failed",
            })
            continue

        try:
            results = analyze_image(
                image,
                template,
                tuple(args.search_roi),
                args.clahe_clip,
                args.clahe_grid,
            )

            gray = results["gray"]
            row = {
                "image": path.name,
                "status": "ok",
            }

            for method in METHODS:
                r = results[method]

                dx = r["x"] - gray["x"]
                dy = r["y"] - gray["y"]

                distance = float(np.hypot(dx, dy))

                row.update({
                    f"{method}_score": r["score"],
                    f"{method}_x": r["x"],
                    f"{method}_y": r["y"],
                    f"{method}_pass": (
                        r["score"] >= args.threshold
                    ),
                    f"{method}_dx": dx,
                    f"{method}_dy": dy,
                    f"{method}_distance": distance,
                    f"{method}_location_consistent": (
                        distance <= args.location_tolerance
                    ),
                })

            row["clahe_score_delta"] = (
                results["clahe"]["score"]
                - results["gray"]["score"]
            )

            row["gradient_score_delta"] = (
                results["gradient"]["score"]
                - results["gray"]["score"]
            )

            comparison = draw_comparison(
                image,
                results,
                args.threshold,
            )

            cv2.imwrite(
                str(comparison_dir / path.name),
                comparison,
            )

        except Exception as exc:
            row = {
                "image": path.name,
                "status": "error",
                "error": str(exc),
            }

        rows.append(row)

        if row["status"] == "ok":
            print(
                f"[{i:02d}/{len(images):02d}] "
                f"{path.name} | "
                f"Gray={row['gray_score']:.4f} | "
                f"CLAHE={row['clahe_score']:.4f} | "
                f"Gradient={row['gradient_score']:.4f} | "
                f"CLAHE shift={row['clahe_distance']:.1f}px"
            )
        else:
            print(
                f"[{i:02d}/{len(images):02d}] "
                f"{path.name} | ERROR: {row.get('error')}"
            )

    df = pd.DataFrame(rows)

    csv_path = output_dir / "matching_comparison.csv"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    valid = df[df["status"] == "ok"]

    print()
    print("=" * 85)
    print("Summary")
    print("=" * 85)

    if valid.empty:
        print("No valid results.")
        return

    for method in METHODS:
        count = int(valid[f"{method}_pass"].sum())
        avg = float(valid[f"{method}_score"].mean())
        minimum = float(valid[f"{method}_score"].min())

        stable_count = int(
            valid[f"{method}_location_consistent"].sum()
        )

        print(
            f"{method.upper():9s} | "
            f"Pass={count:2d}/{len(valid):2d} | "
            f"Mean={avg:.4f} | "
            f"Min={minimum:.4f} | "
            f"Same-location={stable_count}/{len(valid)}"
        )

    print()
    print("Last 5 images")
    print("-" * 85)

    columns = [
        "image",
        "gray_score",
        "clahe_score",
        "gradient_score",
        "clahe_distance",
        "gradient_distance",
    ]

    print(
        valid[columns]
        .tail(5)
        .to_string(index=False)
    )

    print()
    print(f"CSV        : {csv_path}")
    print(f"Comparison : {comparison_dir}")


if __name__ == "__main__":
    main()
