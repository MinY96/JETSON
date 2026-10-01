from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.anomaly.patchcore import PatchCoreModel
from src.anomaly.preprocessing import (
    AspectPadPreprocessor,
    imread_rgb,
)


# ============================================================
# Utility
# ============================================================

def resolve_path(
    path: str | Path,
) -> Path:

    path = Path(path)

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def synchronize(
    device: torch.device,
) -> None:

    if device.type == "cuda":
        torch.cuda.synchronize()


def get_device(
    value: str,
) -> torch.device:

    if value == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(value)


# ============================================================
# Dataset
# ============================================================

class SyntheticDataset(Dataset):

    def __init__(
        self,
        metadata_path: Path,
        *,
        input_size: int,
    ) -> None:

        metadata = pd.read_csv(
            metadata_path,
            encoding="utf-8-sig",
        )

        if metadata.empty:
            raise RuntimeError(
                f"Empty synthetic metadata: "
                f"{metadata_path}"
            )

        required = {
            "synthetic_image",
            "defect_type",
        }

        missing = (
            required
            - set(metadata.columns)
        )

        if missing:
            raise ValueError(
                f"Missing columns: "
                f"{sorted(missing)}"
            )

        self.records = (
            metadata
            .to_dict(
                orient="records"
            )
        )

        self.preprocessor = (
            AspectPadPreprocessor(
                input_size
            )
        )

    def __len__(self) -> int:

        return len(
            self.records
        )

    def __getitem__(
        self,
        index: int,
    ):

        record = self.records[
            index
        ]

        path = resolve_path(
            record[
                "synthetic_image"
            ]
        )

        image = imread_rgb(
            path
        )

        tensor = (
            self.preprocessor(
                image
            )
        )

        return (
            tensor,
            str(path),
            str(
                record[
                    "defect_type"
                ]
            ),
        )


# ============================================================
# Metrics
# ============================================================

@dataclass(frozen=True)
class ClassificationMetrics:

    threshold: float

    tp: int
    tn: int
    fp: int
    fn: int

    accuracy: float
    precision: float
    recall: float
    specificity: float

    fpr: float
    f1: float
    balanced_accuracy: float


def safe_div(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return float("nan")

    return numerator / denominator


def calculate_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> ClassificationMetrics:

    # 0 = Normal
    # 1 = NG
    prediction = (
        scores >= threshold
    )

    target = (
        labels == 1
    )

    tp = int(
        np.sum(
            prediction & target
        )
    )

    tn = int(
        np.sum(
            ~prediction & ~target
        )
    )

    fp = int(
        np.sum(
            prediction & ~target
        )
    )

    fn = int(
        np.sum(
            ~prediction & target
        )
    )

    accuracy = safe_div(
        tp + tn,
        tp + tn + fp + fn,
    )

    precision = safe_div(
        tp,
        tp + fp,
    )

    recall = safe_div(
        tp,
        tp + fn,
    )

    specificity = safe_div(
        tn,
        tn + fp,
    )

    fpr = safe_div(
        fp,
        fp + tn,
    )

    if (
        np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall > 0
    ):
        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
    else:
        f1 = float("nan")

    balanced_accuracy = (
        np.nanmean(
            [
                recall,
                specificity,
            ]
        )
    )

    return ClassificationMetrics(
        threshold=float(
            threshold
        ),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=float(
            accuracy
        ),
        precision=float(
            precision
        ),
        recall=float(
            recall
        ),
        specificity=float(
            specificity
        ),
        fpr=float(
            fpr
        ),
        f1=float(
            f1
        ),
        balanced_accuracy=float(
            balanced_accuracy
        ),
    )


# ============================================================
# Synthetic inference
# ============================================================

@torch.no_grad()
def evaluate_synthetic(
    *,
    model: PatchCoreModel,
    loader: DataLoader,
    device: torch.device,
    roi_name: str,
) -> pd.DataFrame:

    records = []

    for (
        images,
        paths,
        defect_types,
    ) in tqdm(
        loader,
        desc=f"{roi_name} synthetic NG",
    ):

        synchronize(
            device
        )

        start = (
            time.perf_counter()
        )

        (
            scores,
            _
        ) = model.predict(
            images
        )

        synchronize(
            device
        )

        elapsed_ms = (
            (
                time.perf_counter()
                - start
            )
            * 1000.0
        )

        score_values = (
            scores
            .detach()
            .cpu()
            .numpy()
        )

        per_image_ms = (
            elapsed_ms
            / len(paths)
        )

        for (
            path,
            defect_type,
            score,
        ) in zip(
            paths,
            defect_types,
            score_values,
        ):

            records.append(
                {
                    "roi":
                        roi_name,

                    "dataset":
                        "synthetic_ng",

                    "label":
                        1,

                    "defect_type":
                        defect_type,

                    "image_path":
                        path,

                    "anomaly_score":
                        float(score),

                    "inference_ms":
                        per_image_ms,
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# Normal scores
# ============================================================

def load_normal_scores(
    model_dir: Path,
    roi_name: str,
) -> pd.DataFrame:

    csv_path = (
        model_dir
        / "val_scores.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Normal val score not found: "
            f"{csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
    )

    if "anomaly_score" not in df.columns:
        raise ValueError(
            f"anomaly_score missing: "
            f"{csv_path}"
        )

    result = pd.DataFrame(
        {
            "roi":
                roi_name,

            "dataset":
                "normal_val",

            "label":
                0,

            "defect_type":
                "normal",

            "image_path":
                df["image_path"],

            "anomaly_score":
                pd.to_numeric(
                    df[
                        "anomaly_score"
                    ],
                    errors="coerce",
                ),

            "inference_ms":
                df.get(
                    "inference_ms",
                    np.nan,
                ),
        }
    )

    return (
        result
        .dropna(
            subset=[
                "anomaly_score"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Score statistics
# ============================================================

def score_statistics(
    values: np.ndarray,
) -> dict[str, Any]:

    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "count":
            int(
                len(values)
            ),

        "min":
            float(
                np.min(values)
            ),

        "p25":
            float(
                np.quantile(
                    values,
                    0.25,
                )
            ),

        "p50":
            float(
                np.quantile(
                    values,
                    0.50,
                )
            ),

        "p75":
            float(
                np.quantile(
                    values,
                    0.75,
                )
            ),

        "p90":
            float(
                np.quantile(
                    values,
                    0.90,
                )
            ),

        "p95":
            float(
                np.quantile(
                    values,
                    0.95,
                )
            ),

        "p99":
            float(
                np.quantile(
                    values,
                    0.99,
                )
            ),

        "max":
            float(
                np.max(values)
            ),

        "mean":
            float(
                np.mean(values)
            ),

        "std":
            float(
                np.std(values)
            ),
    }


def build_statistics(
    result: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        defect_type,
        group,
    ) in result.groupby(
        "defect_type",
        sort=True,
    ):

        stats = score_statistics(
            group[
                "anomaly_score"
            ].to_numpy(
                dtype=float
            )
        )

        rows.append(
            {
                "defect_type":
                    defect_type,
                **stats,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Exhaustive threshold search
# ============================================================

def generate_thresholds(
    scores: np.ndarray,
) -> np.ndarray:

    values = np.unique(
        np.sort(
            scores.astype(
                float
            )
        )
    )

    if len(values) == 1:
        return np.array(
            [
                np.nextafter(
                    values[0],
                    -np.inf,
                ),
                np.nextafter(
                    values[0],
                    np.inf,
                ),
            ]
        )

    midpoints = (
        values[:-1]
        + values[1:]
    ) / 2.0

    return np.concatenate(
        [
            [
                np.nextafter(
                    values[0],
                    -np.inf,
                )
            ],
            midpoints,
            [
                np.nextafter(
                    values[-1],
                    np.inf,
                )
            ],
        ]
    )


def threshold_search(
    result: pd.DataFrame,
) -> pd.DataFrame:

    scores = (
        result[
            "anomaly_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    labels = (
        result[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )

    thresholds = (
        generate_thresholds(
            scores
        )
    )

    records = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            scores,
            labels,
            threshold,
        )

        records.append(
            asdict(
                metrics
            )
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# Candidate threshold selection
# ============================================================

def best_balanced_threshold(
    search: pd.DataFrame,
) -> pd.Series:

    return (
        search
        .sort_values(
            by=[
                "balanced_accuracy",
                "recall",
                "specificity",
                "f1",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )


def threshold_with_fpr_limit(
    search: pd.DataFrame,
    max_fpr: float,
) -> pd.Series | None:

    candidates = search[
        search[
            "fpr"
        ]
        <= max_fpr
    ]

    if candidates.empty:
        return None

    return (
        candidates
        .sort_values(
            by=[
                "recall",
                "specificity",
                "balanced_accuracy",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )


def metrics_at_threshold(
    result: pd.DataFrame,
    threshold: float,
) -> ClassificationMetrics:

    return calculate_metrics(
        result[
            "anomaly_score"
        ].to_numpy(
            dtype=float
        ),
        result[
            "label"
        ].to_numpy(
            dtype=int
        ),
        threshold,
    )


def build_threshold_candidates(
    result: pd.DataFrame,
    search: pd.DataFrame,
) -> pd.DataFrame:

    normal_scores = (
        result.loc[
            result[
                "label"
            ] == 0,
            "anomaly_score",
        ]
        .to_numpy(
            dtype=float
        )
    )

    candidates: list[
        tuple[
            str,
            float,
        ]
    ] = []

    best = (
        best_balanced_threshold(
            search
        )
    )

    candidates.append(
        (
            "best_balanced_accuracy",
            float(
                best[
                    "threshold"
                ]
            ),
        )
    )

    for fpr in [
        0.00,
        0.01,
        0.02,
        0.05,
    ]:

        row = (
            threshold_with_fpr_limit(
                search,
                fpr,
            )
        )

        if row is not None:

            candidates.append(
                (
                    f"max_fpr_{int(fpr * 100):02d}pct",
                    float(
                        row[
                            "threshold"
                        ]
                    ),
                )
            )

    for q in [
        0.95,
        0.99,
    ]:

        threshold = float(
            np.quantile(
                normal_scores,
                q,
            )
        )

        candidates.append(
            (
                f"normal_p{int(q * 100)}",
                threshold,
            )
        )

    normal_max_threshold = (
        np.nextafter(
            float(
                np.max(
                    normal_scores
                )
            ),
            np.inf,
        )
    )

    candidates.append(
        (
            "normal_max_plus",
            normal_max_threshold,
        )
    )

    # duplicate threshold 제거
    seen = set()

    rows = []

    for (
        name,
        threshold,
    ) in candidates:

        key = round(
            threshold,
            10,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        metrics = metrics_at_threshold(
            result,
            threshold,
        )

        rows.append(
            {
                "strategy":
                    name,
                **asdict(
                    metrics
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Per defect detection
# ============================================================

def build_detection_by_type(
    result: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    synthetic = result[
        result[
            "label"
        ] == 1
    ]

    for _, candidate in (
        candidates.iterrows()
    ):

        threshold = float(
            candidate[
                "threshold"
            ]
        )

        for (
            defect_type,
            group,
        ) in synthetic.groupby(
            "defect_type"
        ):

            scores = group[
                "anomaly_score"
            ].to_numpy(
                dtype=float
            )

            detected = (
                scores
                >= threshold
            )

            rows.append(
                {
                    "strategy":
                        candidate[
                            "strategy"
                        ],

                    "threshold":
                        threshold,

                    "defect_type":
                        defect_type,

                    "count":
                        len(scores),

                    "detected":
                        int(
                            detected.sum()
                        ),

                    "missed":
                        int(
                            (
                                ~detected
                            ).sum()
                        ),

                    "recall":
                        float(
                            detected.mean()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Separation analysis
# ============================================================

def separation_summary(
    result: pd.DataFrame,
) -> dict[str, Any]:

    normal = (
        result.loc[
            result[
                "label"
            ] == 0,
            "anomaly_score",
        ]
        .to_numpy(
            dtype=float
        )
    )

    synthetic = (
        result.loc[
            result[
                "label"
            ] == 1,
            "anomaly_score",
        ]
        .to_numpy(
            dtype=float
        )
    )

    normal_max = float(
        np.max(
            normal
        )
    )

    synthetic_min = float(
        np.min(
            synthetic
        )
    )

    margin = (
        synthetic_min
        - normal_max
    )

    return {
        "normal_max":
            normal_max,

        "synthetic_min":
            synthetic_min,

        "separation_margin":
            margin,

        "perfect_separation":
            bool(
                margin > 0
            ),
    }


# ============================================================
# Plot
# ============================================================

def save_score_boxplot(
    result: pd.DataFrame,
    output_path: Path,
    roi_name: str,
) -> None:

    groups = []

    labels = []

    normal = (
        result[
            result[
                "defect_type"
            ] == "normal"
        ][
            "anomaly_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    groups.append(
        normal
    )

    labels.append(
        "normal"
    )

    defect_types = sorted(
        set(
            result.loc[
                result[
                    "label"
                ] == 1,
                "defect_type",
            ]
        )
    )

    for defect_type in defect_types:

        values = (
            result[
                result[
                    "defect_type"
                ] == defect_type
            ][
                "anomaly_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        groups.append(
            values
        )

        labels.append(
            defect_type
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.boxplot(
        groups,
        tick_labels=labels,
        showfliers=True,
    )

    ax.set_title(
        f"{roi_name.upper()} PatchCore score distribution"
    )

    ax.set_ylabel(
        "Anomaly score"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        fig
    )


# ============================================================
# Error inspection
# ============================================================

def save_error_cases(
    result: pd.DataFrame,
    threshold: float,
    output_dir: Path,
) -> None:

    normal_false_positive = result[
        (
            result[
                "label"
            ] == 0
        )
        & (
            result[
                "anomaly_score"
            ]
            >= threshold
        )
    ].sort_values(
        "anomaly_score",
        ascending=False,
    )

    synthetic_missed = result[
        (
            result[
                "label"
            ] == 1
        )
        & (
            result[
                "anomaly_score"
            ]
            < threshold
        )
    ].sort_values(
        "anomaly_score",
        ascending=True,
    )

    normal_false_positive.to_csv(
        output_dir
        / "normal_false_positive.csv",
        index=False,
        encoding="utf-8-sig",
    )

    synthetic_missed.to_csv(
        output_dir
        / "synthetic_missed.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# ROI evaluation
# ============================================================

def evaluate_roi(
    *,
    roi_name: str,
    roi_config: dict,
    synthetic_root: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:

    model_dir = resolve_path(
        roi_config[
            "model_dir"
        ]
    )

    model_path = (
        model_dir
        / "patchcore.pt"
    )

    synthetic_metadata = (
        synthetic_root
        / "val"
        / roi_name
        / "metadata.csv"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    if not synthetic_metadata.exists():
        raise FileNotFoundError(
            synthetic_metadata
        )

    print()
    print(
        "=" * 70
    )

    print(
        f"PATCHCORE EVALUATION: "
        f"{roi_name.upper()}"
    )

    print(
        "=" * 70
    )

    model = PatchCoreModel.load(
        model_path,
        device,
    )

    normal_scores = (
        load_normal_scores(
            model_dir,
            roi_name,
        )
    )

    synthetic_dataset = (
        SyntheticDataset(
            synthetic_metadata,
            input_size=(
                model.config.input_size
            ),
        )
    )

    loader = DataLoader(
        synthetic_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    synthetic_scores = (
        evaluate_synthetic(
            model=model,
            loader=loader,
            device=device,
            roi_name=roi_name,
        )
    )

    result = pd.concat(
        [
            normal_scores,
            synthetic_scores,
        ],
        ignore_index=True,
    )

    output_dir = (
        output_root
        / roi_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_dir
        / "all_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = (
        build_statistics(
            result
        )
    )

    stats.to_csv(
        output_dir
        / "score_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Threshold search
    # --------------------------------------------------------

    search = (
        threshold_search(
            result
        )
    )

    search.to_csv(
        output_dir
        / "threshold_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidates = (
        build_threshold_candidates(
            result,
            search,
        )
    )

    candidates.to_csv(
        output_dir
        / "threshold_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Detection by defect type
    # --------------------------------------------------------

    detection = (
        build_detection_by_type(
            result,
            candidates,
        )
    )

    detection.to_csv(
        output_dir
        / "detection_by_type.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Separation
    # --------------------------------------------------------

    separation = (
        separation_summary(
            result
        )
    )

    # --------------------------------------------------------
    # Diagnostic threshold
    # --------------------------------------------------------

    best_row = (
        candidates[
            candidates[
                "strategy"
            ]
            == "best_balanced_accuracy"
        ]
    )

    if best_row.empty:
        best_row = (
            candidates.iloc[
                [0]
            ]
        )

    diagnostic_threshold = (
        float(
            best_row.iloc[
                0
            ][
                "threshold"
            ]
        )
    )

    save_error_cases(
        result,
        diagnostic_threshold,
        output_dir,
    )

    save_score_boxplot(
        result,
        output_dir
        / "score_distribution.png",
        roi_name,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print(
        "[SCORE STATISTICS]"
    )

    print(
        stats.to_string(
            index=False
        )
    )

    print()
    print(
        "[SEPARATION]"
    )

    for (
        key,
        value,
    ) in separation.items():

        print(
            f"{key:22s}: {value}"
        )

    print()
    print(
        "[THRESHOLD CANDIDATES]"
    )

    print(
        candidates[
            [
                "strategy",
                "threshold",
                "precision",
                "recall",
                "specificity",
                "fpr",
                "balanced_accuracy",
                "fp",
                "fn",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "roi":
            roi_name,

        "device":
            str(
                device
            ),

        "normal_count":
            int(
                (
                    result[
                        "label"
                    ]
                    == 0
                ).sum()
            ),

        "synthetic_count":
            int(
                (
                    result[
                        "label"
                    ]
                    == 1
                ).sum()
            ),

        "statistics":
            stats.to_dict(
                orient="records"
            ),

        "separation":
            separation,

        "threshold_candidates":
            candidates.to_dict(
                orient="records"
            ),
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patchcore-config",
        type=Path,
        default=Path(
            "config/patchcore.yaml"
        ),
    )

    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=Path(
            "data/synthetic_ng"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/patchcore_validation"
        ),
    )

    parser.add_argument(
        "--roi",
        choices=[
            "left",
            "right",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    patchcore_config = (
        resolve_path(
            args.patchcore_config
        )
    )

    synthetic_root = (
        resolve_path(
            args.synthetic_root
        )
    )

    output_root = (
        resolve_path(
            args.output_root
        )
    )

    with patchcore_config.open(
        "r",
        encoding="utf-8",
    ) as stream:

        settings = yaml.safe_load(
            stream
        )

    device = get_device(
        args.device
    )

    if args.roi == "all":

        roi_names = [
            "left",
            "right",
        ]

    else:

        roi_names = [
            args.roi
        ]

    summaries = {}

    for roi_name in roi_names:

        summaries[
            roi_name
        ] = evaluate_roi(
            roi_name=roi_name,
            roi_config=(
                settings[
                    "rois"
                ][
                    roi_name
                ]
            ),
            synthetic_root=(
                synthetic_root
            ),
            output_root=(
                output_root
            ),
            device=device,
            batch_size=(
                args.batch_size
            ),
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summaries,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 70
    )

    print(
        "[DONE] PatchCore validation completed"
    )

    print(
        f"Output: {output_root}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
