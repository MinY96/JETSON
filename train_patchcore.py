from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader
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


from src.anomaly.patchcore import (
    PatchCoreConfig,
    PatchCoreModel,
)
from src.anomaly.preprocessing import (
    NormalImageDataset,
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


def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def get_device(
    value: str,
) -> torch.device:

    value = value.lower()

    if value == "auto":

        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    return torch.device(
        value
    )


def synchronize(
    device: torch.device,
) -> None:

    if device.type == "cuda":
        torch.cuda.synchronize()


def parse_source(
    path: str,
) -> tuple[
    str,
    int | None,
]:

    stem = Path(
        path
    ).stem

    token = "__frame_"

    if token not in stem:
        return (
            stem,
            None,
        )

    video_name, frame_text = (
        stem.rsplit(
            token,
            1,
        )
    )

    try:
        frame_idx = int(
            frame_text
        )
    except ValueError:
        frame_idx = None

    return (
        video_name,
        frame_idx,
    )


# ============================================================
# Map save
# ============================================================

def save_anomaly_map(
    map_array: np.ndarray,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    minimum = float(
        map_array.min()
    )

    maximum = float(
        map_array.max()
    )

    if maximum > minimum:

        normalized = (
            (
                map_array
                - minimum
            )
            / (
                maximum
                - minimum
            )
            * 255.0
        )

    else:

        normalized = (
            np.zeros_like(
                map_array
            )
        )

    normalized = (
        normalized
        .clip(
            0,
            255,
        )
        .astype(
            np.uint8
        )
    )

    cv2.imwrite(
        str(
            output_path
        ),
        normalized,
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_dataset(
    *,
    model: PatchCoreModel,
    loader: DataLoader,
    split_name: str,
    roi_name: str,
    output_dir: Path,
    device: torch.device,
    save_maps: bool,
) -> pd.DataFrame:

    records = []

    map_dir = (
        output_dir
        / f"{split_name}_maps"
    )

    for images, paths in tqdm(
        loader,
        desc=f"{roi_name} {split_name}",
    ):

        synchronize(
            device
        )

        start = time.perf_counter()

        (
            scores,
            anomaly_maps,
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

        scores_np = (
            scores
            .detach()
            .cpu()
            .numpy()
        )

        maps_np = (
            anomaly_maps
            .detach()
            .cpu()
            .numpy()
        )

        per_image_ms = (
            elapsed_ms
            / len(paths)
        )

        for index, path in enumerate(
            paths
        ):

            (
                video_name,
                frame_idx,
            ) = parse_source(
                path
            )

            record = {
                "roi":
                    roi_name,

                "split":
                    split_name,

                "image_path":
                    path,

                "video_name":
                    video_name,

                "frame_idx":
                    frame_idx,

                "anomaly_score":
                    float(
                        scores_np[
                            index
                        ]
                    ),

                "inference_ms":
                    per_image_ms,
            }

            records.append(
                record
            )

            if save_maps:

                filename = (
                    Path(path).stem
                    + ".png"
                )

                save_anomaly_map(
                    maps_np[
                        index
                    ],
                    map_dir
                    / filename,
                )

    return pd.DataFrame(
        records
    )


def score_statistics(
    dataframe: pd.DataFrame,
) -> dict:

    scores = (
        dataframe[
            "anomaly_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if len(scores) == 0:
        return {}

    return {
        "count":
            int(
                len(scores)
            ),

        "min":
            float(
                np.min(
                    scores
                )
            ),

        "p25":
            float(
                np.quantile(
                    scores,
                    0.25,
                )
            ),

        "p50":
            float(
                np.quantile(
                    scores,
                    0.50,
                )
            ),

        "p75":
            float(
                np.quantile(
                    scores,
                    0.75,
                )
            ),

        "p90":
            float(
                np.quantile(
                    scores,
                    0.90,
                )
            ),

        "p95":
            float(
                np.quantile(
                    scores,
                    0.95,
                )
            ),

        "p99":
            float(
                np.quantile(
                    scores,
                    0.99,
                )
            ),

        "max":
            float(
                np.max(
                    scores
                )
            ),

        "mean":
            float(
                np.mean(
                    scores
                )
            ),

        "std":
            float(
                np.std(
                    scores
                )
            ),
    }


# ============================================================
# Train ROI
# ============================================================

def train_roi(
    *,
    roi_name: str,
    roi_config: dict,
    model_config: dict,
    device: torch.device,
    save_maps: bool,
) -> None:

    print()
    print(
        "=" * 70
    )

    print(
        f"PATCHCORE: {roi_name.upper()}"
    )

    print(
        "=" * 70
    )

    train_dir = resolve_path(
        roi_config[
            "train_dir"
        ]
    )

    val_dir = resolve_path(
        roi_config[
            "val_dir"
        ]
    )

    model_dir = resolve_path(
        roi_config[
            "model_dir"
        ]
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = PatchCoreConfig(
        input_size=int(
            model_config[
                "input_size"
            ]
        ),

        pool_kernel=int(
            model_config[
                "pool_kernel"
            ]
        ),

        candidate_pool_size=int(
            model_config[
                "candidate_pool_size"
            ]
        ),

        coreset_ratio=float(
            model_config[
                "coreset_ratio"
            ]
        ),

        max_memory_patches=int(
            model_config[
                "max_memory_patches"
            ]
        ),

        projection_dim=int(
            model_config[
                "projection_dim"
            ]
        ),

        query_chunk_size=int(
            model_config[
                "query_chunk_size"
            ]
        ),

        seed=int(
            model_config[
                "seed"
            ]
        ),
    )

    batch_size = int(
        model_config[
            "batch_size"
        ]
    )

    num_workers = int(
        model_config[
            "num_workers"
        ]
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_dataset = (
        NormalImageDataset(
            train_dir,
            input_size=(
                config.input_size
            ),
        )
    )

    val_dataset = (
        NormalImageDataset(
            val_dir,
            input_size=(
                config.input_size
            ),
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"Device       : {device}"
    )

    print(
        f"Train images : {len(train_dataset)}"
    )

    print(
        f"Val images   : {len(val_dataset)}"
    )

    print(
        f"Input size   : {config.input_size}"
    )

    model = PatchCoreModel(
        config,
        device,
        pretrained=True,
    )

    # --------------------------------------------------------
    # Fit memory bank
    # --------------------------------------------------------

    synchronize(
        device
    )

    train_start = (
        time.perf_counter()
    )

    fit_summary = model.fit(
        train_loader
    )

    synchronize(
        device
    )

    fit_seconds = (
        time.perf_counter()
        - train_start
    )

    print()
    print(
        "[MEMORY BANK]"
    )

    for key, value in (
        fit_summary.items()
    ):
        print(
            f"{key:24s}: {value}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        model_dir
        / "patchcore.pt"
    )

    model.save(
        model_path,
        extra={
            "roi":
                roi_name,

            "train_images":
                len(
                    train_dataset
                ),

            "fit_seconds":
                fit_seconds,
        },
    )

    print(
        f"\nModel saved: {model_path}"
    )

    # --------------------------------------------------------
    # Evaluate normal train
    # --------------------------------------------------------

    train_scores = evaluate_dataset(
        model=model,
        loader=train_loader,
        split_name="train",
        roi_name=roi_name,
        output_dir=model_dir,
        device=device,
        save_maps=False,
    )

    train_scores.to_csv(
        model_dir
        / "train_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Evaluate normal validation
    # --------------------------------------------------------

    val_scores = evaluate_dataset(
        model=model,
        loader=val_loader,
        split_name="val",
        roi_name=roi_name,
        output_dir=model_dir,
        device=device,
        save_maps=save_maps,
    )

    val_scores.to_csv(
        model_dir
        / "val_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_stats = score_statistics(
        train_scores
    )

    val_stats = score_statistics(
        val_scores
    )

    summary = {
        "roi":
            roi_name,

        "device":
            str(
                device
            ),

        "fit":
            fit_summary,

        "fit_seconds":
            fit_seconds,

        "train_score_statistics":
            train_stats,

        "val_score_statistics":
            val_stats,
    }

    (
        model_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "[TRAIN NORMAL SCORE]"
    )

    print(
        json.dumps(
            train_stats,
            indent=2,
        )
    )

    print()
    print(
        "[VAL NORMAL SCORE]"
    )

    print(
        json.dumps(
            val_stats,
            indent=2,
        )
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/patchcore.yaml"
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
        default="auto",
    )

    parser.add_argument(
        "--save-maps",
        action="store_true",
    )

    args = parser.parse_args()

    config_path = resolve_path(
        args.config
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        settings = yaml.safe_load(
            stream
        )

    model_config = settings[
        "model"
    ]

    seed = int(
        model_config[
            "seed"
        ]
    )

    set_seed(
        seed
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

    for roi_name in roi_names:

        train_roi(
            roi_name=roi_name,
            roi_config=(
                settings[
                    "rois"
                ][
                    roi_name
                ]
            ),
            model_config=model_config,
            device=device,
            save_maps=(
                args.save_maps
            ),
        )


if __name__ == "__main__":
    main()
