from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import rasterio
from rasterio.windows import Window

from config import (
    FEATURES,
    CEH_NODATA_VALUES,
    RANDOM_SEED,
    SPATIAL_BLOCK_SIZE,
    TRAIN_FRACTION,
)
from common_preprocessing import allocate_proportional


def get_block_windows(
    src,
    block_size,
):
    resolution = src.res[0]

    block_pixels = int(
        round(block_size / resolution)
    )

    if block_pixels < 1:
        block_pixels = 1

    for row in range(
        0,
        src.height,
        block_pixels,
    ):

        for col in range(
            0,
            src.width,
            block_pixels,
        ):

            height = min(
                block_pixels,
                src.height - row,
            )

            width = min(
                block_pixels,
                src.width - col,
            )

            yield Window(
                col,
                row,
                width,
                height,
            )

# Defining function to count valid labelled pixels in every spatial block.
def count_valid_pixels_by_block(
    feature_raster,
    reference_raster,
    block_size,
):

    counts = {}

    with rasterio.open(
        feature_raster
    ) as features, rasterio.open(
        reference_raster
    ) as reference:

        if (
            features.width != reference.width
            or features.height != reference.height
        ):
            raise ValueError(
                "Feature and reference rasters "
                "are not aligned."
            )

        if features.transform != reference.transform:
            raise ValueError(
                "Feature and reference transforms differ."
            )

        for window in get_block_windows(
            features,
            block_size,
        ):

            feature_data = features.read(
                window=window
            )

            labels = reference.read(
                1,
                window=window,
            )

            valid = np.isfinite(
                feature_data
            ).all(axis=0)

            for nodata in CEH_NODATA_VALUES:
                valid &= labels != nodata

            valid &= labels >= 1
            valid &= labels <= 21

            if not valid.any():
                continue

            block_id = (
                f"{int(window.col_off)}_"
                f"{int(window.row_off)}"
            )

            values = labels[valid]

            class_counts = {}

            for cls in np.unique(values):

                class_counts[int(cls)] = int(
                    np.sum(values == cls)
                )

            counts[block_id] = {
                "window": window,
                "classes": class_counts,
            }

    return counts

# Allocate class samples, with at least one sample per occupied spatial block where feasible.
def allocate_to_blocks(
    target,
    block_counts,
):

    block_ids = list(
        block_counts.keys()
    )

    if not block_ids:
        return {}

    counts = np.array(
        [
            block_counts[b]
            for b in block_ids
        ],
        dtype=int,
    )

    total = counts.sum()

    if total == 0:
        return {}

    target = min(
        target,
        int(total),
    )

    allocation = np.floor(
        target * counts / total
    ).astype(int)

    if target >= len(block_ids):

        for i in range(
            len(block_ids)
        ):
            if allocation[i] == 0:
                allocation[i] = 1

    allocation = np.minimum(
        allocation,
        counts,
    )

    while allocation.sum() > target:

        candidates = np.where(
            allocation > 1
        )[0]

        if len(candidates) == 0:
            candidates = np.where(
                allocation > 0
            )[0]

        if len(candidates) == 0:
            break

        idx = candidates[
            np.argmax(
                allocation[candidates]
            )
        ]

        allocation[idx] -= 1

    while allocation.sum() < target:

        candidates = np.where(
            allocation < counts
        )[0]

        if len(candidates) == 0:
            break

        remaining = (
            counts[candidates]
            - allocation[candidates]
        )

        idx = candidates[
            np.argmax(remaining)
        ]

        allocation[idx] += 1

    return {
        block_ids[i]: int(allocation[i])
        for i in range(
            len(block_ids)
        )
        if allocation[i] > 0
    }

# Defining function to create spatially and categorically stratified sample. Intended internal evaluation: 70% training and 30% testing
def create_reference_samples(
    feature_raster,
    reference_raster,
    output_path,
    total_samples=20_000,
    min_samples_per_class=100,
    train_fraction=TRAIN_FRACTION,
    external=False,
    block_size=SPATIAL_BLOCK_SIZE,
    seed=RANDOM_SEED,
):

    rng = np.random.default_rng(
        seed
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    block_counts = count_valid_pixels_by_block(
        feature_raster,
        reference_raster,
        block_size,
    )

    global_counts = {}

    for block in block_counts.values():

        for cls, count in block[
            "classes"
        ].items():

            global_counts[cls] = (
                global_counts.get(cls, 0)
                + count
            )

    classes = np.array(
        sorted(global_counts)
    )

    counts = np.array(
        [
            global_counts[c]
            for c in classes
        ],
        dtype=int,
    )

    class_targets = allocate_proportional(
        total_samples,
        counts,
        min_samples_per_class,
    )

    class_target_map = {
        int(cls): int(n)
        for cls, n in zip(
            classes,
            class_targets,
        )
        if n > 0
    }

    rows = []

    for cls, class_target in class_target_map.items():

        class_blocks = {}

        for block_id, block in block_counts.items():

            count = block[
                "classes"
            ].get(cls, 0)

            if count > 0:
                class_blocks[
                    block_id
                ] = count

        block_allocations = allocate_to_blocks(
            class_target,
            class_blocks,
        )

        for block_id, n_samples in block_allocations.items():

            block_window = block_counts[
                block_id
            ]["window"]

            with rasterio.open(
                feature_raster
            ) as features, rasterio.open(
                reference_raster
            ) as reference:

                feature_data = features.read(
                    window=block_window
                )

                labels = reference.read(
                    1,
                    window=block_window,
                )

                valid = np.isfinite(
                    feature_data
                ).all(axis=0)

                valid &= labels == cls

            coordinates = np.argwhere(
                valid
            )

            if len(coordinates) == 0:
                continue

            n_samples = min(
                n_samples,
                len(coordinates),
            )

            selected = rng.choice(
                len(coordinates),
                size=n_samples,
                replace=False,
            )

            selected_coords = coordinates[
                selected
            ]

            for local_row, local_col in selected_coords:

                global_row = (
                    int(block_window.row_off)
                    + int(local_row)
                )

                global_col = (
                    int(block_window.col_off)
                    + int(local_col)
                )

                values = feature_data[
                    :,
                    local_row,
                    local_col,
                ]

                x, y = features.xy(
                    global_row,
                    global_col,
                )

                record = {
                    "row": global_row,
                    "col": global_col,
                    "x": x,
                    "y": y,
                    "block_id": block_id,
                    "label": int(cls),
                }

                for name, value in zip(
                    FEATURES,
                    values,
                ):
                    record[name] = float(
                        value
                    )

                rows.append(record)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No valid reference samples generated."
        )


    # Trainging/ testing split

    if external:

        df["split"] = "external_test"

    else:

        split_values = []

        # Split separately within each
        # class/spatial-block combination.
        for _, group in df.groupby(
            ["block_id", "label"],
            sort=False,
        ):

            indices = np.array(group.index.to_numpy(), dtype=np.int64, copy=True)

            rng.shuffle(indices)

            n_train = int(
                round(
                    len(indices)
                    * train_fraction
                )
            )

            if len(indices) >= 2:
                n_train = max(
                    1,
                    min(
                        len(indices) - 1,
                        n_train,
                    ),
                )

            train_indices = set(
                indices[:n_train]
            )

            for index in indices:

                split_values.append(
                    (
                        index,
                        "train"
                        if index in train_indices
                        else "test",
                    )
                )

        split_map = dict(
            split_values
        )

        df["split"] = [
            split_map[i]
            for i in df.index
        ]

    # Important fields first.
    ordered_columns = [
        "row",
        "col",
        "x",
        "y",
        "block_id",
        "label",
        "split",
    ] + FEATURES

    df = df[
        ordered_columns
    ]

    df.to_parquet(
        output_path,
        engine="pyarrow",
        index=False
    )

    print(
        f"Saved {len(df):,} samples: "
        f"{output_path}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        df.groupby(
            ["label", "split"]
        ).size()
    )

    return df