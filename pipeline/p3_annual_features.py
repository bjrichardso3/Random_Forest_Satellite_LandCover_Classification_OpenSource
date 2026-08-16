from pathlib import Path
from collections import defaultdict
import re

import numpy as np
import rasterio
from rasterio.windows import Window

from config import FEATURES

# Defining function to extract the year from a filename
def extract_year(filename):

    matches = re.findall(
        r"(20\d{6})",
        filename,
    )

    if not matches:
        return None

    return matches[0][:4]

# Defining function to sort/ list files for a particular year
def list_year_files(
    input_folder,
    year,
):
    input_folder = Path(input_folder)

    files = []

    for path in input_folder.glob(
        "*_features30m.tif"
    ):

        extracted = extract_year(
            path.name
        )

        if extracted == str(year):
            files.append(path)

    return sorted(files)

# Defining function to calculate the annual mean of pixels for a given year
def annual_mean_features(
    input_folder,
    output_path,
    year,
    window_size=512,
):

    scenes = list_year_files(
        input_folder,
        year,
    )

    if not scenes:
        raise RuntimeError(
            f"No preprocessed scenes found "
            f"for {year}."
        )

    output_path = Path(output_path)

    if output_path.exists():
        print(
            f"Annual feature composite already exists: "
            f"{output_path}"
        )
        return output_path

    with rasterio.open(
        scenes[0]
    ) as ref:

        profile = ref.profile.copy()

        height = ref.height
        width = ref.width

    profile.update(
        count=len(FEATURES),
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=3,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        for row in range(
            0,
            height,
            window_size,
        ):

            for col in range(
                0,
                width,
                window_size,
            ):

                h = min(
                    window_size,
                    height - row,
                )

                w = min(
                    window_size,
                    width - col,
                )

                window = Window(
                    col,
                    row,
                    w,
                    h,
                )

                sums = np.zeros(
                    (
                        len(FEATURES),
                        h,
                        w,
                    ),
                    dtype="float32",
                )

                counts = np.zeros(
                    (
                        len(FEATURES),
                        h,
                        w,
                    ),
                    dtype="uint16",
                )

                for scene_path in scenes:

                    with rasterio.open(
                        scene_path
                    ) as src:

                        data = src.read(
                            window=window
                        ).astype(
                            "float32"
                        )

                    valid = np.isfinite(
                        data
                    )

                    sums += np.where(
                        valid,
                        data,
                        0,
                    )

                    counts += valid.astype(
                        "uint16"
                    )

                mean = np.full(
                    sums.shape,
                    np.nan,
                    dtype="float32",
                )

                valid = counts > 0

                mean[valid] = (
                    sums[valid]
                    / counts[valid]
                )

                dst.write(
                    mean,
                    window=window,
                )

        for index, feature in enumerate(
            FEATURES,
            start=1,
        ):
            dst.set_band_description(
                index,
                feature,
            )

        dst.update_tags(
            composite="temporal_mean",
            year=str(year),
            n_scenes=str(len(scenes)),
            seasonal_window="June-August",
        )

    print(
        f"Created annual feature composite: "
        f"{output_path}"
    )

    return output_path