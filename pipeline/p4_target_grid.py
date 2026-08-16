from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from config import FEATURES
from common_preprocessing import create_target_grid

# Defining function to resample the annual feature stack (satellite) to a target grid. Uses Weighted Average resampling as satellite data is continuous
def resample_feature_raster(
    input_raster,
    output_raster,
    aoi_geometry,
    resolution,
):

    grid = create_target_grid(
        aoi_geometry,
        resolution,
    )

    output_raster = Path(
        output_raster
    )

    output_raster.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        input_raster
    ) as src:

        profile = src.profile.copy()

        profile.update(
            {
                "height": grid["height"],
                "width": grid["width"],
                "transform": grid["transform"],
                "crs": grid["crs"],
                "count": len(FEATURES),
                "dtype": "float32",
                "nodata": np.nan,
                "compress": "deflate",
                "predictor": 3,
            }
        )

        with rasterio.open(
            output_raster,
            "w",
            **profile,
        ) as dst:

            for band in range(
                1,
                len(FEATURES) + 1,
            ):

                destination = np.full(
                    (
                        grid["height"],
                        grid["width"],
                    ),
                    np.nan,
                    dtype="float32",
                )

                reproject(
                    source=rasterio.band(
                        src,
                        band,
                    ),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=np.nan,
                    dst_transform=grid["transform"],
                    dst_crs=grid["crs"],
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                )

                dst.write(
                    destination,
                    band,
                )

                dst.set_band_description(
                    band,
                    FEATURES[band - 1],
                )

    print(
        f"Created {resolution} m feature stack: "
        f"{output_raster}"
    )

    return output_raster

# Defining function to resample the reference grid (CEH Land Cover) to the target grid using mode resampling as the CEH Land Cover data is categorical.
def resample_reference_raster(
    reference_raster,
    output_raster,
    aoi_geometry,
    resolution,
):

    grid = create_target_grid(
        aoi_geometry,
        resolution,
    )

    output_raster = Path(
        output_raster
    )

    output_raster.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        reference_raster
    ) as src:

        profile = {
            "driver": "GTiff",
            "height": grid["height"],
            "width": grid["width"],
            "count": 1,
            "dtype": "uint8",
            "crs": grid["crs"],
            "transform": grid["transform"],
            "nodata": 255,
            "compress": "deflate",
        }

        destination = np.full(
            (
                grid["height"],
                grid["width"],
            ),
            255,
            dtype="uint8",
        )

        reproject(
            source=rasterio.band(
                src,
                1,
            ),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=255,
            resampling=Resampling.mode,
        )

        with rasterio.open(
            output_raster,
            "w",
            **profile,
        ) as dst:

            dst.write(
                destination,
                1,
            )

    print(
        f"Created {resolution} m CEH reference: "
        f"{output_raster}"
    )

    return output_raster