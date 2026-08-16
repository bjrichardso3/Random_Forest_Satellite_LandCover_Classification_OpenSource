from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

# Importing variables from config.py
from config import (
    COMMON_SATELLITE_RESOLUTION,
    LANDSAT_BAND_MAP,
    LANDSAT_CLOUD_BITS,
    LANDSAT_WATER_BIT,
    SENTINEL_BANDS,
    SENTINEL_CLOUD_CLASSES,
    SENTINEL_INVALID_CLASSES,
    SENTINEL_WATER_CLASS,
    FEATURES,
)

#Importing functions from common_preprocessing.py
from common_preprocessing import (
    calculate_indices,
    create_target_grid,
    geometry_mask_for_grid,
    get_sensor_type,
    load_aoi_geometry,
)

# Setting scene bands
SCENE_BANDS = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
]

# Defining function to map the asset bands for a given scene directory and sensor type
def asset_path_map(scene_dir, sensor):

    scene_dir = Path(scene_dir)

    if sensor.startswith("landsat"):
        mapping = LANDSAT_BAND_MAP[sensor]["bands"]
    else:
        mapping = SENTINEL_BANDS

    paths = {}

    for common_name, asset_name in mapping.items():

        matches = list(
            scene_dir.glob(
                f"*_{asset_name}.tif"
            )
        )

        if not matches:
            raise FileNotFoundError(
                f"Missing asset {asset_name} "
                f"in {scene_dir}"
            )

        paths[common_name] = matches[0]

    return paths

# Defining function to read a raster file, resample, and reproject it onto the target grid
def read_to_grid(
    path,
    grid,
    resampling,
    dtype="float32",
):
    with rasterio.open(path) as src:

        destination = np.full(
            (
                grid["height"],
                grid["width"],
            ),
            np.nan,
            dtype=dtype,
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )

    return destination

# Defining function to scale the Landsat reflectance values
def scale_landsat(data):
    return (data.astype("float32") * 0.0000275 - 0.2)

# Defining function to scale the Sentinel reflectance values
def scale_sentinel(data):
    return (data.astype("float32") / 10_000.0)

# Defining function to cloud mask landsat scenes using bit mask
def landsat_cloud_mask(qa):

    mask = np.zeros(
        qa.shape,
        dtype=bool,
    )

    for bit in LANDSAT_CLOUD_BITS:

        mask |= (
            (qa.astype("uint16") & (1 << bit))
            != 0
        )

    return mask

# Defining function to water mask landsat scenes using bit mask
def landsat_water_flag(qa):

    return (
        (
            qa.astype("uint16")
            & (1 << LANDSAT_WATER_BIT)
        )
        != 0
    ).astype("float32")

# Defining function to cloud mask sentinel scenes using bit mask
def sentinel_cloud_mask(scl):

    return np.isin(
        scl.astype("uint8"),
        SENTINEL_CLOUD_CLASSES,
    )

# Defining function to mask invalid sentinel scenes using bit mask
def sentinel_invalid_mask(scl):

    return np.isin(
        scl.astype("uint8"),
        SENTINEL_INVALID_CLASSES,
    )

# Defining function to water mask sentinel scenes using bit mask
def sentinel_water_flag(scl):

    return (
        scl.astype("uint8")
        == SENTINEL_WATER_CLASS
    ).astype("float32")

# Defining function to process a satellite scene and save the output to a specified folder
def process_scene(
    scene_dir,
    output_folder,
    aoi_geometry,
):
    """
    Process one scene onto the common 30 m BNG grid.

    Output bands:
        1-6   spectral
        7     NDVI
        8     NBR
        9     NDWI
        10    SAVI
        11    EVI
        12    water fraction/flag
    """

    scene_dir = Path(scene_dir)
    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    scene_name = scene_dir.name

    output_path = (
        output_folder
        / f"{scene_name}_features30m.tif"
    )

    if output_path.exists():
        print(
            f"Already processed: {scene_name}"
        )
        return output_path

    sensor = get_sensor_type(scene_name)

    grid = create_target_grid(
        aoi_geometry,
        COMMON_SATELLITE_RESOLUTION,
    )

    asset_paths = asset_path_map(
        scene_dir,
        sensor,
    )

    spectral = []

    for band_name in SCENE_BANDS:

        data = read_to_grid(
            asset_paths[band_name],
            grid,
            Resampling.average,
        )

        if sensor.startswith("landsat"):
            data = scale_landsat(data)
        else:
            data = scale_sentinel(data)

        spectral.append(data)

    spectral = np.stack(
        spectral
    ).astype("float32")

    # Quality/ cloud/ water mask application

    if sensor.startswith("landsat"):

        qa = read_to_grid(
            asset_paths["qa"],
            grid,
            Resampling.nearest,
            dtype="float32",
        )

        qa = np.nan_to_num(
            qa,
            nan=0,
        ).astype("uint16")

        cloud_mask = landsat_cloud_mask(
            qa
        )

        water = landsat_water_flag(
            qa
        )

    else:

        scl = read_to_grid(
            asset_paths["qa"],
            grid,
            Resampling.mode,
            dtype="float32",
        )

        scl = np.nan_to_num(
            scl,
            nan=0,
        ).astype("uint8")

        cloud_mask = sentinel_cloud_mask(
            scl
        )

        invalid_mask = sentinel_invalid_mask(
            scl
        )

        cloud_mask |= invalid_mask

        water = sentinel_water_flag(
            scl
        )

    # Valid data application

    valid = np.isfinite(
        spectral
    ).all(axis=0)

    # Avoid treating the fill value / all-zero imagery
    # as valid observations.
    valid &= np.any(
        spectral != 0,
        axis=0,
    )

    valid &= ~cloud_mask

    # Water is NOT removed.

    # AOI clipping.
    aoi_mask = geometry_mask_for_grid(
        aoi_geometry,
        grid,
    )

    valid &= aoi_mask

    spectral[:, ~valid] = np.nan

    water = water.astype("float32")
    water[~valid] = np.nan

    # Applying indices calculation

    indices = calculate_indices(
        spectral
    )

    indices[:, ~valid] = np.nan

    output_data = np.concatenate(
        [
            spectral,
            indices,
            water[None, :, :],
        ],
        axis=0,
    )

    assert output_data.shape[0] == len(
        FEATURES
    )

    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": output_data.shape[0],
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
    }

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        dst.write(output_data)

        for index, name in enumerate(
            FEATURES,
            start=1,
        ):
            dst.set_band_description(
                index,
                name,
            )

        dst.update_tags(
            sensor=sensor,
            common_resolution="30m",
            water_masked="false",
            cloud_masked="true",
        )

    print(
        f"Created: {output_path}"
    )

    return output_path


def preprocess_folder(
    raw_folder,
    output_folder,
    aoi_path,
):
    aoi_geometry = load_aoi_geometry(
        aoi_path
    )

    raw_folder = Path(raw_folder)

    scene_dirs = [
        p
        for p in raw_folder.iterdir()
        if p.is_dir()
    ]

    for scene_dir in sorted(
        scene_dirs
    ):

        try:

            process_scene(
                scene_dir,
                output_folder,
                aoi_geometry,
            )

        except Exception as exc:

            print(
                f"Failed "
                f"{scene_dir.name}: {exc}"
            )

    print(
        "Common satellite preprocessing complete."
    )