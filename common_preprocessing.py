from pathlib import Path
import json
import math
import re

import geopandas as gpd
import numpy as np
import rasterio
import joblib
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from shapely.geometry import mapping
from shapely.ops import unary_union

#Importing Directories that are defined in config.py and ensuring all exist
def ensure_directories():
    from config import (
        DATA_ROOT,
        RAW_DIR,
        PREPROCESSED_30M_DIR,
        ANNUAL_FEATURE_DIR,
        TARGET_GRID_DIR,
        REFERENCE_GRID_DIR,
        SAMPLE_DIR,
        MODEL_DIR,
        RESULTS_ROOT,
        TRAINING_RESULTS_DIR,
        EVALUATION_RESULTS_DIR,
        CLASSIFICATION_RESULTS_DIR,
    )

    directories = [
        DATA_ROOT,
        RAW_DIR,
        PREPROCESSED_30M_DIR,
        ANNUAL_FEATURE_DIR,
        TARGET_GRID_DIR,
        REFERENCE_GRID_DIR,
        SAMPLE_DIR,
        MODEL_DIR,
        RESULTS_ROOT,
        TRAINING_RESULTS_DIR,
        EVALUATION_RESULTS_DIR,
        CLASSIFICATION_RESULTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Defining function to load in Area of Interest (AOI) geometry from vector or raster data
def load_aoi_geometry(aoi_path):
    aoi_path = Path(aoi_path)

    if not aoi_path.exists():
        raise FileNotFoundError(aoi_path)

    suffix = aoi_path.suffix.lower()

    if suffix in {".shp", ".gpkg", ".geojson", ".json", ".gml"}:

        gdf = gpd.read_file(aoi_path)

        if gdf.empty:
            raise ValueError("AOI vector contains no features.")

        if gdf.crs is None:
            raise ValueError("AOI vector has no CRS.")

        gdf = gdf.to_crs("EPSG:27700")

        geometry = unary_union(gdf.geometry)

        if geometry.is_empty:
            raise ValueError("AOI geometry is empty.")

        return geometry

    if suffix in {".tif", ".tiff"}:

        from shapely.geometry import shape

        with rasterio.open(aoi_path) as src:

            if src.crs is None:
                raise ValueError("AOI raster has no CRS.")

            data = src.read(1)

            if src.nodata is None:
                valid = data != 0
            else:
                valid = (
                    (data != src.nodata)
                    & (data != 0)
                )

            if not valid.any():
                raise ValueError(
                    "AOI raster contains no valid non-zero pixels."
                )

            polygons = []

            for geom, value in features.shapes(
                data.astype("uint8"),
                mask=valid,
                transform=src.transform,
            ):
                if value != 0:
                    polygons.append(shape(geom))

            if not polygons:
                raise ValueError(
                    "Could not construct AOI from raster."
                )

            geometry = unary_union(polygons)

            if src.crs.to_epsg() != 27700:
                geometry = gpd.GeoSeries(
                    [geometry],
                    crs=src.crs,
                ).to_crs("EPSG:27700").iloc[0]

            return geometry

    raise ValueError(
        f"Unsupported AOI format: {suffix}"
    )

# Create a target grid. Uses British National Grid (EPSG:27700) as the common satellite grid with 30m resolution. 
def create_target_grid(aoi_geometry, resolution):

    if resolution <= 0:
        raise ValueError("Resolution must be positive.")

    minx, miny, maxx, maxy = aoi_geometry.bounds

    left = math.floor(minx / resolution) * resolution
    bottom = math.floor(miny / resolution) * resolution

    right = math.ceil(maxx / resolution) * resolution
    top = math.ceil(maxy / resolution) * resolution

    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))

    transform = from_origin(
        left,
        top,
        resolution,
        resolution,
    )

    return {
        "crs": "EPSG:27700",
        "transform": transform,
        "width": width,
        "height": height,
        "resolution": resolution,
        "bounds": (left, bottom, right, top),
    }


def geometry_mask_for_grid(aoi_geometry, grid):
    return features.geometry_mask(
        [mapping(aoi_geometry)],
        transform=grid["transform"],
        invert=True,
        out_shape=(grid["height"], grid["width"]),
    )


def write_grid_metadata(path, grid):
    path = Path(path)

    data = {
        "crs": grid["crs"],
        "resolution": grid["resolution"],
        "width": grid["width"],
        "height": grid["height"],
        "bounds": list(grid["bounds"]),
        "transform": list(grid["transform"]),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# Defining function to extract the date from the Landsat/ Sentinel Scene Identifiers (YYYMMDD)
def extract_date_from_name(name):

    match = re.search(r"(20\d{6})", name)

    if match:
        value = match.group(1)

        year = int(value[:4])
        month = int(value[4:6])
        day = int(value[6:8])

        if 1 <= month <= 12 and 1 <= day <= 31:
            return value

    return None

# Defining functinon to determine the sensor type (landsat-7, landsat-8, landsat-9, sentinel-2)
def get_sensor_type(scene_name):
    name = scene_name.upper()

    if "LE07" in name or "LANDSAT_7" in name:
        return "landsat-7"

    if "LC08" in name or "LANDSAT_8" in name:
        return "landsat-8"

    if "LC09" in name or "LANDSAT_9" in name:
        return "landsat-9"

    if "S2A" in name or "S2B" in name or "SENTINEL" in name:
        return "sentinel-2"

    raise ValueError(f"Unknown sensor: {scene_name}")

# Defining function to calculate a safe normalized difference (insulated against division by zero and NaN values)
def safe_normalized_difference(a, b):
    numerator = a - b
    denominator = a + b

    result = np.full_like(
        numerator,
        np.nan,
        dtype="float32",
    )

    valid = np.isfinite(numerator) & np.isfinite(denominator)
    valid &= denominator != 0

    result[valid] = (
        numerator[valid] /
        denominator[valid]
    )

    return result

# Defining function to calculate vegetation indices (NDVI, NBR, NDWI, SAVI, EVI) from the satellite bands
def calculate_indices(data):

    blue = data[0]
    green = data[1]
    red = data[2]
    nir = data[3]
    swir1 = data[4]
    swir2 = data[5]

    ndvi = safe_normalized_difference(nir, red)
    nbr = safe_normalized_difference(nir, swir2)
    ndwi = safe_normalized_difference(green, nir)

    eps = 1e-10

    savi_den = nir + red + 0.5 + eps

    savi = np.full_like(
        nir,
        np.nan,
        dtype="float32",
    )

    valid = np.isfinite(savi_den) & (savi_den != 0)

    savi[valid] = (
        ((nir[valid] - red[valid]) * 1.5)
        / savi_den[valid]
    )

    evi_den = (nir + 6 * red - 7.5 * blue + 1 + eps)

    evi = np.full_like(nir, np.nan, dtype="float32")

    valid = np.isfinite(evi_den) & (evi_den != 0)

    evi[valid] = (2.5 * ((nir[valid] - red[valid]) / evi_den[valid]))

    return np.stack([ndvi, nbr, ndwi, savi, evi]).astype("float32")

# Defining function to determine the number of samples to allocate to each class based on the total number of samples, the counts of each class, and a minimum number of samples per class.
def allocate_proportional(total, counts, minimum):

    counts = np.asarray(counts, dtype=int)

    available = counts.sum()

    if available == 0:
        return np.zeros_like(counts)

    total = min(int(total), int(available))

    if total < minimum * np.count_nonzero(counts):
        minimum = max(
            1,
            total // max(1, np.count_nonzero(counts))
        )

    proportions = counts / available

    allocation = np.floor(
        proportions * total
    ).astype(int)

    existing = counts > 0

    allocation[existing] = np.maximum(
        allocation[existing],
        np.minimum(minimum, counts[existing]),
    )

    allocation = np.minimum(
        allocation,
        counts,
    )

    # Remove excess.
    while allocation.sum() > total:

        candidates = np.where(
            allocation > np.minimum(minimum, counts)
        )[0]

        if len(candidates) == 0:
            break

        idx = candidates[
            np.argmax(allocation[candidates])
        ]

        allocation[idx] -= 1

    # Add remaining samples according to abundance.
    while allocation.sum() < total:

        candidates = np.where(
            allocation < counts
        )[0]

        if len(candidates) == 0:
            break

        remaining = counts[candidates] - allocation[candidates]

        idx = candidates[
            np.argmax(remaining)
        ]

        allocation[idx] += 1

    return allocation


def load_model_bundle(model_path):

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError(
            "Model file is not a model bundle."
        )

    if "model" not in bundle:
        raise ValueError(
            "Model bundle does not contain 'model'."
        )

    return bundle