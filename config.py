from pathlib import Path
import re

# Project Paths

# Project/code location on the network drive.
PROJECT_ROOT = Path(r"B:\Projects\Random Forest Satellite Land Cover Classification")

SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
RESULTS_ROOT = PROJECT_ROOT / "Results"

PIPELINE_ROOT = SCRIPTS_ROOT / "Pipeline"
RF_ROOT = SCRIPTS_ROOT / "Random Forest Model"


# Large Data Location: Raw satellite scenes and intermediate rasters are located on the Convex research data warehouse.

DATA_ROOT = Path(r"P:\Random_Forest_Land_Cover_Classification\Data")

RAW_DIR = DATA_ROOT / "Raw"
PREPROCESSED_30M_DIR = DATA_ROOT / "Preprocessed_30m"
ANNUAL_FEATURE_DIR = DATA_ROOT / "Annual_features"
TARGET_GRID_DIR = DATA_ROOT / "Target_grids"
REFERENCE_GRID_DIR = DATA_ROOT / "Reference_grids"
SAMPLE_DIR = DATA_ROOT / "Samples"



# Model/ Results

MODEL_DIR = SCRIPTS_ROOT / "Models"

TRAINING_RESULTS_DIR = RESULTS_ROOT / "Training"
EVALUATION_RESULTS_DIR = RESULTS_ROOT / "Evaluation"
CLASSIFICATION_RESULTS_DIR = RESULTS_ROOT / "Classifications"


# Planetary Computer STAC API

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

COLLECTIONS = [
    "landsat-c2-l2",
    "sentinel-2-l2a",
]


# Temporal Window (Currently set as summer months as default but this can be changed/ customized)

DEFAULT_START_MONTH = 6
DEFAULT_START_DAY = 1

DEFAULT_END_MONTH = 8
DEFAULT_END_DAY = 31


# Common Satellite Grid set as British National Grid (EPSG:27700) with 30m resolution.

# Satellite observations are first harmonised onto this grid.
COMMON_SATELLITE_RESOLUTION = 30
COMMON_CRS = "EPSG:27700"


# Final classification resolutions of 50m, 250m, and 1000m as default. This can be changed/ customized.
TARGET_RESOLUTIONS = [50, 250, 1000]

# Spatial sampling block. Set at 10km as default to provide spatial independence. This can be changed/ customized.
SPATIAL_BLOCK_SIZE = 10_000


# Random Seeds for reproducibility. Currently set to 42 as default. This can be change/ customized or have other random seeds added.

RANDOM_SEED = 42


# Sampling. Setting default total samples to 20,000 and minimum samples per CEH class to 100. This can be changed/ customized.

DEFAULT_TOTAL_SAMPLES = 20_000

# Minimum number of samples per CEH class where that class exists.
DEFAULT_MIN_SAMPLES_PER_CLASS = 100

TRAIN_FRACTION = 0.70


# Random Forest Configuration of Hyperparameters.

RF_N_ESTIMATORS = 500
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_LEAF = 1

# Balanced subsampling: useful where rare classes remain rare despite the proportional sampling strategy.
RF_CLASS_WEIGHT = "balanced_subsample"

RF_N_JOBS = -1


# Cross Validation Specification. Default is set to 5-fold cross validation.

CV_FOLDS = 5


# Feature Names

FEATURES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "ndvi",
    "nbr",
    "ndwi",
    "savi",
    "evi",
    "water_fraction",
]


# Landsat Band Mapping

LANDSAT_BAND_MAP = {
    "landsat-7": {
        "bands": {
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir": "nir08",
            "swir1": "swir16",
            "swir2": "swir22",
            "qa": "qa_pixel",
        }
    },
    "landsat-8": {
        "bands": {
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir": "nir08",
            "swir1": "swir16",
            "swir2": "swir22",
            "qa": "qa_pixel",
        }
    },
    "landsat-9": {
        "bands": {
            "blue": "blue",
            "green": "green",
            "red": "red",
            "nir": "nir08",
            "swir1": "swir16",
            "swir2": "swir22",
            "qa": "qa_pixel",
        }
    },
}


# Sentinel-2 Band Mapping

SENTINEL_BANDS = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "swir1": "B11",
    "swir2": "B12",
    "qa": "SCL",
}


# Masking Rules for Cloud and Water Detection

# Landsat QA_PIXEL:
# bit 1 = dilated cloud
# bit 2 = cirrus
# bit 3 = cloud
# bit 4 = cloud shadow
# bit 7 = water
LANDSAT_CLOUD_BITS = [1, 2, 3, 4]
LANDSAT_WATER_BIT = 7


# Sentinel-2 SCL:
# 0 = No data
# 1 = Saturated/defective
# 3 = Cloud shadow
# 8 = Cloud medium probability
# 9 = Cloud high probability
# 10 = Thin cirrus

# Water = 6 and is NOT masked.
# Snow/ice = 11 and is also retained (more so for generalizability to other seasons or regions).
SENTINEL_INVALID_CLASSES = [0, 1]
SENTINEL_CLOUD_CLASSES = [3, 8, 9, 10]
SENTINEL_WATER_CLASS = 6


# CEH → Shetran Reclassification Scheme

CEH_TO_SHETRAN = {
    1: 4,
    2: 5,
    3: 1,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 6,
    9: 6,
    10: 6,
    11: 6,
    12: 2,
    13: 8,
    14: 8,
    15: 2,
    16: 2,
    17: 2,
    18: 2,
    19: 6,
    20: 7,
    21: 7,
}

CEH_NODATA_VALUES = [0, 255]
OUTPUT_NODATA = 255


# Output File Naming Conventions

# Definig function to convert an area of interest name into a safe identifier
def sanitise_aoi_name(aoi_name):
    """
    Convert an AOI/catchment name into a deterministic, filesystem-safe
    identifier suitable for filenames and directory names.
    """

    if aoi_name is None:
        raise ValueError("aoi_name must be provided.")

    name = str(aoi_name).strip()

    if not name:
        raise ValueError("aoi_name cannot be empty.")

    name = re.sub(
        r"[\s\-/\\]+",
        "_",
        name,
    )

    name = "".join(
        character
        for character in name
        if character.isalnum() or character == "_"
    )

    name = name.strip("_")

    if not name:
        raise ValueError(
            f"AOI name {aoi_name!r} does not contain any "
            "valid filename characters."
        )

    return name

# Defining function to return the resolution for use in file names
def resolution_label(resolution):
    """
    Examples:
        50 -> '50m'
        250 -> '250m'
        1000 -> '1000m'
    """

    return f"{int(resolution)}m"

# Setting the file name for the annual feature composite
def annual_features_name(aoi_name, year):

    aoi = sanitise_aoi_name(aoi_name)

    return (
        f"annual_features_"
        f"{aoi}_"
        f"{int(year)}.tif"
    )

# Setting the file name for the feature stack on the target grid
def target_features_name(
    aoi_name,
    year,
    resolution,
):

    aoi = sanitise_aoi_name(aoi_name)

    resolution = resolution_label(resolution)

    return (
        f"annual_features_"
        f"{aoi}_"
        f"{int(year)}_"
        f"{resolution}.tif"
    )

# Setting the file name for the CEH reference raster (authoritative land cover)
def ceh_reference_name(
    aoi_name,
    year,
    resolution,
):

    aoi = sanitise_aoi_name(aoi_name)

    resolution = resolution_label(resolution)

    return (
        f"ceh_"
        f"{aoi}_"
        f"{int(year)}_"
        f"{resolution}.tif"
    )

# Setting the file name of spatially/ categorically stratified sample dataset
def sample_name(
    aoi_name,
    year,
    resolution,
):

    aoi = sanitise_aoi_name(aoi_name)

    resolution = resolution_label(resolution)

    return (
        f"samples_"
        f"{aoi}_"
        f"{int(year)}_"
        f"{resolution}.parquet"
    )

def external_sample_name(
    aoi_name,
    evaluation_year,
    model_year,
    resolution,
):
    aoi = sanitise_aoi_name(aoi_name)
    resolution = resolution_label(resolution)

    return (
        f"external_samples_"
        f"{aoi}_"
        f"{int(evaluation_year)}_"
        f"{int(model_year)}_"
        f"{resolution}.parquet"
    )

# Setting random forest model file name
# The year identifies the reference year used for model training
def model_name(
    aoi_name,
    year,
    resolution,
):

    aoi = sanitise_aoi_name(aoi_name)

    resolution = resolution_label(resolution)

    return (
        f"rf_ceh_"
        f"{aoi}_"
        f"{int(year)}_"
        f"{resolution}.joblib"
    )

# Setting the file name of the classified raster. classification_level should normally be CEH, SHETRAN, or confidence
def classification_name(
    classification_level,
    aoi_name,
    year,
    resolution,
):

    aoi = sanitise_aoi_name(aoi_name)

    resolution = resolution_label(resolution)

    return (
        f"classification_"
        f"{classification_level}_"
        f"{aoi}_"
        f"{int(year)}_"
        f"{resolution}.tif"
    )

# Defining function to set the prefix for evaluation file names
def evaluation_prefix(
    evaluation_type,
    aoi_name,
    model_year,
    resolution,
    evaluation_year=None,
):
    """
    Prefix for evaluation products.

    Internal 70/30 evaluation:
        internal_70_30_CEH_Coalburn_2022_50m

    External temporal/spatial evaluation:
        external_2023_CEH_Coalburn_2022_50m

    model_year is the year used to train the model.
    evaluation_year is the year of the independent reference data when an external evaluation is performed.
    """

    aoi = sanitise_aoi_name(aoi_name)

    resolution = resolution_label(resolution)

    if evaluation_type == "internal_70_30":

        return (
            f"internal_70_30_"
            f"{aoi}_"
            f"{int(model_year)}_"
            f"{resolution}"
        )

    if evaluation_type == "external":

        if evaluation_year is None:
            raise ValueError(
                "evaluation_year is required "
                "for external evaluation."
            )

        return (
            f"external_"
            f"{int(evaluation_year)}_"
            f"{aoi}_"
            f"{int(model_year)}_"
            f"{resolution}"
        )

    raise ValueError(
        f"Unknown evaluation_type: "
        f"{evaluation_type}"
    )