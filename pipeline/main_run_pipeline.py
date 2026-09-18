from pathlib import Path

from config import (
    RAW_DIR,
    PREPROCESSED_30M_DIR,
    ANNUAL_FEATURE_DIR,
    TARGET_GRID_DIR,
    MODEL_DIR,
    CLASSIFICATION_RESULTS_DIR,
    TARGET_RESOLUTIONS,
    sanitise_aoi_name,
    annual_features_name,
    target_features_name,
    model_name,
    classification_name,
    resolution_label,
)

from common_preprocessing import (
    ensure_directories,
    load_aoi_geometry,
)

from pipeline.p1_download_satellite import (
    download_satellite,
)

from pipeline.p2_preprocess import (
    preprocess_folder,
)

from pipeline.p3_annual_features import (
    annual_mean_features,
)

from pipeline.p4_target_grid import (
    resample_feature_raster,
)

from pipeline.p5_rf_classify import (
    classify_raster,
)

# Defining function to run the entire pipeline
def run_pipeline(
    aoi,
    start_year,
    end_year,
    resolution,
    model_path=None,
):
    ensure_directories()

    aoi_geometry = load_aoi_geometry(
        aoi
    )

    # The AOI identifier should be explicitly derived from
    # the AOI filename unless a separate identifier is supplied.
    aoi_id = sanitise_aoi_name(
        Path(aoi).stem
    )

    # Raw satellite scenes may be shared between AOIs and are organised only by acquisition year
    raw_dir = (
        RAW_DIR
    )

    # AOI-specific data directories.

    preprocessed_dir = (
        PREPROCESSED_30M_DIR
        / aoi_id
    )

    annual_feature_dir = (
        ANNUAL_FEATURE_DIR
        / aoi_id
    )

    target_grid_dir = (
        TARGET_GRID_DIR
        / aoi_id
    )

    if model_path is None:
        raise ValueError(
            "Model_path must be supplied for classification."
        )

    model_path = Path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    # 1. Download satellite data

    for year in range(
        start_year,
        end_year + 1,
    ):

        year_raw_dir = (
            RAW_DIR
            / str(year)
        )

        download_satellite(
            aoi,
            year_raw_dir,
            year,
            year,
        )

    # 2. Common preprocessing

    for year in range(
        start_year,
        end_year + 1,
    ):

        year_raw_dir = (
            RAW_DIR
            / str(year)
        )

        year_preprocessed_dir = (
            preprocessed_dir
            / str(year)
        )

        preprocess_folder(
            year_raw_dir,
            year_preprocessed_dir,
            aoi,
        )

    # 3. Annual features and classification

    for year in range(
        start_year,
        end_year + 1,
    ):

        print(
            f"CLASSIFICATION YEAR: {year}"
        )

        year_preprocessed_dir = (
            preprocessed_dir
            / str(year)
        )

        annual_path = (
            annual_feature_dir
            / str(year)
            / annual_features_name(
                aoi_id,
                year,
            )
        )

        annual_mean_features(
            year_preprocessed_dir,
            annual_path,
            year,
        )

        target_features = (
            target_grid_dir
            / target_features_name(
                aoi_id,
                year,
                resolution,
            )
        )

        resample_feature_raster(
            annual_path,
            target_features,
            aoi_geometry,
            resolution,
        )

        result_dir = (
            CLASSIFICATION_RESULTS_DIR
            / aoi_id
            / str(year)
            / resolution_label(resolution)
        )

        result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        ceh_output = (
            result_dir
            / classification_name(
                "CEH",
                aoi_id,
                year,
                resolution,
            )
        )

        shetran_output = (
            result_dir
            / classification_name(
                "SHETRAN",
                aoi_id,
                year,
                resolution,
            )
        )

        confidence_output = (
            result_dir
            / classification_name(
                "confidence",
                aoi_id,
                year,
                resolution,
            )
        )

        classify_raster(
            target_features,
            model_path,
            ceh_output,
            shetran_output,
            confidence_output,
        )

    print(
        "\nPIPELINE COMPLETE"
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--aoi",
        required=True,
    )

    parser.add_argument(
        "--start-year",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--end-year",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        choices=TARGET_RESOLUTIONS,
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    args = parser.parse_args()

    run_pipeline(
        args.aoi,
        args.start_year,
        args.end_year,
        args.resolution,
        args.model,
    )

# Ensure environment is activated before running the script: micromamba activate sat_pipeline
