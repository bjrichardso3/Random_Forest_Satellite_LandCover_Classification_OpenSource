from pathlib import Path

from config import (
    PREPROCESSED_30M_DIR,
    RAW_DIR,
    ANNUAL_FEATURE_DIR,
    TARGET_GRID_DIR,
    REFERENCE_GRID_DIR,
    SAMPLE_DIR,
    EVALUATION_RESULTS_DIR,
    sanitise_aoi_name,
    annual_features_name,
    target_features_name,
    ceh_reference_name,
    external_sample_name,
    resolution_label,
)

from common_preprocessing import (
    ensure_directories,
    load_aoi_geometry,
    load_model_bundle,
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
    resample_reference_raster,
)

from random_forest_model.sample_reference import (
    create_reference_samples,
)

from random_forest_model.evaluate_model import (
    evaluate_sample_file,
)

# Defining function to run external evaluation of the trained Random Forest Model (i.e. different year or different AOI)
def run_external_evaluation(
    aoi,
    reference,
    year,
    resolution,
    model_path,
    total_samples=20_000,
):

    bundle = load_model_bundle(
        model_path
    )

    model_year = int(
        bundle["training_year"]
    )

    model_aoi = bundle[
        "aoi_name"
    ]

    aoi_geometry = load_aoi_geometry(
        aoi
    )

    external_aoi_id = sanitise_aoi_name(
        Path(aoi).stem
    )

    raw_dir = (
        Path(PREPROCESSED_30M_DIR).parent
        / "Raw"
        / external_aoi_id
        / str(year)
    )

    preprocessed_dir = (
        PREPROCESSED_30M_DIR
        / external_aoi_id
        / str(year)
    )

    annual_dir = (
        ANNUAL_FEATURE_DIR
        / external_aoi_id
        / str(year)
    )

    ensure_directories()

    print(
        f"\nExternal evaluation"
    )

    print(
        f"Model AOI: {model_aoi}"
    )

    print(
        f"External AOI: {external_aoi_id}"
    )

    print(
        f"Model training year: {model_year}"
    )

    print(
        f"Evaluation year: {year}"
    )

    print(
        f"Resolution: {resolution} m"
    )

    # 1. Satellite data for external AOI/year.
    download_satellite(
        aoi,
        raw_dir,
        year,
        year,
    )

    # 2. Common preprocessing.
    preprocess_folder(
        raw_dir,
        preprocessed_dir,
        aoi,
    )

    # 3. Annual feature composite.
    annual_path = (
        annual_dir
        / annual_features_name(
            external_aoi_id,
            year,
        )
    )

    annual_mean_features(
        preprocessed_dir,
        annual_path,
        year,
    )

    # 4. External AOI target grids.
    feature_path = (
        TARGET_GRID_DIR
        / target_features_name(
            external_aoi_id,
            year,
            resolution,
        )
    )

    reference_path = (
        REFERENCE_GRID_DIR
        / ceh_reference_name(
            external_aoi_id,
            year,
            resolution,
        )
    )

    sample_path = (
        SAMPLE_DIR
        / external_sample_name(
            external_aoi_id,
            year,
            model_year,
            resolution,
        )
    )

    # 5. Resample satellite features.
    resample_feature_raster(
        annual_path,
        feature_path,
        aoi_geometry,
        resolution,
    )

    # 6. Resample external authoritative reference.
    resample_reference_raster(
        reference,
        reference_path,
        aoi_geometry,
        resolution,
    )

    # 7. Create independent external evaluation samples.
    create_reference_samples(
        feature_path,
        reference_path,
        sample_path,
        total_samples=total_samples,
        external=True,
    )

    # 8. Evaluate the trained model.
    evaluation_dir = (
        EVALUATION_RESULTS_DIR
        / "external"
        / str(year)
        / f"{resolution}m"
    )

    evaluate_sample_file(
        sample_path,
        model_path,
        evaluation_dir,
        external_aoi_id,
        model_year,
        resolution,
        split="external_test",
    )

    print(
        "\nEXTERNAL EVALUATION COMPLETE"
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--aoi",
        required=True,
    )

    parser.add_argument(
        "--reference",
        required=True,
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        choices=[50, 250, 1000],
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20_000,
    )

    args = parser.parse_args()

    run_external_evaluation(
        args.aoi,
        args.reference,
        args.year,
        args.resolution,
        args.model,
        args.samples,
    )