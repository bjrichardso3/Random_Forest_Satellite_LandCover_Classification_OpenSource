from pathlib import Path

from config import (
    RAW_DIR,
    PREPROCESSED_30M_DIR,
    ANNUAL_FEATURE_DIR,
    TARGET_GRID_DIR,
    REFERENCE_GRID_DIR,
    SAMPLE_DIR,
    MODEL_DIR,
    EVALUATION_RESULTS_DIR,
    TARGET_RESOLUTIONS,
    DEFAULT_TOTAL_SAMPLES,
    DEFAULT_MIN_SAMPLES_PER_CLASS,
    RANDOM_SEED,
    sample_name,
    model_name,
    annual_features_name,
    target_features_name,
    ceh_reference_name,
    sanitise_aoi_name,
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
    resample_reference_raster,
)

from random_forest_model.sample_reference import (
    create_reference_samples,
)

from random_forest_model.train_model import (
    train_final_model,
)

from random_forest_model.evaluate_model import (
    evaluate_sample_file,
)

# Defining function to run the training workflow for the random forest model, including downloading satellite data, preprocessing, feature extraction, resampling, sampling, training, and evaluation.
def run_training(
    aoi_name,
    aoi,
    reference,
    training_year,
    resolutions,
    total_samples,
    min_samples_per_class,
):

    ensure_directories()

    aoi_geometry = load_aoi_geometry(
        aoi
    )

    aoi_id = sanitise_aoi_name(
        aoi_name
    )

    raw_dir = (
        RAW_DIR
        / aoi_id
        / str(training_year)
    )

    preprocessed_dir = (
        PREPROCESSED_30M_DIR
        / aoi_id
        / str(training_year)
    )

    annual_feature_dir = (
        ANNUAL_FEATURE_DIR
        / aoi_id
        / str(training_year)
    )

    target_grid_dir = (
        TARGET_GRID_DIR
        / aoi_id
    )

    reference_grid_dir = (
        REFERENCE_GRID_DIR
        / aoi_id
    )

    sample_dir = (
        SAMPLE_DIR
        / aoi_id
    )

    model_dir = (
        MODEL_DIR
        / aoi_id
    )

    # 1. Satellite download

    download_satellite(
        aoi,
        raw_dir,
        training_year,
        training_year,
    )

    # 2. Common preprocessing

    preprocess_folder(
        raw_dir,
        preprocessed_dir,
        aoi,
    )

    # 3. Annual feature composite

    annual_feature_filename = annual_features_name(
        aoi_name,
        training_year,
    )

    annual_path = (
        annual_feature_dir
        / annual_feature_filename
    )

    annual_mean_features(
        preprocessed_dir,
        annual_path,
        training_year,
    )

    # 4. Resolution-specific processing

    for resolution in resolutions:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"TRAINING {resolution} m MODEL"
        )

        print(
            "=" * 70
        )

        target_feature_path = (
            target_grid_dir
            / target_features_name(
                aoi_name,
                training_year,
                resolution,
            )
        )

        reference_path = (
            reference_grid_dir
            / ceh_reference_name(
                aoi_name,
                training_year,
                resolution,
            )
        )

        sample_path = (
            sample_dir
            / sample_name(
                aoi_name,
                training_year,
                resolution,
            )
        )

        model_path = (
            model_dir
            / model_name(
                aoi_name,
                training_year,
                resolution,
            )
        )

        evaluation_dir = (
            EVALUATION_RESULTS_DIR
            / "internal"
            / aoi_id
            / str(training_year)
            / resolution_label(resolution)
        )

        # Target satellite grid

        resample_feature_raster(
            annual_path,
            target_feature_path,
            aoi_geometry,
            resolution,
        )

        # Target CEH grid

        resample_reference_raster(
            reference,
            reference_path,
            aoi_geometry,
            resolution,
        )

        # Sampling

        create_reference_samples(
            target_feature_path,
            reference_path,
            sample_path,
            total_samples=total_samples,
            min_samples_per_class=min_samples_per_class,
            external=False,
            seed=RANDOM_SEED,
        )

        # Train

        train_final_model(
            sample_path,
            model_path,
            aoi_name,
            training_year,
            resolution,
        )

        # Independent held-back test set

        evaluate_sample_file(
            sample_path,
            model_path,
            evaluation_dir,
            aoi_name,
            training_year,
            resolution,
            split="test",
            evaluation_type="internal_70_30",
        )

    print(
        "\nTRAINING WORKFLOW COMPLETE"
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Train Random Forest land-cover models "
            "for one AOI and one training year."
        )
    )

    parser.add_argument(
        "--aoi",
        required=True,
        help="Path to the area of interest shapefile.",
    )

    parser.add_argument(
        "--reference",
        required=True,
        help=(
            "Path to the authoritative CEH land-cover "
            "reference raster."
        ),
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Training year.",
    )

    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=TARGET_RESOLUTIONS,
        help="Target model resolutions in metres.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_TOTAL_SAMPLES,
        help="Total number of reference samples.",
    )

    parser.add_argument(
        "--min-per-class",
        type=int,
        default=DEFAULT_MIN_SAMPLES_PER_CLASS,
        help="Minimum target samples per occupied class.",
    )

    args = parser.parse_args()

    aoi_path = Path(args.aoi)

    if not aoi_path.exists():
        raise FileNotFoundError(
            f"AOI file not found: {aoi_path}"
        )

    reference_path = Path(args.reference)

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference raster not found: {reference_path}"
        )

    aoi_name = aoi_path.stem

    run_training(
        aoi_name=aoi_name,
        aoi=args.aoi,
        reference=args.reference,
        training_year=args.year,
        resolutions=args.resolutions,
        total_samples=args.samples,
        min_samples_per_class=args.min_per_class,
    )