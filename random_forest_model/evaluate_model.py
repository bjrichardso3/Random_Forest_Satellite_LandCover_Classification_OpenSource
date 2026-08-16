from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from config import (
    FEATURES,
    CEH_TO_SHETRAN,
    OUTPUT_NODATA,
    evaluation_prefix,
)


# Defining function to calculate class-level metrics (precision, recall, f1-score, support, IoU) from true and predicted labels.
def calculate_class_metrics(
    y_true,
    y_pred,
    labels,
):
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    rows = []

    for i, label in enumerate(labels):

        tp = cm[i, i]

        fn = (
            cm[i, :].sum()
            - tp
        )

        fp = (
            cm[:, i].sum()
            - tp
        )

        denominator = (
            tp + fp + fn
        )

        iou = (
            tp / denominator
            if denominator > 0
            else 0
        )

        rows.append(
            {
                "class": int(label),
                "precision":
                    report[str(label)][
                        "precision"
                    ],
                "recall":
                    report[str(label)][
                        "recall"
                    ],
                "f1":
                    report[str(label)][
                        "f1-score"
                    ],
                "support":
                    report[str(label)][
                        "support"
                    ],
                "IoU": iou,
            }
        )

    return pd.DataFrame(
        rows
    )

# Defining function to plot confusion matrix
def plot_confusion_matrix(
    cm,
    labels,
    output_path,
    title,
    normalize=False,
):
    if normalize:

        denominator = cm.sum(
            axis=1,
            keepdims=True,
        )

        matrix = np.divide(
            cm,
            denominator,
            out=np.zeros_like(
                cm,
                dtype=float,
            ),
            where=denominator != 0,
        )

    else:

        matrix = cm

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(labels) * 0.5),
            max(7, len(labels) * 0.5),
        )
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest",
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        xticks=np.arange(
            len(labels)
        ),
        yticks=np.arange(
            len(labels)
        ),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Reference class",
        xlabel="Predicted class",
        title=title,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

# Defining function to evaluate predictions by calculating overall metrics
def evaluate_predictions(
    y_true,
    y_pred,
    labels,
    output_prefix,
    level_name,
):
    output_prefix = Path(
        output_prefix
    )

    output_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = {
        "level": level_name,
        "n_samples": int(
            len(y_true)
        ),
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
        "kappa":
            float(
                cohen_kappa_score(
                    y_true,
                    y_pred,
                )
            ),
    }

    with open(
        output_prefix.with_suffix(
            ".json"
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    cm_df.index.name = "reference"
    cm_df.columns.name = "prediction"

    cm_df.to_csv(
        output_prefix.with_name(
            output_prefix.stem
            + "_confusion_matrix.csv"
        )
    )

    class_metrics = calculate_class_metrics(
        y_true,
        y_pred,
        labels,
    )

    class_metrics.to_csv(
        output_prefix.with_name(
            output_prefix.stem
            + "_class_metrics.csv"
        ),
        index=False,
    )

    plot_confusion_matrix(
        cm,
        labels,
        output_prefix.with_name(
            output_prefix.stem
            + "_confusion_matrix.png"
        ),
        f"{level_name} confusion matrix",
        normalize=False,
    )

    plot_confusion_matrix(
        cm,
        labels,
        output_prefix.with_name(
            output_prefix.stem
            + "_confusion_matrix_normalised.png"
        ),
        f"{level_name} normalised confusion matrix",
        normalize=True,
    )

    return metrics


def evaluate_sample_file(
    sample_path,
    model_path,
    output_directory,
    aoi_name,
    model_year,
    resolution,
    split="test",
    evaluation_type="internal_70_30",
    evaluation_year=None,
):
    
    sample_path = Path(
        sample_path
    )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = joblib.load(
        model_path
    )

    model = bundle["model"]

    feature_names = bundle[
        "feature_names"
    ]

    df = pd.read_parquet(
        sample_path
    )

    evaluation = df[
        df["split"] == split
    ].copy()

    if evaluation.empty:
        raise ValueError(
            f"No samples with split={split}"
        )

    X = evaluation[
        feature_names
    ]

    y_true = evaluation[
        "label"
    ].astype(int)

    y_pred = model.predict(
        X
    ).astype(int)

    # Convert predictions to a Series using the evaluation
    # dataframe index so that true and predicted labels remain
    # correctly aligned during reclassification.
    y_pred = pd.Series(
        y_pred,
        index=evaluation.index,
        name="prediction",
    )

    labels = sorted(
        set(
            y_true.unique()
        )
        | set(
            y_pred.unique()
        )
    )

    if split == "external_test":
        evaluation_type = "external"
    else:
        evaluation_type = "internal_70_30"

    prefix = (
        output_directory
        / (
            evaluation_prefix(
                evaluation_type,
                aoi_name,
                model_year,
                resolution,
            )
            + "_CEH"
        )
    )

    ceh_metrics = evaluate_predictions(
        y_true,
        y_pred,
        labels,
        prefix,
        "CEH",
    )

    # Reclassification to SHETRAN classes

    y_true_shetran = y_true.map(
        CEH_TO_SHETRAN
    )

    y_pred_shetran = y_pred.map(
        CEH_TO_SHETRAN
    )

    # Remove samples for which a CEH class has no defined
    # SHETRAN equivalent.
    valid = (
        y_true_shetran.notna()
        & y_pred_shetran.notna()
    )

    y_true_shetran = (
        y_true_shetran.loc[valid]
        .astype(int)
    )

    y_pred_shetran = (
        y_pred_shetran.loc[valid]
        .astype(int)
    )

    shetran_labels = sorted(
        set(
            y_true_shetran.unique()
        )
        | set(
            y_pred_shetran.unique()
        )
    )

    shetran_metrics = evaluate_predictions(
        y_true_shetran,
        y_pred_shetran,
        shetran_labels,
        output_directory
        / (
            evaluation_prefix(
                evaluation_type,
                aoi_name,
                model_year,
                resolution,
            )
            + "_SHETRAN"
        ),
        "SHETRAN",
    )

    summary = {
        "CEH": ceh_metrics,
        "SHETRAN": shetran_metrics,
        "model": str(model_path),
        "sample_file": str(sample_path),
        "aoi_name": str(aoi_name),
        "model_year": int(model_year),
        "resolution_m": int(resolution),
        "evaluation_type": evaluation_type,
        "evaluation_year": (
            int(evaluation_year)
            if evaluation_year is not None
            else None
        ),
        "split": split,
    }

    summary_path = (
        output_directory
        / (
            evaluation_prefix(
                evaluation_type,
                aoi_name,
                model_year,
                resolution,
                evaluation_year=evaluation_year,
            )
            + "_summary.json"
        )
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\nCEH evaluation:"
    )

    for key, value in ceh_metrics.items():

        if key not in {
            "level",
            "n_samples",
        }:

            print(
                f"  {key}: "
                f"{value:.4f}"
            )

    print(
        "\nSHETRAN evaluation:"
    )

    for key, value in shetran_metrics.items():

        if key not in {
            "level",
            "n_samples",
        }:

            print(
                f"  {key}: "
                f"{value:.4f}"
            )

    return summary


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Random Forest model "
            "against a reference sample parquet file."
        )
    )

    parser.add_argument(
        "--samples",
        required=True,
        help="Path to the reference sample parquet file.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained Random Forest .joblib file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory for evaluation outputs.",
    )

    parser.add_argument(
        "--aoi-name",
        required=True,
        help="AOI name used for evaluation output naming.",
    )

    parser.add_argument(
        "--model-year",
        type=int,
        required=True,
        help="Training year of the Random Forest model.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="Target raster resolution in metres.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=[
            "train",
            "test",
            "external_test",
        ],
        help="Sample split to evaluate.",
    )

    args = parser.parse_args()

    evaluate_sample_file(
        sample_path=args.samples,
        model_path=args.model,
        output_directory=args.output,
        aoi_name=args.aoi_name,
        model_year=args.model_year,
        resolution=args.resolution,
        split=args.split,
    )