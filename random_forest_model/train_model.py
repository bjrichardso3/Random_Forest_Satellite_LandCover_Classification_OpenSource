from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
)

from config import (
    FEATURES,
    RANDOM_SEED,
    RF_N_ESTIMATORS,
    RF_MAX_FEATURES,
    RF_MIN_SAMPLES_LEAF,
    RF_CLASS_WEIGHT,
    RF_N_JOBS,
    CV_FOLDS,
)

# Defining function to make random forest classifier with specified hyperparameters from config.py.
def make_rf():

    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_features=RF_MAX_FEATURES,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight=RF_CLASS_WEIGHT,
        bootstrap=True,
        oob_score=True,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_SEED,
    )

# Defining function to perform cross-validation on the random forest model using spatial blocks as groups.
# Checks whether spatial cross-validation is possible.
# Spatial CV requires at least two distinct spatial blocks. 
# Small homogeneous AOIs may contain only one block, in which case spatial CV is not possible.
def cross_validate_rf(
    X,
    y,
    groups,
    n_splits=CV_FOLDS,
):
    unique_groups = np.unique(
        groups
    )

    n_splits = min(
        n_splits,
        len(unique_groups),
    )

    if n_splits < 2:
        print(
            "\nWARNING: Spatial cross-validation cannot be performed."
        )
        print(
            f"Only {len(unique_groups)} spatial block(s) are available."
        )
        print(
            "At least two spatial blocks are required."
        )
        print(
            "Spatial CV will be skipped for this AOI."
        )

        return pd.DataFrame(
            columns=[
                "fold",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "kappa",
            ]
        )

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    results = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        model = make_rf()

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        predictions = model.predict(
            X.iloc[test_idx]
        )

        results.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(
                    y.iloc[test_idx],
                    predictions,
                ),
                "balanced_accuracy":
                    balanced_accuracy_score(
                        y.iloc[test_idx],
                        predictions,
                    ),
                "macro_f1":
                    f1_score(
                        y.iloc[test_idx],
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                "weighted_f1":
                    f1_score(
                        y.iloc[test_idx],
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                "kappa":
                    cohen_kappa_score(
                        y.iloc[test_idx],
                        predictions,
                    ),
            }
        )

        print(
            f"CV fold {fold}: "
            f"accuracy="
            f"{results[-1]['accuracy']:.4f}, "
            f"balanced accuracy="
            f"{results[-1]['balanced_accuracy']:.4f}, "
            f"macro F1="
            f"{results[-1]['macro_f1']:.4f}"
        )

    return pd.DataFrame(
        results
    )

# Defining function to train the final random forest model on all training samples.
def train_final_model(
    sample_path,
    model_path,
    aoi_name,
    training_year,
    resolution,
):
    sample_path = Path(
        sample_path
    )

    model_path = Path(
        model_path
    )

    df = pd.read_parquet(
        sample_path
    )

    train = df[
        df["split"] == "train"
    ].copy()

    if train.empty:
        raise ValueError(
            "No training samples found."
        )

    missing = [
        f
        for f in FEATURES
        if f not in train.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    X = train[
        FEATURES
    ]

    y = train[
        "label"
    ].astype(int)

    groups = train[
        "block_id"
    ].astype(str)

    print(
        f"Training samples: {len(train):,}"
    )

    cv_results = cross_validate_rf(
        X,
        y,
        groups,
    )

    model = make_rf()

    model.fit(
        X,
        y,
    )

    bundle = {
        "model": model,
        "feature_names": FEATURES,
        "aoi_name": str(aoi_name),
        "training_year": int(training_year),
        "resolution": int(resolution),
        "random_seed": RANDOM_SEED,
        "algorithm": "RandomForestClassifier",
        "parameters": {
            "n_estimators": RF_N_ESTIMATORS,
            "max_features": RF_MAX_FEATURES,
            "min_samples_leaf":
                RF_MIN_SAMPLES_LEAF,
            "class_weight":
                RF_CLASS_WEIGHT,
            "bootstrap": True,
            "oob_score": True,
        },
        "training_samples": int(
            len(train)
        ),
        "training_class_counts": {
            str(k): int(v)
            for k, v in y.value_counts().items()
        },

        "cv_results": cv_results.to_dict(
            orient="records"
        ),

        "cv_mean": {
            column: float(
                cv_results[column].mean()
            )
            for column in cv_results.columns
            if column != "fold"
        },
        "oob_score": float(
            model.oob_score_
        ),
    }

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        model_path,
    )

    metadata_path = model_path.with_suffix(
        ".json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            bundle,
            f,
            indent=2,
            default=str,
        )

    cv_path = model_path.with_name(
        model_path.stem
        + "_cv.csv"
    )

    cv_results.to_csv(
        cv_path,
        index=False,
    )

    print(
        f"\nFinal model saved: {model_path}"
    )

    print(
        f"OOB accuracy: "
        f"{model.oob_score_:.4f}"
    )

    return bundle