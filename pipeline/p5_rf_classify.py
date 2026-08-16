from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from config import (
    FEATURES,
    CEH_TO_SHETRAN,
    OUTPUT_NODATA,
)

from common_preprocessing import load_model_bundle

# Defining function to classify a raster as part of the pipeline using a trained Random Forest model.
def classify_raster(
    feature_raster,
    model_path,
    ceh_output,
    shetran_output,
    confidence_output=None,
    window_size=512,
):
    bundle = load_model_bundle(
        model_path
    )

    model = bundle["model"]

    model_features = bundle[
        "feature_names"
    ]

    if model_features != FEATURES:
        raise ValueError(
            "Model feature order does not "
            "match current pipeline."
        )

    ceh_output = Path(
        ceh_output
    )

    shetran_output = Path(
        shetran_output
    )

    if confidence_output:
        confidence_output = Path(
            confidence_output
        )

    with rasterio.open(
        feature_raster
    ) as src:

        profile = src.profile.copy()

        profile.update(
            {
                "count": 1,
                "dtype": "uint8",
                "nodata": OUTPUT_NODATA,
                "compress": "deflate",
            }
        )

        confidence_profile = src.profile.copy()

        confidence_profile.update(
            {
                "count": 1,
                "dtype": "float32",
                "nodata": np.nan,
                "compress": "deflate",
            }
        )

        with rasterio.open(
            ceh_output,
            "w",
            **profile,
        ) as ceh_dst, rasterio.open(
            shetran_output,
            "w",
            **profile,
        ) as shetran_dst:

            if confidence_output:

                confidence_dst = rasterio.open(
                    confidence_output,
                    "w",
                    **confidence_profile,
                )

            else:

                confidence_dst = None

            try:

                for row in range(
                    0,
                    src.height,
                    window_size,
                ):

                    for col in range(
                        0,
                        src.width,
                        window_size,
                    ):

                        height = min(
                            window_size,
                            src.height - row,
                        )

                        width = min(
                            window_size,
                            src.width - col,
                        )

                        window = Window(
                            col,
                            row,
                            width,
                            height,
                        )

                        data = src.read(
                            window=window
                        ).astype(
                            "float32"
                        )

                        h, w = (
                            data.shape[1],
                            data.shape[2],
                        )

                        flat = data.reshape(
                            len(FEATURES),
                            -1,
                        ).T

                        valid = np.isfinite(
                            flat
                        ).all(axis=1)

                        ceh = np.full(
                            h * w,
                            OUTPUT_NODATA,
                            dtype="uint8",
                        )

                        shetran = np.full(
                            h * w,
                            OUTPUT_NODATA,
                            dtype="uint8",
                        )

                        confidence = np.full(
                            h * w,
                            np.nan,
                            dtype="float32",
                        )

                        if valid.any():

                            # Convert the valid feature array to a DataFrame using the exact feature names and order used when the Random Forest was trained.
                            X_predict = pd.DataFrame(
                                flat[valid],
                                columns=model_features,
                            )

                            predictions = model.predict(
                                X_predict
                            ).astype(
                                "uint8"
                            )

                            ceh[valid] = (
                                predictions
                            )

                            for (
                                ceh_id,
                                shetran_id,
                            ) in CEH_TO_SHETRAN.items():

                                shetran[
                                    ceh
                                    == ceh_id
                                ] = (
                                    shetran_id
                                )

                            if hasattr(
                                model,
                                "predict_proba",
                            ):

                                probabilities = (
                                    model.predict_proba(
                                        X_predict
                                    )
                                )

                                confidence[
                                    valid
                                ] = (
                                    probabilities.max(
                                        axis=1
                                    )
                                )

                        ceh_dst.write(
                            ceh.reshape(
                                h,
                                w,
                            ),
                            1,
                            window=window,
                        )

                        shetran_dst.write(
                            shetran.reshape(
                                h,
                                w,
                            ),
                            1,
                            window=window,
                        )

                        if confidence_dst:
                            confidence_dst.write(
                                confidence.reshape(
                                    h,
                                    w,
                                ),
                                1,
                                window=window,
                            )

            finally:

                if confidence_dst:
                    confidence_dst.close()

    print(
        f"CEH classification: {ceh_output}"
    )

    print(
        f"SHETRAN classification: "
        f"{shetran_output}"
    )

    if confidence_output:
        print(
            f"Confidence: "
            f"{confidence_output}"
        )