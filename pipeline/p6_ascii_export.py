from pathlib import Path

import rasterio

# Defining function to convert a raster to ASCII format for use in SHETRAN
def raster_to_ascii(
    input_raster,
    output_ascii,
):
    input_raster = Path(
        input_raster
    )

    output_ascii = Path(
        output_ascii
    )

    output_ascii.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        input_raster
    ) as src:

        profile = src.profile.copy()

        profile.update(
            {
                "driver": "AAIGrid",
            }
        )

        with rasterio.open(
            output_ascii,
            "w",
            **profile,
        ) as dst:

            dst.write(
                src.read(1),
                1,
            )

    print(
        f"Exported: {output_ascii}"
    )


def export_folder(
    input_folder,
    output_folder,
):
    input_folder = Path(
        input_folder
    )

    output_folder = Path(
        output_folder
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for raster in input_folder.glob(
        "*.tif"
    ):

        output = (
            output_folder
            / f"{raster.stem}.asc"
        )

        if output.exists():
            continue

        raster_to_ascii(
            raster,
            output,
        )