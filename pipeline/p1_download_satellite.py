import json
import os
from pathlib import Path

import geopandas as gpd
import planetary_computer
import requests
from pystac_client import Client

from config import (
    COLLECTIONS,
    LANDSAT_BAND_MAP,
    SENTINEL_BANDS,
    STAC_URL,
    DEFAULT_START_MONTH,
    DEFAULT_START_DAY,
    DEFAULT_END_MONTH,
    DEFAULT_END_DAY,
)
from common_preprocessing import load_aoi_geometry

# Defining function to access the assets for a given satellite scene item, based on its collection and platform, and returning the corresponding band mapping
def get_assets_for_item(item):

    collection = item.collection_id

    if collection == "landsat-c2-l2":

        platform = item.properties.get("platform")

        if platform not in LANDSAT_BAND_MAP:
            raise ValueError(
                f"Unsupported Landsat platform: {platform}"
            )

        return (
            platform,
            LANDSAT_BAND_MAP[platform]["bands"],
        )

    if collection == "sentinel-2-l2a":
        return (
            "sentinel-2",
            SENTINEL_BANDS,
        )

    raise ValueError(
        f"Unsupported collection: {collection}"
    )

# Defining function to assign a downloaded file a specific output location
def download_file(url, output_path):

    output_path = Path(output_path)

    if output_path.exists() and output_path.stat().st_size > 10_000:
        return

    temporary = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    try:

        with requests.get(
            url,
            stream=True,
            timeout=120,
        ) as response:

            response.raise_for_status()

            with open(temporary, "wb") as dst:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        dst.write(chunk)

        if temporary.stat().st_size < 10_000:
            raise IOError(
                f"Downloaded file appears invalid: {temporary}"
            )

        temporary.replace(output_path)

    finally:

        if temporary.exists():
            temporary.unlink(missing_ok=True)

# Defining function to download satellite data for a given Area of Interest (AOI) and time range
def download_satellite(
    aoi_path,
    output_folder,
    start_year,
    end_year,
    start_month=DEFAULT_START_MONTH,
    start_day=DEFAULT_START_DAY,
    end_month=DEFAULT_END_MONTH,
    end_day=DEFAULT_END_DAY,
):


    # Raw satellite scenes can be shared across AOIs. Scene IDs are used as unique scene folders, allowing different AOIs to reuse the same downloaded satellite scenes without duplication

    output_folder = Path(output_folder)
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    aoi_geometry = load_aoi_geometry(aoi_path)

    minx, miny, maxx, maxy = aoi_geometry.bounds

    # STAC expects WGS84 coordinates.
    bbox = gpd.GeoSeries(
        [aoi_geometry],
        crs="EPSG:27700",
    ).to_crs("EPSG:4326").total_bounds

    catalog = Client.open(STAC_URL)

    items_by_id = {}

    for year in range(start_year, end_year + 1):

        start = (
            f"{year}-{start_month:02d}-{start_day:02d}"
        )

        end = (
            f"{year}-{end_month:02d}-{end_day:02d}"
        )

        print(
            f"Searching satellite scenes: {start} → {end}"
        )

        search = catalog.search(
            collections=COLLECTIONS,
            bbox=tuple(bbox),
            datetime=f"{start}/{end}",
        )

        for item in search.items():
            items_by_id[item.id] = item

    print(
        f"Unique satellite scenes found: "
        f"{len(items_by_id)}"
    )

    for item_id, item in sorted(
        items_by_id.items()
    ):

        scene_dir = output_folder / item_id
        scene_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            platform, band_map = get_assets_for_item(item)

            signed_item = planetary_computer.sign(item)

            metadata = {
                "id": item.id,
                "collection": item.collection_id,
                "platform": platform,
                "datetime": (
                    item.datetime.isoformat()
                    if item.datetime
                    else None
                ),
                "bbox": list(item.bbox)
                if item.bbox
                else None,
                "properties": dict(item.properties),
            }

            with open(
                scene_dir / "scene_metadata.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    metadata,
                    f,
                    indent=2,
                    default=str,
                )

            for common_name, asset_name in band_map.items():

                if asset_name not in signed_item.assets:

                    print(
                        f"{item.id}: missing "
                        f"{asset_name}"
                    )
                    continue

                asset = signed_item.assets[
                    asset_name
                ]

                output_path = (
                    scene_dir
                    / f"{item.id}_{asset_name}.tif"
                )

                print(
                    f"Downloading "
                    f"{item.id}: {asset_name}"
                )

                download_file(
                    asset.href,
                    output_path,
                )

        except Exception as exc:

            print(
                f"Failed scene {item.id}: {exc}"
            )

    print("Satellite download complete.")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--aoi", required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)

    args = parser.parse_args()

    from config import RAW_DIR

    download_satellite(
        args.aoi,
        RAW_DIR,
        args.start_year,
        args.end_year,
    )