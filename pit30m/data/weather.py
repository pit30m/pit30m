"""Functions for dealing with Pit30M's weather metadata."""

import glob
import json
import os
import zipfile
from multiprocessing import Pool
from typing import Any

import fsspec
import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

_MIN_JSON_CHARS = 5
DEFAULT_WEATHER_DATA_URI = "s3://pit30m/metadata/darksky_api_cache.zip"

def _process_weather_file(file_path: str) -> dict[str, Any]:
    """Reads a single weather JSON file and returns a record dict or None."""
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]

    # Parse lat, lon, and timestamp from filename
    lat_str, lon_str, timestamp_str = base_name.split("_")
    lat = float(lat_str)
    lon = float(lon_str)
    filename_timestamp = int(timestamp_str)

    try:
        with open(file_path, "r") as f:
            text = f.read()
            if len(text) < _MIN_JSON_CHARS:
                return None  # Skip invalid/empty-ish content

            data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None

    # Build record (time might be missing, so we fall back to filename_timestamp)
    record = {
        "lat": lat,
        "lon": lon,
        "time": data.get("time", filename_timestamp),
    }
    # Add other fields if present
    for key, val in data.items():
        if key != "time":
            record[key] = val

    return record

def aggregate_weather_data(input_dir="/mnt/weather", output_geojson="/tmp/weather_data.geojson", num_processes=8):
    """Aggregates weather data into a single GeoDataFrame using parallel processing.

    Please refer to 'aggregate_weather_data_s3' for detailed documentation.
    """
    file_paths = glob.glob(os.path.join(input_dir, "*.json"))

    print(f"Processing {len(file_paths)} weather files...")
    pool = Parallel(n_jobs=num_processes, batch_size=100)
    records = pool(delayed(_process_weather_file)(fp) for fp in file_paths)
    records = [r for r in records if r is not None]

    # Convert the raw records into the GeoDataFrame other parts of our tooling can use.
    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    print(f"Writing output to {output_geojson}")
    gdf.to_file(output_geojson, driver="GeoJSON")
    print("Done.")


def aggregate_weather_data_s3(output_geojson_fpath: str, num_processes: int, s3_fpath: str = DEFAULT_WEATHER_DATA_URI,
                            local_staging_dir: str = "/tmp/pit30m-weather-raw"):
    """Aggregates weather data from S3 into a single GeoDataFrame using parallel processing.

    The raw Pit30M data is a huge collection of JSON files, each containing weather data for a specific time/lat/lon.
    This format is very slow and annoying to deal with, so the devkit first downloads the data and aggregates it into
    a compact GeoJSON which we can then query at log reading time, visualize, analyze, etc.

    Args:
        output_geojson_fpath:   Output file path for the GeoJSON.
        num_processes:          Number of parallel processes to use for processing the raw data, once it is downloaded.
                                The sweet spot is 4--8.
        s3_fpath:               S3 URI for the weather data ZIP file. You should probably not change this, since the
                                path is part of the dataset's S3 bucket.
        local_staging_dir:      Local directory to stage the downloaded extracted files, before processing them into
                                a GeoJSON.
    """
    print("Downloading weather data...")
    with fsspec.open(s3_fpath, "rb") as f:
        with zipfile.ZipFile(f) as z:
            z.extractall(local_staging_dir)

    print("Aggregating weather data...")
    return aggregate_weather_data(local_staging_dir, output_geojson_fpath, num_processes)
