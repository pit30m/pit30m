"""Utility functions for timing, data association, and indexing."""

import fsspec
import numpy as np

MAX_IMG_RELPATH_LEN = 22 # = len("0090/000000.night.webp")
# 1M images in a log would mean 100k seconds = A 27h nonstop log.

# Please refer to the NumPy documentation for exact details on type dimensions.
# https://numpy.org/doc/stable/reference/arrays.dtypes.html
INDEX_V0_0_DTYPE = np.dtype([
    ("rel_path", str, MAX_IMG_RELPATH_LEN),
    ("img_time", np.double),
    ("shutter_s", np.double),
    ("seq_counter", np.int64),
    ("gain_db", np.double),
    ("cp_present", bool),
    ("cp_valid", bool),
    ("cp_time_s", np.double),
    ("cp_x", np.double),
    ("cp_y", np.double),
    ("cp_z", np.double),
    ("cp_roll", np.double),
    ("cp_pitch", np.double),
    ("cp_yaw", np.double),
    ("mrp_present", bool),
    ("mrp_valid", bool),
    ("mrp_time", np.double),
    ("mrp_submap_id", bytes, 16),
    ("mrp_x", np.double),
    ("mrp_y", np.double),
    ("mrp_z", np.double),
    ("mrp_roll", np.double),
    ("mrp_pitch", np.double),
    ("mrp_yaw", np.double),
    # UTM coordinates are provided for CONVENIENCE, computed from MRP based on the v0 (unrefined)
    # submap UTM coordinates. Ideally you'll eventually want to use the 'Map' class and compute your
    # own UTMs once Andrei B. manages to release refined submap UTMs.
    ("utm_present", bool),
    ("utm_valid", bool),
    ("utm_x", np.double), # Easting
    ("utm_y", np.double), # Northing
    ("utm_z", np.double), # TODO(andrei): Make sure you populate UTM Z (altitude) eventually
])


def fetch_metadata_for_image(img_uri: str) -> tuple[str, tuple]:
    """Returns the original URI and the image metadata.

    Used in indexing operations. Average users should not be reading metadata files directly
    as there are many of them and they are tiny, so it is very inefficient. Users should be
    using the camera/LiDAR indexes, which are loaded once then can reside in memory without the
    need to hit S3 for every tiny metadata query (e.g., what's this image's timestamp?)

    The many tiny metadata files were just dumped originally for simplicity and maximum
    robustness.
    """
    meta_uri = img_uri.replace(".day", ".night").replace(".night.webp", ".meta.npy").replace(".webp", ".meta.npy")
    with fsspec.open(meta_uri) as meta_f:
        meta = np.load(meta_f, allow_pickle=True).tolist()
        timestamp_s = float(meta["capture_seconds"])
        # Not used
        # transmission_s = float(meta["transmission_seconds"])

        entry = (
            timestamp_s, float(meta["shutter_seconds"]), int(meta["sequence_counter"]), float(meta["gain_db"])
        )
        return img_uri, entry


def associate(query_timestamps: np.ndarray, target_timestamps: np.ndarray, max_delta_s: float = 0.5) -> np.ndarray:
    """Associates timestamps from two arrays, returning the indices of the closest matches.

    The result will contain an index into 'db_ts' for each entry in 'q_ts' (query timestamps).
    All timestamps are expected to be float64, seconds.

    Args:
        query_timestamps:   The timestamps to find matches for.
        target_timestamps:  The timestamps to match against. Must be sorted!!
        max_delta_s:        Warn if the gap between the query and the result is bigger than this. Note that we will
                            still return this index, so the end user is responsible for filtering out the results.
    """
    # TODO(andrei): Within a few ms at worst, poses are evenly spaced. We can use
    # arithmetic to dramatically constrain our search range to turn log n binary search
    # into a constant time look-up.
    #
    # TODO(andrei): Interpolate poses linearly. Poses are 100Hz so nearest-neighbor lookups can
    # be at most 10-ish ms off which can cause 0.33m of error for 33 mps driving (120kph) in
    # a worst-case scenario, so not trivial.
    #
    # Fortunately, for such small intervals, linear interpolation should be OK.
    result = np.zeros(query_timestamps.shape, dtype=np.int64)

    assert target_timestamps.shape[0] > 0
    assert target_timestamps.dtype == np.float64 or target_timestamps.dtype == np.float32
    assert query_timestamps.dtype == np.float64 or query_timestamps.dtype == np.float32

    # TODO(andrei): speed up further with np.interpolate
    for q_idx, q_ts in enumerate(query_timestamps):
        target_idx = np.searchsorted(target_timestamps, q_ts, side="left")
        prev = target_idx - 1
        next = target_idx

        if next < len(target_timestamps):
            delta_prev = abs(q_ts - target_timestamps[prev])
            delta_next = abs(q_ts - target_timestamps[next])
            if delta_prev < delta_next:
                target_idx = prev
            else:
                target_idx = next
        else:
            target_idx = target_idx - 1

        delta_s = abs(q_ts - target_timestamps[target_idx])
        if max_delta_s > 0 and delta_s > max_delta_s:
            print(f"WARNING: Timestamp association gap is {delta_s:.3f}s for query {q_idx}.")
        result[q_idx] = target_idx

    return result


