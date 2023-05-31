"""Utility functions for timing, data association, and indexing."""

import math
import multiprocessing as mp
import os
from urllib.parse import urlparse

import fsspec
import lz4
import numpy as np
from joblib import Memory, Parallel, delayed

from pit30m.fs_util import cached_glob_images, cached_glob_lidar_sweeps
from pit30m.time_utils import gps2utc
from pit30m.util import print_list_with_limit

# 1M images in a log would mean 100k seconds = A 27h nonstop log. We can't overflow this max length.
MAX_IMG_RELPATH_LEN = 22  # = len("0090/000000.night.webp")
MAX_LIDAR_RELPATH_LEN = 15  # = len("0000/007959.npz")
START_OF_2011_UNIX = 1293861600

memory = Memory(location=os.path.expanduser("~/.cache/pit30m"), verbose=0)

# Please refer to the NumPy documentation for exact details on type dimensions.
# https://numpy.org/doc/stable/reference/arrays.dtypes.html

# Deprecated, please use V0.
CAM_INDEX_V0_0_DTYPE = np.dtype(
    [
        ("rel_path", str, MAX_LIDAR_RELPATH_LEN),
        # NOTE(andrei): GPS timestamp in the pre-release index.
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
        ("utm_x", np.double),  # Easting
        ("utm_y", np.double),  # Northing
        ("utm_z", np.double),  # TODO(andrei): Make sure you populate UTM Z (altitude) eventually
    ]
)
CAM_INDEX_V1_0_DTYPE = np.dtype(
    [
        # TODO-LOW(andrei): Index a number and day/night to save space.
        ("rel_path", str, MAX_IMG_RELPATH_LEN),
        # Unix timestamp
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
        ("utm_x", np.double),  # Easting
        ("utm_y", np.double),  # Northing
        ("utm_z", np.double),  # Altitude
    ]
)


LIDAR_INDEX_V0_0_DTYPE = np.dtype(
    [
        ("rel_path", str, 14),
        ("lidar_time", np.double),
        ("lidar_min_time", np.double),
        ("lidar_max_time", np.double),
        ("lidar_mean_time", np.double),
        ("lidar_median_time", np.double),
        ("num_points", np.int64),
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
        ("utm_x", np.double),  # Easting
        ("utm_y", np.double),  # Northing
        ("utm_z", np.double),  # Altitude
    ]
)
LIDAR_INDEX_V1_0_DTYPE = np.dtype(
    [
        ("rel_path", str, MAX_LIDAR_RELPATH_LEN),
        ("lidar_time", np.double),
        ("lidar_min_time", np.double),
        ("lidar_max_time", np.double),
        ("lidar_mean_time", np.double),
        ("lidar_median_time", np.double),
        ("num_points", np.int64),
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
        ("utm_x", np.double),  # Easting
        ("utm_y", np.double),  # Northing
        ("utm_z", np.double),  # Altitude
    ]
)


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
        timestamp_gps_s = float(meta["capture_seconds"])

        # We basically assert the timestamp we read is actually GPS time. Uber ATG did not exist in 2010!
        assert timestamp_gps_s < START_OF_2011_UNIX
        # Not used
        # transmission_s = float(meta["transmission_seconds"])
        timestamp_unix_s = gps2utc(timestamp_gps_s).timestamp()

        entry = (timestamp_unix_s, float(meta["shutter_seconds"]), int(meta["sequence_counter"]), float(meta["gain_db"]))
        return img_uri, entry


@memory.cache(verbose=0)
def fetch_metadata_for_lidar(lidar_uri: str) -> tuple[str, tuple]:
    """Returns LiDAR timing metadata.

    All returned timestamps are UNIX timestamps.
    """
    with fsspec.open(lidar_uri) as compressed_f:
        with lz4.frame.open(compressed_f, "rb") as f:
            lidar_data = np.load(f)
            point_times_gps = lidar_data["seconds"]
            # See the camera metadata function for why we assert this.
            assert np.min(point_times_gps) < START_OF_2011_UNIX

            if lidar_data["points"].ndim != 2 or lidar_data["points"].shape[-1] != 3:
                return (
                    "Error",
                    lidar_uri,
                    "unexpected-points-shape",
                    "{}".format(lidar_data["points"].shape),
                )
            if lidar_data["points"].dtype != np.float32:
                return "Error", lidar_uri, "unexpected-points-dtype", str(lidar_data["points"].dtype)
            if lidar_data["points_H_sensor"].ndim != 2 or lidar_data["points_H_sensor"].shape[-1] != 3:
                return (
                    "Error",
                    lidar_uri,
                    "unexpected-points_H_sensor-shape",
                    str(lidar_data["points_H_sensor"].shape),
                )
            if lidar_data["points_H_sensor"].dtype != np.float32:
                return (
                    "Error",
                    lidar_uri,
                    "unexpected-points_H_sensor-dtype",
                    str(lidar_data["points_H_sensor"].dtype),
                )
            if len(lidar_data["intensity"]) != len(lidar_data["points"]):
                return (
                    "Error",
                    lidar_uri,
                    "unexpected-intensity-shape",
                    "{} vs. {} points".format(lidar_data["intensity"].shape, lidar_data["points"].shape),
                )
            if len(lidar_data["seconds"]) != len(lidar_data["points"]):
                return (
                    "Error",
                    lidar_uri,
                    "unexpected-point-time-shape",
                    "{} vs. {} points".format(lidar_data["seconds"].shape, lidar_data["points"].shape),
                )

            # Assumption - no leap second during the sweep.
            first_point_gps = point_times_gps[0]
            first_point_unix = gps2utc(first_point_gps).timestamp()
            assert first_point_unix > first_point_gps
            naive_delta = first_point_unix - point_times_gps[0]
            assert naive_delta > 0

            point_times_unix = point_times_gps + naive_delta

            return (
                "OK",
                lidar_uri,
                point_times_unix.min(),
                point_times_unix.max(),
                point_times_unix.mean(),
                np.median(point_times_unix),
                lidar_data["points"].shape,
            )


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


def build_camera_index(in_root, log_reader, cam_dir, logger, index_version):
    """Internal function to build an index for a sensor in a log.

    Grabs all available images and tries to associate them with poses by timestamp, creating an index in numpy binary
    format (easy to parse in C++ as well, for instance). The index has an entry for every image - if that image did not
    have a matching pose (especially common at the start and end of logs), then the appropriate fields `mrp_present`
    and `cp_present` will be set to False.

    Please refer to the LogReader documentation and the project README for details on how poses work and what MRP and CP
    means.
    """
    logger.info("Reading continuous pose data")
    cp_dense = log_reader.continuous_pose_dense
    cp_times = np.array(cp_dense[:, 0])
    in_scheme = urlparse(in_root).scheme
    in_fs = fsspec.filesystem(in_scheme)
    if index_version == 0:
        camera_index_dtype = CAM_INDEX_V0_0_DTYPE
    elif index_version == 1:
        camera_index_dtype = CAM_INDEX_V1_0_DTYPE
    else:
        raise ValueError(f"Unknown index version {index_version}")

    logger.info("Reading UTM and MRP")
    # Note that UTM is inferred from MRP via (very much imperfect) submap UTMs
    utm_poses = log_reader.utm_poses_dense
    mrp_poses = log_reader.map_relative_poses_dense
    assert len(mrp_poses) == len(utm_poses)
    mrp_times = np.array(mrp_poses["time"])

    logger.info("Listing all images from %s", cam_dir)
    image_uris = cached_glob_images(cam_dir, in_fs)
    logger.info("There are %d images in %s", len(image_uris), cam_dir)

    h = len(image_uris) / 10.0 / 3600.0
    print(f"{h:.2f} hours of driving")

    # NOTE(andrei): Seems to be worth over-subscribing!
    #
    # Coarse numbers from my personal machine, 16 core 1Gbps connection to S3.
    # 16, 1min = 8k samples
    # 32, 1min = 16k samples
    # 64, 1min = 32.3k samples
    # 128, 1 min = ~54k samples (finished my input)
    # 128, 30s = 29k
    # 256, 30s = 54k (though I think my upload speed is unnaturally fast... (for the requests))
    # 256, 15s (bs = 4) = 23.5k, 28k, ...
    # 256, 15s, bs = 8  = 28.8k
    # 512, 15s = 14.9k
    #
    # This subsequently got a fair bit slower once I actually parsed the npy, oh well
    pool = Parallel(n_jobs=mp.cpu_count() * 8, verbose=1, batch_size=8)
    logger.info("Fetching metadata...")
    res = pool(delayed(fetch_metadata_for_image)(x) for x in image_uris)
    logger.info("Fetched %d results", len(res))

    image_times = np.array([float(entry[1][0]) for entry in res])
    logger.info("Associating...")
    logger.info(
        "%s %s %s %s",
        image_times.dtype,
        str(image_times.shape),
        str(image_times[0]) if len(image_times) else "n/A",
        str(type(image_times[0])) if len(image_times) else "n/A",
    )
    logger.info("%s %s %s %s", mrp_times.dtype, str(mrp_times.shape), str(mrp_times[0]), str(type(mrp_times[0])))
    assert np.all(mrp_times[1:] > mrp_times[:-1])

    utm_and_mrp_index = associate(image_times, mrp_times, max_delta_s=-1.0)
    logger.info("Associating complete.")
    matched_timestamps_mrp = mrp_times[utm_and_mrp_index]
    deltas_mrp = np.abs(matched_timestamps_mrp - image_times)

    logger.info("Associating CP...")
    cp_index = associate(image_times, cp_times, max_delta_s=-1.0)
    logger.info("Associating complete.")
    matched_timestamps_cp = cp_times[cp_index]
    deltas_cp = np.abs(matched_timestamps_cp - image_times)

    unindexed_frames = []
    status = []
    raw_index = []
    for row_idx, ((img_fpath, img_data), pose_idx, cp_idx, delta_mrp_s, delta_cp_s) in enumerate(
        zip(res, utm_and_mrp_index, cp_index, deltas_mrp, deltas_cp)
    ):
        img_time, shutter_s, seq_counter, gain_db = img_data
        if delta_mrp_s > 0.10:
            mrp = None
            mrp_present = False
            utm_present = False
            mrp_valid = False
            mrp_time_s, mrp_x, mrp_y, mrp_z, mrp_roll, mrp_pitch, mrp_yaw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            mrp_submap_id = b"\x00" * 16
            utm = None
            utm_x, utm_y, utm_z = 0.0, 0.0, 0.0
        else:
            mrp = mrp_poses[pose_idx, ...].tolist()
            utm = utm_poses[pose_idx, ...].tolist()
            mrp_present = True
            utm_present = True
            mrp_time_s, mrp_valid, mrp_submap_id, mrp_x, mrp_y, mrp_z, mrp_roll, mrp_pitch, mrp_yaw = mrp
            utm_x, utm_y, utm_z = utm
            if math.isnan(utm_x):
                utm_x = 0.0
                utm_y = 0.0
                utm_z = 0.0
                utm_present = False

            # Handle submap IDs which were truncated upon encoded due to ending with a zero.
            mrp_submap_id = mrp_submap_id.ljust(16, b"\x00")
            assert type(mrp_submap_id) == bytes
            assert 16 == len(mrp_submap_id)

        if delta_cp_s > 0.10:
            cp_present = False
            cp_valid = False
            cp_time_s, cp_x, cp_y, cp_z, cp_roll, cp_pitch, cp_yaw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            cp = cp_dense[cp_idx, ...]
            cp_present = True
            (cp_time_s, cp_valid, cp_x, cp_y, cp_z, cp_roll, cp_pitch, cp_yaw) = cp
            cp_valid = bool(cp_valid)

        img_rel_fpath = "/".join(img_fpath.split("/")[-2:])
        assert len(img_rel_fpath) <= MAX_IMG_RELPATH_LEN
        img_rel_fpath = img_rel_fpath.ljust(MAX_IMG_RELPATH_LEN, " ")
        raw_index.append(
            (
                img_rel_fpath,
                img_time,
                shutter_s,
                seq_counter,
                gain_db,
                cp_present,
                cp_valid,
                cp_time_s,
                cp_x,
                cp_y,
                cp_z,
                cp_roll,
                cp_pitch,
                cp_yaw,
                mrp_present,
                mrp_valid,
                mrp_time_s,
                mrp_submap_id,
                mrp_x,
                mrp_y,
                mrp_z,
                mrp_roll,
                mrp_pitch,
                mrp_yaw,
                utm_present,
                utm_present,
                utm_x,
                utm_y,
                utm_z,
            )
        )

    index = np.array(raw_index, dtype=camera_index_dtype)
    index = np.sort(index, order="img_time")

    cp_p = index["cp_present"]
    print("CP total:", len(cp_p))
    print("CP valid:", sum(cp_p))

    mrp_p = index["mrp_present"]
    print("MRP total:", len(mrp_p))
    print("MRP valid:", sum(mrp_p))
    return index


def build_lidar_index(in_root, log_reader, lidar_dir, _logger, index_version: int):
    """Internal function to build an index for a LiDAR in a log."""
    _logger.info("Reading continuous pose data")
    cp_dense = log_reader.continuous_pose_dense
    cp_times = np.array(cp_dense[:, 0])
    in_scheme = urlparse(in_root).scheme
    in_fs = fsspec.filesystem(in_scheme)

    if index_version == 0:
        lidar_dtype = LIDAR_INDEX_V0_0_DTYPE
    elif index_version == 1:
        lidar_dtype = LIDAR_INDEX_V1_0_DTYPE
    else:
        raise ValueError(f"Unknown index version {index_version}")

    _logger.info("Reading UTM and MRP")
    # Note that UTM is inferred from MRP via (very much imperfect) submap UTMs
    utm_poses = log_reader.utm_poses_dense
    mrp_poses = log_reader.map_relative_poses_dense
    assert len(mrp_poses) == len(utm_poses)
    mrp_times = np.array(mrp_poses["time"])

    _logger.info("Listing all LiDAR sweeps from %s", lidar_dir)
    lidar_sweep_uris = cached_glob_lidar_sweeps(lidar_dir, in_fs)
    _logger.info("There are %d LiDARs in %s", len(lidar_sweep_uris), lidar_dir)

    h = len(lidar_sweep_uris) / 10.0 / 3600.0
    print(f"{h:.2f} hours of driving")

    pool = Parallel(n_jobs=mp.cpu_count() * 4, verbose=1, batch_size=8)
    _logger.info("Fetching metadata...")
    all_lidar_meta_info = pool(delayed(fetch_metadata_for_lidar)(x) for x in lidar_sweep_uris)
    _logger.info("Fetched %d results", len(all_lidar_meta_info))

    errors = [x for x in all_lidar_meta_info if x[0] != "OK"]

    ts_path = os.path.join(lidar_dir, "timestamps.npz.lz4")
    with in_fs.open(ts_path, "rb") as f:
        with lz4.frame.open(f, mode="rb") as ff:
            # These are the timestamps around which I dumped ~0.1s of LiDAR data.
            #
            # Motion compensation details are hazy; I think the LiDAR should be compensated to this frame, though it may
            # also be to some other predefined time which was hardcoded by the log reader. Either way, the information
            # is *there* (we have per-point times), it's just a matter of playing with projecting the LiDAR to the
            # cameras in order to identify the exact time whose pose the LiDAR data is wrt to.
            #
            # This estimate will however be enough for simple visual localization tasks for now. I have a more
            # overarching goal of sorting out LiDAR as the continuous-time sensor that it is once we get the first
            # version of the devkit out.
            dumped_timestamps = np.load(ff)["data"]["timestamp"].ravel()

    if len(errors) > 0:
        _logger.error("Found %d errors reading LiDAR for indexing purposes!!!", len(errors))
        print_list_with_limit(errors, 10, logger=_logger)
        log_id = log_reader.log_id
        raise RuntimeError(f"Could not index LIDAR for log ID {log_id} due to errors (see log)")

    _logger.info("No errors found reading %d LiDAR sweep files for indexing.", len(all_lidar_meta_info))

    # dmins = []
    # dmaxs = []
    lidar_info = []
    # Please read the above detailed comment for what we mean by "LiDAR time".
    lidar_times = []
    for idx, (andrei_timestamp, (_, lidar_uri, min_time, max_time, mean_time, p50_time, shape)) in enumerate(
        zip(dumped_timestamps, all_lidar_meta_info)
    ):
        # NOTE(andrei): LIDAR is rolling shutter...
        # if idx % 200 == 0:
        #     print(f"{idx} / {len(lidar_sweep_uris)}")
        #     print(f"min_time: {min_time}")
        #     print(f"max_time: {max_time}")
        #     # print(f"mean_time: {mean_time}")
        #     print(f"{float(andrei_timestamp) = }")
        #     # print(f"dmin/dmax: {dmin}/{dmax}")
        num_points, pcd_dim = shape
        assert 3 == pcd_dim

        lidar_info.append((andrei_timestamp, lidar_uri, min_time, max_time, mean_time, p50_time, num_points))
        lidar_times.append(andrei_timestamp)

    lidar_times = np.array(lidar_times, dtype=np.float64)

    #     dmin = andrei_timestamp - min_time
    #     dmax = max_time - andrei_timestamp
    #     dmins.append(dmin)
    #     dmaxs.append(dmax)

    # print(np.mean(dmins))
    # print(np.mean(dmaxs))

    _logger.info("Associating...")
    _logger.info(
        "%s %s %s %s",
        type(lidar_times),
        str(len(lidar_times)),
        str(lidar_times[0]) if len(lidar_times) else "n/A",
        str(type(lidar_times[0])) if len(lidar_times) else "n/A",
    )
    _logger.info("%s %s %s %s", mrp_times.dtype, str(mrp_times.shape), str(mrp_times[0]), str(type(mrp_times[0])))
    assert np.all(mrp_times[1:] > mrp_times[:-1])

    utm_and_mrp_index = associate(lidar_times, mrp_times, max_delta_s=-1.0)
    _logger.info("Associating complete.")

    matched_timestamps_mrp = mrp_times[utm_and_mrp_index]
    deltas_mrp = np.abs(matched_timestamps_mrp - lidar_times)

    _logger.info("Associating CP...")
    cp_index = associate(lidar_times, cp_times, max_delta_s=-1.0)
    _logger.info("Associating complete.")
    matched_timestamps_cp = cp_times[cp_index]
    deltas_cp = np.abs(matched_timestamps_cp - lidar_times)

    # TODO(andrei): Code duplication between this and the camera index builder.
    raw_index = []
    for (
        (lidar_time, lidar_fpath, min_time, max_time, mean_time, p50_time, npts),
        pose_idx,
        cp_idx,
        delta_mrp_s,
        delta_cp_s,
    ) in zip(lidar_info, utm_and_mrp_index, cp_index, deltas_mrp, deltas_cp):
        if delta_mrp_s > 0.10:
            mrp = None
            mrp_present = False
            mrp_valid = False
            utm_present = False
            mrp_time_s, mrp_x, mrp_y, mrp_z, mrp_roll, mrp_pitch, mrp_yaw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            mrp_submap_id = b"\x00" * 16
            utm = None
            utm_x, utm_y, utm_z = 0.0, 0.0, 0.0
        else:
            mrp = mrp_poses[pose_idx, ...].tolist()
            utm = utm_poses[pose_idx, ...].tolist()
            mrp_present = True
            utm_present = True
            mrp_time_s, mrp_valid, mrp_submap_id, mrp_x, mrp_y, mrp_z, mrp_roll, mrp_pitch, mrp_yaw = mrp
            utm_x, utm_y, utm_z = utm
            if math.isnan(utm_x):
                utm_x = 0.0
                utm_y = 0.0
                utm_z = 0.0
                utm_present = False

            # Handle submap IDs which were truncated upon encoded due to ending with a zero.
            mrp_submap_id = mrp_submap_id.ljust(16, b"\x00")
            assert type(mrp_submap_id) == bytes
            assert 16 == len(mrp_submap_id)

        if delta_cp_s > 0.10:
            cp_present = False
            cp_valid = False
            cp_time_s, cp_x, cp_y, cp_z, cp_roll, cp_pitch, cp_yaw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            cp = cp_dense[cp_idx, ...]
            cp_present = True
            (cp_time_s, cp_valid, cp_x, cp_y, cp_z, cp_roll, cp_pitch, cp_yaw) = cp
            cp_valid = bool(cp_valid)

        lidar_rel_fpath = "/".join(lidar_fpath.split("/")[-2:])
        assert len(lidar_rel_fpath) <= MAX_IMG_RELPATH_LEN
        lidar_rel_fpath = lidar_rel_fpath.ljust(MAX_IMG_RELPATH_LEN, " ")
        raw_index.append(
            (
                lidar_rel_fpath,
                lidar_time,
                min_time,
                max_time,
                mean_time,
                p50_time,
                npts,
                cp_present,
                cp_valid,
                cp_time_s,
                cp_x,
                cp_y,
                cp_z,
                cp_roll,
                cp_pitch,
                cp_yaw,
                mrp_present,
                mrp_valid,
                mrp_time_s,
                mrp_submap_id,
                mrp_x,
                mrp_y,
                mrp_z,
                mrp_roll,
                mrp_pitch,
                mrp_yaw,
                # No MRP => No UTM
                utm_present,
                utm_present,
                utm_x,
                utm_y,
                utm_z,
            )
        )

    index = np.array(raw_index, dtype=lidar_dtype)
    index = np.sort(index, order="lidar_time")

    cp_p = index["cp_present"]
    print("CP total:", len(cp_p))
    print("CP valid:", sum(cp_p))

    mrp_p = index["mrp_present"]
    print("MRP total:", len(mrp_p))
    print("MRP valid:", sum(mrp_p))

    return index
