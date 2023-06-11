import io
import os
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Iterable, Iterator, Optional, Tuple, Type
from urllib.parse import urlparse
from uuid import UUID

import fsspec
import lz4
import numpy as np
import numpy.typing as npt
import utm
from joblib import Memory
from numpy.lib import recfunctions as rfn
from PIL import Image

from pit30m.camera import CamName
from pit30m.data.partitions import (
    GeoPartition,
    Partition,
    PreProcessPartition,
    QueryBasePartition,
    SizePartition,
)
from pit30m.data.submap import Map, SubmapPoseNotFoundException
from pit30m.indexing import associate, associate_np
from pit30m.time_utils import gps_seconds_to_utc

memory = Memory(location=os.path.expanduser("~/.cache/pit30m"), verbose=0)


@dataclass
class CameraImage:
    # TODO consider replacing the cam with a Camera object which can be used to get the intrinsics and stuff
    image: np.ndarray
    cam_name: str
    capture_timestamp: float
    shutter_time_s: float
    gain_db: float


@dataclass
class LiDARFrame:
    xyz_continuous: np.ndarray
    xyz_sensor: np.ndarray
    intensity: np.ndarray
    point_times: np.ndarray
    laser_theta: np.ndarray = None
    raw_power: np.ndarray = None
    laser_id: np.ndarray = None
    # ['laser_theta', 'seconds', 'raw_power', 'intensity', 'points', 'points_H_sensor', 'laser_id']


VELODYNE_NAME = "hdl64e_12_middle_front_roof"

# NOTE(julieta) 17T or 17N are probably ok for PIT, commit to "N"
UTM_ZONE_NUMBER = 17
UTM_ZONE_LETTER = "N"

PARTITIONS_BASEPATH = "s3://pit30m/partitions/"
LATEST_INDEX_VERSION = 1

# Sensor must be at most this fart from the pose to be considered a match
SENSOR_TO_POSE_MATCH_EPSILON_S = 0.5

# Original dtype for unified raw pose arrays. Regular users should be using specialized getters, such as those for
# continuous or map-relative poses, not the raw poses. '?' represents bool.
RAW_POSE_DTYPE = np.dtype(
    [
        ("transmission_time", "<f8"),
        ("transmission_sequence_counter", "<i8"),
        ("capture_time", "<f8"),
        ("poses_and_differentials_valid", "?"),
        (
            "map_relative",
            [
                ("submap", "S16"),
                ("x", "<f8"),
                ("y", "<f8"),
                ("z", "<f8"),
                ("yaw", "<f8"),
                ("pitch", "<f8"),
                ("roll", "<f8"),
                ("pose_covariance", "<f8", (6, 6)),
                ("valid", "?"),
            ],
        ),
        (
            "continuous",
            [
                ("x", "<f8"),
                ("y", "<f8"),
                ("z", "<f8"),
                ("yaw", "<f8"),
                ("pitch", "<f8"),
                ("roll", "<f8"),
                ("pose_covariance", "<f8", (6, 6)),
                ("vx", "<f8"),
                ("vy", "<f8"),
                ("vz", "<f8"),
                ("roll_rate", "<f8"),
                ("pitch_rate", "<f8"),
                ("yaw_rate", "<f8"),
                ("velocity_covariance", "<f8", (6, 6)),
                ("acceleration", "<f8", (3, 1)),
                ("valid", "?"),
            ],
        ),
    ]
)


def gps_to_unix_timestamp(gps_seconds: float) -> float:
    return gps_seconds_to_utc(gps_seconds).timestamp()


class LogReader:
    def __init__(
        self,
        log_root_uri: str,
        pose_fname: str = "all_poses.npz",
        wgs84_pose_fname: str = "wgs84.npz",
        map: Map = None,
        index_version: int = LATEST_INDEX_VERSION,
        partitions: Optional[Iterable[Partition]] = None,
    ):
        """Lightweight, low-level S3-aware utility for interacting with a specific log.

        Devkit users should consider using existing higher-level APIs, such as `pit30m.torch.dataset.Pit30MDataset`
        unless they have a specific use case that requires low-level access to the data.

        Specifically, using abstractions like pytorch DataLoaders can dramatically improve throughput, e.g., going from
        20-30 images per second to 100+.

        Args:
            log_root_uri: URI to the root of the log. This should be a directory containing a "cameras", "lidars", and other dirs
            pose_fname: Name of the pose file. This is usually "all_poses.npz"
            wgs84_pose_fname: Name of the WGS84 pose file (ie, global coords). This is usually "wgs84.npz"
            map: Map object. If not provided, will try to load it from the log root
            index_version: Version of the index to use. Currently only 0 is supported.
            partitions: Set of partitions to load. These are used to load subset of the date (e.g., training queries).
                defaults to None, which means that no sensor measurements are filtered.
        """
        self._log_root_uri = log_root_uri.rstrip("/")
        self._pose_fname = pose_fname
        self._wgs84_pose_fname = wgs84_pose_fname
        self._map = Map() if map is None else map
        # TODO(julieta) Semantic version this
        self._index_version = index_version
        self.partitions = set() if partitions is None else partitions

    def __repr__(self) -> str:
        return f"Pit30M Log Reader: {self._log_root_uri}"

    @cached_property
    def _partitions_mask(self) -> np.ndarray:
        """
        Returns a boolean np array that accounts for the requested partitions, by setting sensor readings that should
        be skipped to False. Computed for the raw pose data @ 100 Hz.

        Returns:
            A structured array with two fields:
            1. "capture_time": float 64 series of pose GPS timestamps
            2. "mask": boolean indicating whether each pose is included in the requested partitions
        """
        # breakpoint()
        n_sensor_measurements = len(self.raw_pose_data)
        partition_indices = self._partition_assigments
        partition_indices += (np.full(n_sensor_measurements, True),)

        combined_indices_mask = np.logical_and.reduce(partition_indices)
        timestamps = self.raw_pose_data["capture_time"]

        # HACK: Convert the raw poses' timestamps to UNIX to make them comparable to those... everywhere else
        # for i, timestamp in enumerate(timestamps):
        #     if not np.isnan(timestamp):
        #         timestamps[i] = gps_to_unix_timestamp(timestamp)
        timestamps = rfn.unstructured_to_structured(timestamps[:, np.newaxis], dtype=[("timestamp", "<f8")])

        combined_indices_mask = rfn.unstructured_to_structured(
            combined_indices_mask[:, np.newaxis], dtype=[("mask", "?")]
        )

        mask_with_timestamps = rfn.merge_arrays((timestamps, combined_indices_mask))
        return mask_with_timestamps

    @cached_property
    def partition_fs(self):
        """Filesystem object used to read log data."""
        return fsspec.filesystem(urlparse(PARTITIONS_BASEPATH).scheme, anon=True)

    def _load_partition(self, Partition: Type[Partition]) -> np.ndarray:
        partition_fpath = os.path.join(PARTITIONS_BASEPATH, Partition.path_name(), f"{self.log_id}.npz")
        if not self.partition_fs.exists(partition_fpath):
            raise FileNotFoundError(f"Partition file not found: {partition_fpath}")
        with self.partition_fs.open(partition_fpath, "rb") as f:
            return np.load(f)["partition"]

    @cached_property
    def preprocess_partition(self) -> np.ndarray:
        """Fetches this log's preprocess partition from s3"""
        return self.load_partition(PreProcessPartition)

    @cached_property
    def geo_partition(self) -> np.ndarray:
        """Fetches this log's geo partition from s3"""
        return self.load_partition(GeoPartition)

    @cached_property
    def query_base_partition(self) -> np.ndarray:
        """Fetches this log's query base partition from s3"""
        return self.load_partition(QueryBasePartition)

    # @property
    # @memory.cache(verbose=0)
    @cached_property
    def _partition_assigments(self) -> Tuple[np.ndarray]:
        """Fetches partition indices from S3 and converts them to boolean arrays according to the reader's partition
        values. Note that these partitions are done wrt the raw pose data @ 100Hz, not the sensor data; therefore,
        we must index sensor data into these partitions to find out which partitions they belong to.

        Returns:
            A tuple of boolean arrays. Each array has n_dense_poses entries, and each entry represents membership of the
            pose to a requested partition.
        """

        # breakpoint()
        fs = fsspec.filesystem(urlparse(PARTITIONS_BASEPATH).scheme, anon=True)

        partition_indices = tuple()
        for partition in self.partitions:
            partition_fpath = os.path.join(PARTITIONS_BASEPATH, partition.path_name(), f"{self.log_id}.npz")
            if not fs.exists(partition_fpath):
                raise FileNotFoundError(f"Partition file not found: {partition_fpath}")

            with fs.open(partition_fpath, "rb") as f:
                idx = np.load(f)["partition"]
                # Convert the fetched partition index into boolean array that matches the requested value
                partition_indices += (partition.value_to_index(partition, idx),)

        return partition_indices

    @property
    def log_id(self) -> UUID:
        # TODO(julieta) make sure that this is a valid UUID
        return UUID(os.path.basename(self._log_root_uri))

    @property
    def cam_root(self) -> str:
        """Root for all the camera data (each camera is a subdirectory)."""
        return os.path.join(self._log_root_uri, "cameras")

    @property
    def lidar_root(self) -> str:
        # Simpler than cameras since there's always a single lidar.
        return os.path.join(self._log_root_uri, "lidars", VELODYNE_NAME)

    def get_cam_root(self, cam_name: CamName) -> str:
        assert isinstance(cam_name, CamName)
        return os.path.join(self._log_root_uri, "cameras", cam_name.value)

    @cached_property
    def fs(self):
        """Filesystem object used to read log data."""
        return fsspec.filesystem(urlparse(self._log_root_uri).scheme, anon=True)

    @lru_cache(maxsize=16)
    def get_lidar_geo_index(self, sort_by: str = "lidar_time") -> np.ndarray:
        """Returns a lidar index of dtype LIDAR_INDEX_V0_0_DTYPE.

        WARNING: 'rel_path' entries in indexes may be padded with spaces on the right since they are fixed-width
        strings. If you need to use them directly, make sure you use `.strip()` to remove the spaces.

        Args:
            sort_by: name of the field that we want to sort by. Defaults to `lidar_time`
        Returns:
            A structured numpy array with the lidar observations and their metadata.
        """
        index_fpath = os.path.join(self.lidar_root, "index", f"index_v{self._index_version}.npz")
        if not self.fs.exists(index_fpath):
            raise FileNotFoundError(f"Index file not found: {index_fpath}!")

        with self.fs.open(index_fpath, "rb") as f:
            # TODO(andrei): Pre-sort the indexes at gen time.
            index = np.load(f)["index"]

        index = index[np.argsort(index[sort_by])]
        return index

    def filter_partitions(self, query_timestamps: np.ndarray, max_delta_s: float = 0.2) -> np.ndarray:
        """Returns a boolean mask that indicates which sensor readings belong to the requested partitions."""

        # Now, leave only the subset that matches the requested partition
        partitions_mask = self._partitions_mask

        # Associate with partititions mask
        matched_indices, over = associate_np(query_timestamps, partitions_mask["timestamp"], max_delta_s=max_delta_s)
        matched_bool = partitions_mask["mask"][matched_indices]
        matched_bool[over] = False

        # Filter with the obtained mask
        return matched_bool

    @lru_cache(maxsize=16)
    def get_cam_geo_index(
        self,
        cam_name: CamName = CamName.MIDDLE_FRONT_WIDE,
        sort_by: str = "img_time",
        max_delta_s=0.1,
    ) -> np.ndarray:
        """Returns a camera index of dtype CAM_INDEX_V0_0_DTYPE.
        Args:
            cam_name: name of the camera index to load
            sort_by: name of the field that we want to sort by. Defaults to `img_time`
        Returns:
            A structured numpy array with the camera observations and their metadata.
        """

        # Get the pre-computed index
        # TODO(julieta) v0 and v1 have different name formatting on s3
        index_fpath = os.path.join(self.get_cam_root(cam_name), "index", f"index_v{self._index_version:02d}.npz")
        if not self.fs.exists(index_fpath):
            raise FileNotFoundError(f"Index file not found: {index_fpath}!")

        with self.fs.open(index_fpath, "rb") as f:
            index = np.load(f)["index"]

        index = index[np.argsort(index[sort_by])]

        # Filter by partitions
        partitions_mask = self.filter_partitions(index["img_time"], max_delta_s)
        index = index[partitions_mask]

        return index

    def calib(self):
        calib_fpath = os.path.join(self._log_root_uri, "mono_camera_calibration.npy")
        with self.fs.open(calib_fpath, "rb") as f:
            # UnicodeError if encoding not specified because Andrei was lazy when originally dumping the dataset
            # and didn't export calibration in a human-friendly format.
            data = np.load(f, allow_pickle=True, encoding="latin1")
            # Ugly code since I numpy-saved a dict...
            data = data.tolist()["data"]
            # They all contain the values of CamNames as values
            #
            # I need to play with these different values and double check what the raw images are present as, but in
            # principle ...Z = calibration so that image Y is Z world, etc.
            #
            # Main question: Are the raw images dumped already rectified one way or another?
            data["ALIGNED_WITH_VEHICLE_Z"]
            data["MONOCULAR_RECTIFIED"]
            data["MONOCULAR_UNRECTIFIED"]
            #
            return data

    def stereo_calib(self):
        calib_fpath = os.path.join(self._log_root_uri, "stereo_camera_calibration.npy")
        with self.fs.open(calib_fpath, "rb") as f:
            data = np.load(f, allow_pickle=True, encoding="latin1")
            # Ugly code since I numpy-saved a dict...
            data = data.tolist()["data"]
            # TODO(andrei): Validate, clean, document, write an example, etc.
            return data

    @cached_property
    def raw_pose_data(self) -> npt.NDArray[RAW_POSE_DTYPE]:
        """Returns the internal raw pose array, which needs manual association with other data types. 100Hz.

        In practice, users should use the camera/LiDAR iterators instead. The raw arrays are full of possibly confusing
        or unexpected aspects, e.g., GPS instead of UNIX time.
        """
        pose_fpath = os.path.join(self._log_root_uri, self._pose_fname)
        with self.fs.open(pose_fpath, "rb") as raw_pose_f:
            return np.load(raw_pose_f)["data"]

    @cached_property
    def continuous_pose_dense(self) -> np.ndarray:
        """Smooth pose in an arbitrary reference frame.

        Not useful for global localization, since each log will have its own coordinate system, but useful for SLAM-like
        evaluation, since you will have a continuous pose trajectory.

        Pose times are UNIX seconds.
        """
        pose_data = []
        for pose in self.raw_pose_data:
            pose_data.append(
                (
                    # TODO(julieta) The overhead of this conversion might be very large at scale. Consider vectorizing
                    gps_seconds_to_utc(pose["capture_time"]).timestamp(),
                    pose["poses_and_differentials_valid"],
                    pose["continuous"]["x"],
                    pose["continuous"]["y"],
                    pose["continuous"]["z"],
                    pose["continuous"]["roll"],
                    pose["continuous"]["pitch"],
                    pose["continuous"]["yaw"],
                )
            )
        pose_index = np.array(sorted(pose_data, key=lambda x: x[0]))

        # # Filter by partitions
        # partitions_mask = self.filter_partitions(pose_index[:, 0], max_delta_s=0.2)
        # pose_index = pose_index[partitions_mask]

        return pose_index

    @cached_property
    def map_relative_poses_dense(self) -> np.ndarray:
        """T x 9 array with unix time, validity, submap ID, and the 6-DoF pose within that submap.

        WARNING:
            - As of 2023-02, the submaps are not 100% globally consistent. Topometric pose accuracy is cm-level, but
            the utms of the maps are noisy for historic reasons.
            - As of 2023-02, some submap IDs may have insufficient bytes.
                - Use `.ljust` to fix that - see the `utm_poses_dense` function for info.
        """
        # Poses contain just the data as a structured numpy array. Each pose object contains a continuous, smooth,
        # log-specific pose, and a map-relative pose. The map relative pose (MRP) takes vehicle points into the
        # current submap, whose ID is indicated by the 'submap' field of the MRP.
        #
        # Be sure to check the 'valid' flag of the poses before using them!
        #
        # The data also has pose and velocity covariance information, but I have never used directly so I don't know
        # if it's well-calibrated.
        pose_data = []
        for pose in self.raw_pose_data:
            # TODO(andrei): Custom, interpretable dtype!
            pose_data.append(
                (
                    # TODO(julieta) consider vectorizing
                    gps_seconds_to_utc(pose["capture_time"]).timestamp(),
                    pose["poses_and_differentials_valid"],
                    pose["map_relative"]["submap"],
                    pose["map_relative"]["x"],
                    pose["map_relative"]["y"],
                    pose["map_relative"]["z"],
                    pose["map_relative"]["roll"],
                    pose["map_relative"]["pitch"],
                    pose["map_relative"]["yaw"],
                )
            )
        pose_index = np.array(
            sorted(pose_data, key=lambda x: x[0]),
            dtype=np.dtype(
                [
                    ("time", np.float64),
                    ("valid", bool),
                    ("submap_id", "|S32"),
                    ("x", np.float64),
                    ("y", np.float64),
                    ("z", np.float64),
                    ("roll", np.float64),
                    ("pitch", np.float64),
                    ("yaw", np.float64),
                ]
            ),
        )

        partitions_mask = self.filter_partitions(pose_index["time"], max_delta_s=0.2)
        pose_index = pose_index[partitions_mask]

        return pose_index

    @cached_property
    def utm_poses_dense(self) -> Tuple[np.ndarray, np.ndarray]:
        """UTM poses for the log, ordered by time.

        Returns:
            A tuple with two elements:
                - An n-long boolean array indicating whether the poses are valid
                - An n-by-3 array with the UTM xy coordinates and altitudes of the poses
        """
        mrp = self.map_relative_poses_dense
        xyzs = rfn.structured_to_unstructured(mrp[["x", "y", "z"]])

        # Handle submap IDs which were truncated upon encoded due to ending with a zero.
        submaps = [UUID(bytes=submap_uuid_bytes.ljust(16, b"\x00")) for submap_uuid_bytes in mrp["submap_id"]]

        try:
            xyzs = self._map.to_utm(xyzs, submaps, strict=False)
        except SubmapPoseNotFoundException as e:
            raise RuntimeError(f"The pose of one of the submaps from log {self.log_id} was not found.") from e

        # Filter by partitions
        partitions_mask = self.filter_partitions(mrp["time"], max_delta_s=0.2)
        mrp = mrp[partitions_mask]
        xyzs = xyzs[partitions_mask]

        return mrp["valid"], xyzs

    @cached_property
    def raw_wgs84_poses(self) -> np.ndarray:
        """Raw WGS84 poses, not optimized offline. 10Hz."""
        wgs84_fpath = os.path.join(self._log_root_uri, self._wgs84_pose_fname)
        with self.fs.open(wgs84_fpath, "rb") as in_compressed_f:
            return np.load(in_compressed_f)["data"]

    @cached_property
    def raw_wgs84_poses_as_utm(self) -> np.ndarray:
        """Returns a N x 3 array of online (non-optimized) UTM poses, computed from raw WGS84 ones, @10Hz, ordered as
        [timestamp, easting, northing].
        """
        wgs84 = self.raw_wgs84_poses
        easting, northing, zn, zl = utm.from_latlon(
            wgs84["latitude"],
            wgs84["longitude"],
            force_zone_letter=UTM_ZONE_LETTER,
            force_zone_number=UTM_ZONE_NUMBER,
        )
        assert zn == UTM_ZONE_NUMBER, f"utm zone number is not {UTM_ZONE_NUMBER}"
        assert zl == UTM_ZONE_LETTER, f"utm zone letter is not {UTM_ZONE_LETTER}"
        return np.vstack([wgs84["timestamp"], easting, northing]).T

    @cached_property
    def raw_wgs84_poses_dense(self) -> np.ndarray:
        """Returns an N x 7 array of online (non-optimized) WGS84 poses, ordered by their UNIX timestamp.

        TODO(andrei): Degrees or radians?

        Rows are: (timestamp_seconds, lon, lat, alt, roll, pitch, yaw (heading)).
        """
        raw = self.raw_wgs84_poses
        wgs84_data = []
        for wgs84 in raw:
            wgs84_data.append(
                (
                    gps_seconds_to_utc(wgs84["timestamp"]).timestamp(),
                    wgs84["longitude"],
                    wgs84["latitude"],
                    wgs84["altitude"],
                    wgs84["roll"],
                    wgs84["pitch"],
                    wgs84["heading"],
                )
            )
        wgs84_data_np = np.array(sorted(wgs84_data, key=lambda x: x[0]))
        return wgs84_data_np

    def get_image(self, cam_name: CamName, idx: int) -> CameraImage:
        """Loads a camera image by index in log, used in torch data loading."""
        index_entry = self.get_cam_geo_index(cam_name)[idx]
        rel_path = index_entry["rel_path"].strip()
        fpath = os.path.join(self.get_cam_root(cam_name), rel_path)

        # NOTE(andrei): ~35-40ms to open a LOCAL webp image, 120-200ms to open an S3 webp image. PyTorch does not like
        # high latency data loading, so we will need to rewrite parts of the dataloader to perform true async reading
        # separate from the dataloader parallelism.
        with self.fs.open(fpath, "rb") as f:
            bts = f.read()
            with io.BytesIO(bts) as fbuf:
                with Image.open(fbuf) as img:
                    image_np = np.array(img)
            return CameraImage(
                image=image_np,
                cam_name=cam_name,
                capture_timestamp=gps_to_unix_timestamp(index_entry["img_time"]),
                gain_db=index_entry["gain_db"],
                shutter_time_s=index_entry["shutter_s"],
            )

    def camera_iterator(self, cam_name: CamName, start: int = 0, step: int = 1) -> Iterator[CameraImage]:
        assert start >= 0
        assert step > 0
        index = self.get_cam_geo_index(cam_name)
        for row_index in range(start, len(index), step):
            yield self.get_image(cam_name, row_index)

    def get_lidar(self, idx: int) -> LiDARFrame:
        """Loads the LiDAR scan for the given relative path, used in torch data loading."""
        index_entry = self.get_lidar_geo_index()[idx]
        rel_path = index_entry["rel_path"].strip()
        if self._index_version == 0:
            # TODO(julieta) re-dump the indices so that they include the correct path and remove this once we get rid of
            # v0 indexes.
            rel_path += "z.lz4"

        fpath = os.path.join(self.lidar_root, rel_path)
        with self.fs.open(fpath, "rb") as f_compressed:
            with lz4.frame.open(f_compressed, "rb") as f:
                npf = np.load(f)
                return LiDARFrame(
                    xyz_continuous=npf["points"],
                    xyz_sensor=npf["points_H_sensor"],
                    intensity=npf["intensity"],
                    point_times=npf["seconds"],
                )

    # def lidar_iterator(self):
    #     index = self.get_lidar_geo_index()
    #     for row in index.itertuples():
    #         lidar_fpath = os.path.join(self.lidar_root, row.lidar_fpath)
    #         with self.fs.open(lidar_fpath, "rb") as compressed_f:
    #             with lz4.frame.open(compressed_f, "rb") as f:
    #                 npf = np.load(f)
    #                 yield LiDARFrame(
    #                     # TODO add remaining fields like raw power, etc.
    #                     xyz_continuous=npf["points"],
    #                     xyz_sensor=npf["points_H_sensor"],
    #                     intensity=npf["intensity"],
    #                     point_times=npf["seconds"],
    #                 )
