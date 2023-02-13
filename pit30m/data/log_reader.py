import csv
import os
import time
from dataclasses import dataclass
from functools import cached_property, lru_cache
from urllib.parse import urlparse
from uuid import UUID

import fsspec
import ipdb
import lz4
import numpy as np
import pandas as pd
from PIL import Image

from pit30m.camera import CamName
from pit30m.data.submap import Map


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



class LogReader:

    def __init__(
        self,
        log_root_uri: str,
        pose_fname: str = "all_poses.npz.lz4",
        wgs84_pose_fname: str = "wgs84.npz.lz4",
        map: Map = None,
        index_version: int = 0,
    ):
        """Lightweight, low-level S3-aware utility for interacting with a specific log.

        Devkit users should consider using existing higher-level APIs, such as `pit30m.torch.dataset.Pit30MDataset`
        unless they have a specific use case that requires low-level access to the data.

        Specifically, using abstractions like pytorch DataLoaders can dramatically improve throughput, e.g., going from
        20-30 images per second to 100+.
        """
        self._log_root_uri = log_root_uri.rstrip("/")
        self._pose_fname = pose_fname
        self._wgs84_pose_fname = wgs84_pose_fname
        if map is None:
            # By default try to build map with data relative to the log root
            log_parent = os.path.dirname(self._log_root_uri)
            map = Map.from_submap_utm_uri(os.path.join(log_parent, "submap_utm.pkl"))
        self._map = map
        self._index_version = index_version


    @property
    def log_id(self):
        return os.path.basename(self._log_root_uri)

    @property
    def cam_root(self):
        """Root for all the camera data (each camera is a subdirectory)."""
        return os.path.join(self._log_root_uri, "cameras")

    @property
    def lidar_root(self):
        # Simpler than cameras since there's always a single lidar.
        return os.path.join(self._log_root_uri, "lidars", VELODYNE_NAME)

    def get_cam_root(self, cam_name: CamName):
        assert isinstance(cam_name, CamName)
        return os.path.join(self._log_root_uri, "cameras", cam_name.value)

    @lru_cache(maxsize=16)
    def get_lidar_geo_index(self):
        """Returns a lidar index of dtype LIDAR_INDEX_V0_0_DTYPE.

        WARNING: 'rel_path' entries in indexes may be padded with spaces on the right since they are fixed-width
        strings. If you need to use them directly, make sure you use `.strip()` to remove the spaces.
        """
        index_fpath = os.path.join(self.lidar_root, "index", "wgs84.csv")
        fs = fsspec.filesystem(urlparse(index_fpath).scheme)
        if not fs.exists(index_fpath):
            raise ValueError(f"Index file not found: {index_fpath}!")

        with fs.open(index_fpath, "r") as f:
            return pd.read_csv(f)

    @lru_cache(maxsize=16)
    def get_cam_geo_index(self, cam_name: str) -> np.ndarray:
        """Returns a camera index of dtype CAM_INDEX_V0_0_DTYPE."""
        index_fpath = os.path.join(self.get_cam_root(cam_name), "index", f"index_v{self._index_version}.npz")
        fs = fsspec.filesystem(urlparse(index_fpath).scheme)
        if not fs.exists(index_fpath):
            raise ValueError(f"Index file not found: {index_fpath}!")

        with fs.open(index_fpath, "rb") as f:
            return np.load(f)["index"]

    def calib(self):
        calib_fpath = os.path.join(self._log_root_uri, "mono_camera_calibration.npy")
        with fsspec.open(calib_fpath, "rb") as f:
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
        with fsspec.open(calib_fpath, "rb") as f:
            data = np.load(f, allow_pickle=True, encoding="latin1")
            # Ugly code since I numpy-saved a dict...
            data = data.tolist()["data"]
            # TODO(andrei): Validate, clean, document, write an example, etc.
            return data


    @cached_property
    def raw_pose_data(self) -> np.ndarray:
        """Returns the raw pose array, which needs manual association with other data types. 100Hz.

        In practice, users should use the camera/LiDAR iterators instead.

        TODO(andrei): Document dtype (users now have to manually check dtype to learn).
        """
        pose_fpath = os.path.join(self._log_root_uri, self._pose_fname)
        with fsspec.open(pose_fpath, "rb") as in_compressed_f:
            with lz4.frame.open(in_compressed_f, "rb") as wgs84_f:
                return np.load(wgs84_f)["data"]

    @cached_property
    def continuous_pose_dense(self) -> np.ndarray:
        """Smooth pose in an arbitrary reference frame.

        Not useful for global localization, since each log will have its own coordinate system, but useful for SLAM-like
        evaluation, since you will have a continuous pose trajectory.
        """
        pose_data = []
        # TODO(andrei): Document the timestamps carefully.
        for pose in self.raw_pose_data:
            pose_data.append((pose["capture_time"],
                            pose["poses_and_differentials_valid"],
                            pose["continuous"]["x"],
                            pose["continuous"]["y"],
                            pose["continuous"]["z"],
                            pose["continuous"]["roll"],
                            pose["continuous"]["pitch"],
                            pose["continuous"]["yaw"]))
        pose_index = np.array(sorted(pose_data, key=lambda x: x[0]))
        return pose_index

    @cached_property
    def map_relative_poses_dense(self) -> np.ndarray:
        """T x 9 array with time, validity, submap ID, and the 6-DoF pose within that submap.

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
        # XXX(andrei): Document the timestamps carefully. Remember that GPS time, if applicable, can be confusing!
        for pose in self.raw_pose_data:
            # TODO(andrei): Custom, interpretable dtype!
            pose_data.append((
                pose["capture_time"],
                pose["poses_and_differentials_valid"],
                pose["map_relative"]["submap"],
                pose["map_relative"]["x"],
                pose["map_relative"]["y"],
                pose["map_relative"]["z"],
                pose["map_relative"]["roll"],
                pose["map_relative"]["pitch"],
                pose["map_relative"]["yaw"],
            ))
        pose_index = np.array(
            sorted(pose_data, key=lambda x: x[0]),
            dtype=np.dtype([
                ("time", np.float64),
                ("valid", np.bool),
                ("submap_id", "|S32"),
                ("x", np.float64),
                ("y", np.float64),
                ("z", np.float64),
                ("roll", np.float64),
                ("pitch", np.float64),
                ("yaw", np.float64),
            ]),
        )
        return pose_index

    @cached_property
    def utm_poses_dense(self) -> np.ndarray:
        """UTM poses for the log, ordered by time.

        TODO(andrei): Update to provide altitude.
        """
        mrp = self.map_relative_poses_dense
        xyzs = np.stack((mrp["x"], mrp["y"], mrp["z"]), axis=1)
        # Handle submap IDs which were truncated upon encoded due to ending with a zero.
        submaps = [UUID(bytes=submap_uuid_bytes.ljust(16, b"\x00")) for submap_uuid_bytes in mrp["submap_id"]]
        return self._map.to_utm(xyzs, submaps)

    @cached_property
    def raw_wgs84_poses(self) -> np.ndarray:
        """Raw WGS84 poses, not optimized offline. 10Hz."""
        wgs84_fpath = os.path.join(self._log_root_uri, self._wgs84_pose_fname)
        with fsspec.open(wgs84_fpath, "rb") as in_compressed_f:
            with lz4.frame.open(in_compressed_f, "rb") as wgs84_f:
                return np.load(wgs84_f)["data"]

    @cached_property
    def raw_wgs84_poses_dense(self) -> np.ndarray:
        """Returns an N x 7 array of online (non-optimized) WGS84 poses, ordered by timestamp.

        TODO(andrei): Degrees or radians?

        Rows are: (timestamp_seconds, lon, lat, alt, roll, pitch, yaw (heading)).
        """
        raw = self.raw_wgs84_poses
        wgs84_data = []
        for wgs84 in raw:
            wgs84_data.append((wgs84["timestamp"],
                            wgs84["longitude"],
                            wgs84["latitude"],
                            wgs84["altitude"],
                            wgs84["roll"],
                            wgs84["pitch"],
                            wgs84["heading"]))
        wgs84_data = np.array(sorted(wgs84_data, key=lambda x: x[0]))
        return wgs84_data

    def get_image(self, cam_name: CamName, idx: int) -> CameraImage:
        """Loads a camera image by index in log, used in torch data loading."""
        index_entry = self.get_cam_geo_index(cam_name)[idx]
        rel_path = index_entry["rel_path"]
        fpath = os.path.join(self.get_cam_root(cam_name), rel_path)

        # NOTE(andrei): ~35-40ms to open a LOCAL webp image, 120-200ms to open an S3 webp image. PyTorch does not like
        # high latency data loading, so we will need to rewrite parts of the dataloader to perform true async reading
        # separate from the dataloader parallelism.
        with fsspec.open(fpath, "rb") as f:
            image_np = np.asarray(Image.open(f))
            return CameraImage(
                image=image_np,
                cam_name=cam_name.value,
                capture_timestamp=index_entry["img_time"],
                gain_db=index_entry["gain_db"],
                shutter_time_s=index_entry["shutter_s"],
            )

    def get_lidar(self, rel_path: str) -> LiDARFrame:
        """Loads the LiDAR scan for the given relative path, used in torch data loading."""
        fpath = os.path.join(self.get_cam_root(CamName.MIDDLE_FRONT_WIDE), rel_path)
        with fsspec.open(fpath, "rb") as f_compressed:
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
    #         with fsspec.open(lidar_fpath, "rb") as compressed_f:
    #             with lz4.frame.open(compressed_f, "rb") as f:
    #                 npf = np.load(f)
    #                 yield LiDARFrame(
    #                     # TODO add remaining fields like raw power, etc.
    #                     xyz_continuous=npf["points"],
    #                     xyz_sensor=npf["points_H_sensor"],
    #                     intensity=npf["intensity"],
    #                     point_times=npf["seconds"],
    #                 )
