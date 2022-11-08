import csv
import lz4
from dataclasses import dataclass
import ipdb
import os
from functools import cached_property, lru_cache
from urllib.parse import urlparse
import pandas as pd
import numpy as np
from uuid import UUID
import fsspec
from PIL import Image

from pit30m.camera import CamName
from pit30m.data.submap import Map

@dataclass
class CameraImage:
    # TODO consider replacing the cam with a Camera object which can be used to get the intrinsics and stuff
    cam_name: str
    capture_timestamp: float
    img: np.ndarray
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


    @property
    def log_id(self):
        return os.path.basename(self._log_root_uri)

    @property
    def cam_root(self):
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
        index_fpath = os.path.join(self.lidar_root, "index", "wgs84.csv")
        fs = fsspec.filesystem(urlparse(index_fpath).scheme)
        if not fs.exists(index_fpath):
            raise ValueError(f"Index file not found: {index_fpath}!")

        with fs.open(index_fpath, "r") as f:
            return pd.read_csv(f)

    @lru_cache(maxsize=16)
    def get_cam_geo_index(self, cam_name: str):
        index_fpath = os.path.join(self.get_cam_root(cam_name), "index", "wgs84.csv")
        fs = fsspec.filesystem(urlparse(index_fpath).scheme)
        if not fs.exists(index_fpath):
            raise ValueError(f"Index file not found: {index_fpath}!")

        with fs.open(index_fpath, "r") as f:
            return pd.read_csv(f)

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
            # import ipdb; ipdb.set_trace()
            # print()
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
        # XXX(andrei): Document the timestamps carefully. Remember that GPS time, if applicable, can be confusing!
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
    def map_relative_pose_dense(self) -> np.ndarray:
        """T x 9 array with time, validity, submap ID, and the 6-DoF pose within that submap."""
        pose_data = []
        # XXX(andrei): Document the timestamps carefully. Remember that GPS time can be confusing!
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
        mrp = self.map_relative_pose_dense
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

        Rows are: (timestamp_seconds, lon, lat, alt, roll, pitch, yaw).
        """
        raw = self.raw_wgs84_poses
        wgs84_data = []
        for wgs84 in raw:
            wgs84_data.append((wgs84["timestamp"],
                            wgs84["longitude"],
                            wgs84["latitude"],
                            wgs84["altitude"],
                            wgs84["heading"],
                            wgs84["pitch"],
                            wgs84["roll"]))
        wgs84_data = np.array(sorted(wgs84_data, key=lambda x: x[0]))
        return wgs84_data

    def get_image(self, cam_name: CamName, timestamp: float):
        """Returns the image for the given camera and timestamp."""
        # NOTE(andrei): For dataloader.
        pass

    def get_lidar(self, timestamp: float):
        """Returns the LiDAR scan for the given timestamp."""
        # NOTE(andrei): For dataloader.
        pass

    def lidar_iterator(self):
        index = self.get_lidar_geo_index()
        for row in index.itertuples():
            lidar_fpath = os.path.join(self.lidar_root, row.lidar_fpath)
            with fsspec.open(lidar_fpath, "rb") as compressed_f:
                with lz4.frame.open(compressed_f, "rb") as f:
                    npf = np.load(f)
                    # print(npf.files)
                    yield LiDARFrame(
                        # TODO add remaining fields like raw power, etc.
                        xyz_continuous=npf["points"],
                        xyz_sensor=npf["points_H_sensor"],
                        intensity=npf["intensity"],
                        point_times=npf["seconds"],
                    )

    # def raw_camera_iterator(self, cam_name: CamName):
    #     pass

    def camera_iterator(self, cam_name: CamName):
        """Iterator over camera images with metadata"""
        # TODO(andrei): requires an index to have been built.
        index = self.get_cam_geo_index(cam_name)
        cam_root = self.get_cam_root(cam_name)
        for row in index.itertuples():
            img_fpath = os.path.join(cam_root, row.img_fpath_in_cam)
            with fsspec.open(img_fpath, "rb") as f:
                img = Image.open(f)

            yield CameraImage(
                cam_name=cam_name,
                capture_timestamp=row.capture_seconds,
                img=np.asarray(img),
                gain_db=row.gain_db,
                shutter_time_s=row.shutter_seconds,
            )