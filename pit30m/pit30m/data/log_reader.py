import csv
import lz4
from dataclasses import dataclass
import os
from functools import cached_property, lru_cache
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import fsspec
from PIL import Image

from pit30m.camera import CamName

@dataclass
class CameraImage:
    # TODO consider replacing the cam with a Camera object which can be used to get the intrinsics and stuff
    cam_name: str
    capture_timestamp: float
    img: np.ndarray
    shutter_time_s: float
    gain_db: float


class Map:

    @staticmethod
    def from_uri(uri: str) -> "Map":
        pass

    def __init__(self, map_data):
        """Represents the Pit30M topometric map.

        To get a "global" pose from a log pose, you need to take its map-relative pose, and then compose that with the
        map's UTM coordinates.

        This would break if the map spanned multiple UTM zones, in which case we'd be forced to use WGS84 spherical
        coordinates or ECEF Euclidean coordinates for everything but it doesn't. Pittsburgh and its surrounding areas
        fit comfortably within UTM Zone 17N, i.e., EPSG:32617.

        TODO(andrei): Should we bundle this info in the devkit itself for fast retrieval? It should be pretty tiny.
        At the very least we should cache it somewhere like in "~/.cache/pit30m".
        """
        pass


class LogReader:

    def __init__(
        self,
        log_root_uri: str,
        pose_fname: str = "all_poses.npz.lz4",
        wgs84_pose_fname: str = "wgs84.npz.lz4",
    ):
        """Low-level S3-aware utility for interacting with a specific log.

        Devkit users should consider using existing higher-level APIs, such as `pit30m.torch.dataset.Pit30MDataset`
        unless they have a specific use case that requires low-level access to the data.

        Specifically, using abstractions like pytorch DataLoaders can dramatically improve throughput, e.g., going from
        20-30 images per second to 100+.
        """
        self._log_root_uri = log_root_uri
        self._pose_fname = pose_fname
        self._wgs84_pose_fname = wgs84_pose_fname

    @property
    def cam_root(self):
        return os.path.join(self._log_root_uri, "cameras")


    def get_cam_root(self, cam_name: CamName):
        assert isinstance(cam_name, CamName)
        return os.path.join(self._log_root_uri, "cameras", cam_name.value)


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
            data = np.load(f)
            import ipdb; ipdb.set_trace()
            print()
            return data


    @cached_property
    def raw_poses(self) -> np.ndarray:
        """Returns the raw pose array, which needs manual association with other data types. 100Hz.

        In practice, users should use the camera/LiDAR iterators instead.

        TODO(andrei): Document dtype (users now have to manually check dtype to learn).
        """
        pose_fpath = os.path.join(self._log_root_uri, self._pose_fname)
        with fsspec.open(pose_fpath, "rb") as in_compressed_f:
            with lz4.frame.open(in_compressed_f, "rb") as wgs84_f:
                return np.load(wgs84_f)["data"]

    @cached_property
    def wgs84_poses(self) -> np.ndarray:
        """Raw WGS84 poses, not optimized offline. 10Hz."""
        wgs84_fpath = os.path.join(self._log_root_uri, self._wgs84_pose_fname)
        with fsspec.open(wgs84_fpath, "rb") as in_compressed_f:
            with lz4.frame.open(in_compressed_f, "rb") as wgs84_f:
                return np.load(wgs84_f)["data"]

    @cached_property
    def wgs84_poses_dense(self) -> np.ndarray:
        """Returns an N x 7 array of WGS84 poses, ordered by timestamp.

        TODO(andrei): Degrees or radians?

        Rows are: (timestamp_seconds, lon, lat, alt, roll, pitch, yaw).
        """
        raw = self.wgs84_poses
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


    def camera_iterator(self, cam_name: CamName):
        # Iterator over camera images with metadata
        # TODO(andrei): requires an index to have been built.
        index = self.get_cam_geo_index(cam_name)
        for row in index.itertuples():
            img_fpath = os.path.join(self.get_cam_root(cam_name), row.img_fpath_in_cam)
            with fsspec.open(img_fpath, "rb") as f:
                img = Image.open(f)

            yield CameraImage(
                cam_name=cam_name,
                capture_timestamp=row.capture_seconds,
                img=np.asarray(img),
                gain_db=row.gain_db,
                shutter_time_s=row.shutter_seconds,
            )