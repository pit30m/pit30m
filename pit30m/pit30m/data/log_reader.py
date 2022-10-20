

import csv
from dataclasses import dataclass
import os
from functools import lru_cache
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import fsspec
from PIL import Image

from pit30m.camera import CamNames

@dataclass
class CameraImage:
    # TODO consider replacing the cam with a Camera object which can be used to get the intrinsics and stuff
    cam_name: str
    capture_timestamp: float
    img: np.ndarray
    shutter_time_s: float
    gain_db: float


class LogReader:

    def __init__(self, log_root_uri: str):
        """Low-level S3-aware utility for interacting with a specific log.

        Devkit users should consider using existing higher-level APIs, such as `pit30m.torch.dataset.Pit30MDataset`
        unless they have a specific use case that requires low-level access to the data.

        Specifically, using abstractions like pytorch DataLoaders can dramatically improve throughput, e.g., going from
        20-30 images per second to 100+.
        """
        self._log_root_uri = log_root_uri

    @property
    def cam_root(self):
        return os.path.join(self._log_root_uri, "cameras")


    def get_cam_root(self, cam_name: CamNames):
        assert isinstance(cam_name, CamNames)
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


    def camera_iterator(self, cam_name: CamNames):
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