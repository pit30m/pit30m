
import geojson
import os
import csv
from pickle import UnpicklingError
import numpy as np
import lz4
import fire
import fsspec
from tqdm import tqdm
from typing import Optional, List
from urllib.parse import urlparse, urljoin
from pit30m.camera import CAM_NAMES

class MonkeyWrench:

    def __init__(self, dataset_root: str = "s3://the/bucket") -> None:
        """Dataset administration tool for the Pit30M dataset (geo-indexing, data integrity checks, etc.)

        For regular end users, `pit30m.cli` is almost always what you should use.

        Users of the dataset won't have permissions to modify the official dataset, but they can use this tool to create
        custom indexes and subsets of the dataset.
        """
        self._root = dataset_root


    def index(self, log_id: str, out_index_fpath: Optional[str] = None):
        """Create index files for the raw data in the dataset.

        At dump time, the dataset just contained numbered image files with small associated metadata files. This meant
        it was basically impossible to find images by GPS location or timestamp. This tool creates indexes that allow
        for fast lookups by GPS location or timestamp.

        Building large indexes from scratch can take a while, even on the order of hours on some machines. We are
        dealing with roughly 400 million images, after all.
        """
        self.index_all_cameras(log_id=log_id, out_index_fpath=out_index_fpath)

    def merge_indexes(self, log_ids: List[str]):
        """Given a list of log IDs with indexed data, merge the indexes into a single index."""
        # TODO(andrei): Implement this.
        ...

    def index_all_cameras(self, log_id: str, out_index_fpath: Optional[str] = None):
        """Create an index of the images in the given log.

        This is useful for quickly finding images in a given region, or for finding the closest image to a given GPS
        location.

        Building large indexes from scratch can take a while, even on the order of hours on some machines.
        """
        for cam in CAM_NAMES:
            self.index_camera(log_id=log_id, cam_name=cam, out_index_fpath=out_index_fpath)

    def index_lidar(self, log_id: str, lidar_name: str = "hdl64e_12_middle_front_roof", out_index_fpath: Optional[str] = None):
        """Same as 'index_all_cameras', except for the LiDAR sweeps."""
        pass

    def index_camera(self, log_id: str, cam_name: str, out_index_fpath: Optional[str] = None):
        """Please see `index_all_cameras` for info."""
        in_fs = fsspec.filesystem(urlparse(self._root).scheme)

        log_root = os.path.join(self._root, log_id.lstrip("/"))
        cam_dir = os.path.join(log_root, "cameras", cam_name.lstrip("/"))

        # if out_index_fpath is None:
        #     out_index_fpath = os.path.join(log_root, "index", f"{cam_name}.geojson")
        if out_index_fpath is None:
            out_index_fpath = os.path.join(cam_dir, "index")

        regular_index_fpath = os.path.join(out_index_fpath, "index.csv")
        wgs84_index_fpath = os.path.join(out_index_fpath, "wgs84.csv")
        out_fs = fsspec.filesystem(urlparse(out_index_fpath).scheme)

        # if not os.path.exists(os.path.join(log_root, "all_poses.npz.lz4")):
        #     continue

        poses_fpath = os.path.join(log_root, "all_poses.npz.lz4")
        with in_fs.open(poses_fpath, "rb") as in_compressed_f:
            with lz4.frame.open(in_compressed_f, "rb") as wgs84_f:
                poses = np.load(wgs84_f)["data"]

        # TODO(andrei): Check if these WGS84 values are raw or adjusted based on localization since it makes a big
        # impact on any localization or reconstruction work. For indexing we should be fine either way.
        wgs84_fpath = os.path.join(log_root, "wgs84.npz.lz4")
        with in_fs.open(wgs84_fpath, "rb") as in_compressed_f:
            with lz4.frame.open(in_compressed_f, "rb") as wgs84_f:
                wgs84s = np.load(wgs84_f)["data"]

        # NOTE(andrei): WGS84 data is coarser, 10Hz, not 100Hz.
        wgs84_data = []
        for wgs84 in wgs84s:
            wgs84_data.append((wgs84["timestamp"],
                            wgs84["longitude"],
                            wgs84["latitude"],
                            wgs84["altitude"],
                            wgs84["heading"],
                            wgs84["pitch"],
                            wgs84["roll"]))
        wgs84_data = np.array(sorted(wgs84_data, key=lambda x: x[0]))
        wgs84_times = np.array(wgs84_data[:, 0])

        pose_data = []
        for pose in poses:
            pose_data.append((pose["capture_time"],
                            pose["poses_and_differentials_valid"],
                            pose["continuous"]["x"],
                            pose["continuous"]["y"],
                            pose["continuous"]["z"]))
        pose_index = np.array(sorted(pose_data, key=lambda x: x[0]))
        pose_times = np.array(pose_index[:, 0])
        # print(pose_times.shape)

        index = []
        for entry in tqdm(in_fs.glob(os.path.join(cam_dir, "*", "*.webp"))):
            img_fpath = entry
            meta_fpath = entry.replace(".day", ".meta").replace(".night", ".meta").replace(".webp", ".npy")
            img_fpath_in_cam = "/".join(img_fpath.split("/")[-2:])

            with in_fs.open(meta_fpath) as meta_f:
                try:
                    # The tolist actually extracts a dict...
                    meta = np.load(meta_f, allow_pickle=True).tolist()
                    index.append((meta["capture_seconds"], img_fpath_in_cam, meta["shutter_seconds"],
                                meta["sequence_counter"], meta["gain_db"]))
                except UnpicklingError:
                    # TODO(andrei): Remove this hack once you re-extract with your ETL code
                    # hack for corrupted metadata, which should be fixed in the latest ETL
                    print(f"WARNING: Error reading {meta_fpath = }")
                    continue
                except ModuleNotFoundError:
                    # TODO(andrei): Remove this one too
                    # seems like corrupted pickles can trigger this, oof
                    print(f"WARNING: Error reading {meta_fpath = }")
                    continue

        # Sort by the capture time so we can easily search images by a timestamp
        image_index = sorted(index, key=lambda x: x[0])

        imgs_with_pose = []
        imgs_with_wgs84 = []
        for img_row in tqdm(image_index):
            img_time = float(img_row[0])
            # TODO(andrei): Within a few ms at worst, poses are evenly spaced. We can use
            # arithmetic to dramatically constrain our search range to turn log n binary search
            # into a constant time look-up.
            #
            # TODO(andrei): Interpolate poses linearly. Poses are 100Hz so nearest-neighbor lookups can
            # be at most 10-ish ms off which can cause 0.33m of error for 33 mps driving (120kph) in
            # a worst-case scenario, so not trivial.
            #
            # Fortunately, for such small intervals, linear interpolation should be OK.
            pose_idx = np.searchsorted(pose_times, img_time, side="left")
            pose_idx = pose_idx + 1
            if pose_idx >= len(pose_index):
                pose_idx = len(pose_index) - 1
            delta_s = abs(img_time - pose_times[pose_idx])
            if delta_s > 0.1:
                print(f"WARNING: {img_time = } does not have a valid pose in this log [{delta_s = }]")
            else:
                imgs_with_pose.append((pose_index[pose_idx], img_row))

            wgs84_idx = np.searchsorted(wgs84_times, img_time, side="left")
            wgs84_idx += 1
            if wgs84_idx >= len(wgs84_data):
                wgs84_idx = len(wgs84_data) - 1

            delta_wgs_s = abs(img_time - wgs84_times[wgs84_idx])
            if delta_wgs_s > 0.5:
                print(f"WARNING: {img_time = } does not have a valid WGS84 coordinate in this log [{delta_s = }]")
            else:
                imgs_with_wgs84.append((wgs84_data[wgs84_idx], img_row))


        # TODO(andrei): Write timestamp index and WGS84 index.
        # TODO(andrei): Don't write CP, because then you get nonsense when aggregating across logs.
        # with open(out_index_fpath, "w", newline="") as csvfile:
        #     spamwriter = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     spamwriter.writerow(["capture_time", "poses_and_differentials_valid", "x", "y", "z", "capture_seconds", "img_fpath_in_cam", "shutter_seconds", "sequence_counter", "gain_db"])
        #     for pose_row, img_row in imgs_with_pose:
        #         spamwriter.writerow(list(pose_row) + list(img_row))

        if not out_fs.exists(out_index_fpath):
            out_fs.mkdir(out_index_fpath)
        with out_fs.open(wgs84_index_fpath, "w", newline="") as csvfile:
            spamwriter = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["timestamp", "longitude", "latitude", "altitude", "heading", "pitch", "roll", "capture_seconds", "img_fpath_in_cam", "shutter_seconds", "sequence_counter", "gain_db"])
            for wgs84_row, img_row in imgs_with_wgs84:
                spamwriter.writerow(list(wgs84_row) + list(img_row))


if __name__ == "__main__":
    fire.Fire(MonkeyWrench)