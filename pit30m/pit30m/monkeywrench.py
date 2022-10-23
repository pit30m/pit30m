
import os
import csv
import geojson
from pickle import UnpicklingError
import numpy as np
import lz4
import fire
import fsspec
from tqdm import tqdm
from typing import Optional, List
from urllib.parse import urlparse, urljoin
from pit30m.camera import CamName
from pit30m.data.log_reader import LogReader
from pit30m.indexing import associate
from joblib import Parallel, delayed


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
        for cam in CamName:
            self.index_camera(log_id=log_id, cam_name=cam, out_index_fpath=out_index_fpath)

    def index_lidar(
        self,
        log_id: str,
        lidar_name: str = "hdl64e_12_middle_front_roof",
        sweep_time_convention: str = "end",
        out_index_fpath: Optional[str] = None
    ):
        """Same as 'index_all_cameras', except for the LiDAR sweeps."""
        in_fs = fsspec.filesystem(urlparse(self._root).scheme)
        out_fs = fsspec.filesystem(urlparse(out_index_fpath).scheme)
        log_root = os.path.join(self._root, log_id.lstrip("/"))
        log_reader = LogReader(log_root_uri=log_root)
        lidar_dir = os.path.join(log_root, "lidars", lidar_name.lstrip("/"))
        if out_index_fpath is None:
            out_index_fpath = os.path.join(lidar_dir, "index")

        wgs84_index_fpath = os.path.join(out_index_fpath, "wgs84.csv")
        dumped_ts_fpath = os.path.join(lidar_dir, "timestamps.npz.lz4")

        def _get_lidar_time(lidar_uri):
            try:
                with in_fs.open(lidar_uri, "rb") as compressed_f:
                    with lz4.frame.open(compressed_f, "rb") as f:
                        lidar_data = np.load(f)
                        point_times = lidar_data["seconds"]
                        return point_times.min(), point_times.max(), point_times.mean(), np.median(point_times), lidar_data["points"].shape
                        # sweep_delta_s = point_times.max() - point_times.min()
            except EOFError:
                # Some files may be corrupted? I'll need to triple check this with the diagnostic tools.
                print("EOFError", lidar_uri)
                return None

        # TODO(andrei): This seems difficult to leverage as the number of timestamps seems to differ from the number of
        # dumped sweeps, so aligning the two would be challenging. Perhaps I could just use this data to check some of
        # my assumptions later.
        #
        # with in_fs.open(dumped_ts_fpath, "rb") as compressed_f:
        #     with lz4.frame.open(compressed_f, "rb") as f:
        #         timestamps = np.load(f)["data"]
        #         import ipdb; ipdb.set_trace()
        #         print()

        sample_uris = in_fs.glob(os.path.join(lidar_dir, "*", "*.npz.lz4"))
        # sample_uris = sample_uris[:1000]
        pool = Parallel(n_jobs=-1, verbose=10)
        time_stats = pool(delayed(_get_lidar_time)(lidar_uri) for lidar_uri in sample_uris)
        wgs84 = log_reader.wgs84_poses_dense

        sweep_times_raw = []
        valid_sample_uris = []
        for sample_uri, time_stat_entry in zip(sample_uris, time_stats):
            if time_stat_entry is None:
                print(f"WARNING: Failed to read {sample_uri}")
                continue
            (min_s, max_s, mean_s, med_s, shape) = time_stat_entry
            sweep_delta_s = max_s - min_s
            if abs(sweep_delta_s - 0.1) > 0.01:
                print(f"{sample_uri}: sweep_delta_s = {sweep_delta_s:.4f}s | pcd.{shape = }")

            valid_sample_uris.append(sample_uri)
            if sweep_time_convention == "end":
                sweep_times_raw.append(max_s)
            elif sweep_time_convention == "start":
                sweep_times_raw.append(min_s)
            elif sweep_time_convention == "mean":
                sweep_times_raw.append(mean_s)
            elif sweep_time_convention == "median":
                sweep_times_raw.append(med_s)
            else:
                raise ValueError("Unknown sweep time convention: " + sweep_time_convention)

        sweep_times = np.array(sweep_times_raw)
        del sample_uri

        # TODO(andrei): Index by MRP!
        # poses = log_reader.raw_poses
        # pose_data = []
        # for pose in poses:
        #     pose_data.append((pose["capture_time"],
        #                     pose["poses_and_differentials_valid"],
        #                     pose["continuous"]["x"],
        #                     pose["continuous"]["y"],
        #                     pose["continuous"]["z"]))
        # pose_index = np.array(sorted(pose_data, key=lambda x: x[0]))
        # pose_times = np.array(pose_index[:, 0])

        wgs84_times = wgs84[:, 0]
        wgs84_corr_idx = associate(sweep_times, wgs84_times)
        wgs84_delta = abs(wgs84[wgs84_corr_idx, 0] - sweep_times)
        # Recall WGS84 messages are at 10Hz so we have to be a bit more lax than when checking pose assoc
        bad_offs = wgs84_delta > 0.10
        print(bad_offs.sum(), "bad offsets")
        if bad_offs.sum() > 0:
            print(np.where(bad_offs))
            print()

        lidars_with_wgs84 = []
        assert len(sweep_times) == len(bad_offs) == len(wgs84_corr_idx)
        assert len(valid_sample_uris) == len(sweep_times)
        for sweep_uri, sweep_time, bad_off, wgs84_idx in zip(valid_sample_uris, sweep_times, bad_offs, wgs84_corr_idx):
            if bad_off:
                # TODO Should we flag these in the index?
                continue

            lidar_fpath = "/".join(sweep_uri.split("/")[-2:])

            # img row would include capture seconds, path, then other elements
            # imgs_with_wgs84.append((wgs84_data[wgs84_idx], img_row))
            lidars_with_wgs84.append((wgs84[wgs84_idx, :], (sweep_time, lidar_fpath)))


        # NOTE(andrei): For some rare sweeps (most first sweeps in a log) this will have gaps.
        # NOTE(andrei): The LiDAR is motion-compensated. TBD which timestamp is the canonical one.
        if not out_fs.exists(out_index_fpath):
            out_fs.mkdir(out_index_fpath)
        with out_fs.open(wgs84_index_fpath, "w", newline="") as csvfile:
            spamwriter = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([
                "timestamp", "longitude", "latitude", "altitude", "heading", "pitch", "roll",
                f"sweep_seconds_{sweep_time_convention}", "lidar_fpath"])
            for wgs84_row, lidar_row in lidars_with_wgs84:
                spamwriter.writerow(list(wgs84_row) + list(lidar_row))


    def index_camera(self, log_id: str, cam_name: CamName, out_index_fpath: Optional[str] = None):
        """Please see `index_all_cameras` for info."""
        in_fs = fsspec.filesystem(urlparse(self._root).scheme)

        log_root = os.path.join(self._root, log_id.lstrip("/"))
        log_reader = LogReader(log_root_uri=log_root)
        cam_dir = os.path.join(log_root, "cameras", cam_name.value.lstrip("/"))

        # if out_index_fpath is None:
        #     out_index_fpath = os.path.join(log_root, "index", f"{cam_name}.geojson")
        if out_index_fpath is None:
            out_index_fpath = os.path.join(cam_dir, "index")

        regular_index_fpath = os.path.join(out_index_fpath, "index.csv")
        wgs84_index_fpath = os.path.join(out_index_fpath, "wgs84.csv")
        out_fs = fsspec.filesystem(urlparse(out_index_fpath).scheme)

        # if not os.path.exists(os.path.join(log_root, "all_poses.npz.lz4")):
        #     continue

        poses = log_reader.raw_poses

        # TODO(andrei): Check if these WGS84 values are raw or adjusted based on localization since it makes a big
        # impact on any localization or reconstruction work. For indexing we should be fine either way.
        #
        # NOTE(andrei): WGS84 data is coarser, 10Hz, not 100Hz.
        wgs84_data = log_reader.wgs84_poses_dense
        wgs84_times = np.array(wgs84_data[:, 0])

        # TODO(andrei): Index by MRP!
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