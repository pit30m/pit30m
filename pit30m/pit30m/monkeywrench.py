
import os
import ipdb
import csv
import geojson
from pickle import UnpicklingError
from datetime import datetime
import numpy as np
import lz4
import fire
import fsspec
from PIL import Image
import lzma
from lzma import LZMAError
from tqdm import tqdm
from typing import Optional, List, Tuple
from urllib.parse import urlparse, urljoin
from pit30m.camera import CamName
from pit30m.data.log_reader import LogReader
from pit30m.indexing import associate
from joblib import Parallel, delayed

EXPECTED_IMAGE_SIZE = (1200, 1920, 3)


class MonkeyWrench:

    def __init__(self,
        dataset_root: str = "s3://the/bucket",
        metadata_root: str = "file:///mnt/data/pit30m/",
    ) -> None:
        """Dataset administration tool for the Pit30M dataset (geo-indexing, data integrity checks, etc.)

        For regular end users, `pit30m.cli` is almost always what you should use.

        Users of the dataset won't have permissions to modify the official dataset, but they can use this tool to create
        custom indexes and subsets of the dataset.
        """
        self._root = dataset_root
        self._metadata_root = metadata_root


    def index(self, log_id: str, out_index_fpath: Optional[str] = None, check: bool = False):
        """Create index files for the raw data in the dataset.

        At dump time, the dataset just contained numbered image files with small associated metadata files. This meant
        it was basically impossible to find images by GPS location or timestamp. This tool creates indexes that allow
        for fast lookups by GPS location or timestamp.

        Building large indexes from scratch can take a while, even on the order of hours on some machines. We are
        dealing with roughly 400 million images, after all.

        Args:
            log_id:             The ID of the log to analyze and index.
            out_index_fpath:    Where to write indexes. If None (default) they will be written to the same directory as
                                the respective sensor.
            check:              If True, run expensive integrity checks on the data, e.g., actually read each image,
                                check its shape, load the metadata and ensure it's not corrupted, load the LiDAR and
                                check its shape too, etc.
        """
        self.index_all_cameras(log_id=log_id, out_index_fpath=out_index_fpath, check=check)
        self.index_lidar(log_id=log_id, out_index_fpath=out_index_fpath, check=check)

    def merge_indexes(self, log_ids: List[str]):
        """Given a list of log IDs with indexed data, merge the indexes into a single index."""
        # TODO(andrei): Implement this.
        ...

    def index_all_cameras(self, log_id: str, out_index_fpath: Optional[str] = None, check: bool = False):
        """Create an index of the images in the given log.

        This is useful for quickly finding images in a given region, or for finding the closest image to a given GPS
        location.

        Building large indexes from scratch can take a while, even on the order of hours on some machines.
        """
        for cam in CamName:
            self.index_camera(log_id=log_id, cam_name=cam, out_index_fpath=out_index_fpath, check=check)

    def index_lidar(
        self,
        log_id: str,
        lidar_name: str = "hdl64e_12_middle_front_roof",
        sweep_time_convention: str = "end",
        out_index_fpath: Optional[str] = None,
        check: bool = False
    ):
        """Same as 'index_all_cameras', except for the LiDAR sweeps."""
        if check:
            raise RuntimeError("XXX not implemented yet")
        in_fs = fsspec.filesystem(urlparse(self._root).scheme)
        out_fs = fsspec.filesystem(urlparse(out_index_fpath).scheme)
        log_root = os.path.join(self._root, log_id.lstrip("/"))
        log_reader = LogReader(log_root_uri=log_root)
        lidar_dir = os.path.join(log_root, "lidars", lidar_name.lstrip("/"))
        if out_index_fpath is None:
            out_index_fpath = os.path.join(lidar_dir, "index")

        wgs84_index_fpath = os.path.join(out_index_fpath, "raw_wgs84.csv")
        utm_index_fpath = os.path.join(out_index_fpath, "utm.csv")
        unindexed_fpath = os.path.join(out_index_fpath, "unindexed.csv")
        dumped_ts_fpath = os.path.join(lidar_dir, "timestamps.npz.lz4")
        # Non-sorted list of outputs used in reporting if check is True.
        status = []

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
        raw_wgs84 = log_reader.raw_wgs84_poses_dense

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

        wgs84_times = raw_wgs84[:, 0]
        wgs84_corr_idx = associate(sweep_times, wgs84_times)
        wgs84_delta = abs(raw_wgs84[wgs84_corr_idx, 0] - sweep_times)
        # Recall WGS84 messages are at 10Hz so we have to be a bit more lax than when checking pose assoc
        bad_offs = wgs84_delta > 0.10
        print(bad_offs.sum(), "bad offsets")
        # if bad_offs.sum() > 0:
        #     print(np.where(bad_offs))
        #     print()

        lidars_with_wgs84 = []
        assert len(sweep_times) == len(bad_offs) == len(wgs84_corr_idx)
        assert len(valid_sample_uris) == len(sweep_times)
        for sweep_uri, sweep_time, bad_off, wgs84_idx in zip(valid_sample_uris, sweep_times, wgs84_delta, wgs84_corr_idx):
            bad_off = wgs84_delta > 0.10
            if bad_off:
                status.append((sweep_uri, sweep_time, f"bad raw WGS84 offset, {wgs84_delta:.4f}s"))
                # TODO Should we flag these in the index?
                continue

            lidar_fpath = "/".join(sweep_uri.split("/")[-2:])

            # img row would include capture seconds, path, then other elements
            # imgs_with_wgs84.append((wgs84_data[wgs84_idx], img_row))
            lidars_with_wgs84.append((raw_wgs84[wgs84_idx, :], (sweep_time, lidar_fpath)))


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


    def index_camera(self, log_id: str, cam_name: CamName, out_index_fpath: Optional[str] = None,
                    check: bool = False):
        """Please see `index_all_cameras` for info."""
        in_fs = fsspec.filesystem(urlparse(self._root).scheme)

        log_root = os.path.join(self._root, log_id.lstrip("/"))
        log_reader = LogReader(log_root_uri=log_root)
        cam_dir = os.path.join(log_root, "cameras", cam_name.value.lstrip("/"))

        # Collects diagnostics for writing the health report, if check is True. The diagnostics will NOT be sorted and
        # may contain duplicates.
        status = []

        # if out_index_fpath is None:
        #     out_index_fpath = os.path.join(log_root, "index", f"{cam_name}.geojson")
        if out_index_fpath is None:
            out_index_fpath = os.path.join(cam_dir, "index")

        regular_index_fpath = os.path.join(out_index_fpath, "index.csv")
        raw_wgs84_index_fpath = os.path.join(out_index_fpath, "raw_wgs84.csv")
        utm_index_fpath = os.path.join(out_index_fpath, "utm.csv")
        out_fs = fsspec.filesystem(urlparse(out_index_fpath).scheme)
        unindexed_fpath = os.path.join(out_index_fpath, "unindexed.csv")
        report_fpath = os.path.join(out_index_fpath, "report.csv")

        # if not os.path.exists(os.path.join(log_root, "all_poses.npz.lz4")):
        #     continue

        # TODO(andrei): Check if these WGS84 values are raw or adjusted based on localization since it makes a big
        # impact on any localization or reconstruction work. For indexing we should be fine either way.
        #
        # NOTE(andrei): WGS84 data is coarser, 10Hz, not 100Hz.
        raw_wgs84_data = log_reader.raw_wgs84_poses_dense
        raw_wgs84_times = np.array(raw_wgs84_data[:, 0])

        # print(pose_times.shape)

        cp_dense = log_reader.continuous_pose_dense
        cp_times = cp_dense[:, 0]

        utm_poses = log_reader.utm_poses_dense
        mrp_poses = log_reader.map_relative_pose_dense
        # Check a dataset invariant as a smoke tests
        assert len(mrp_poses) == len(utm_poses)
        utm_times = utm_poses[:, 0]

        index = []
        for entry in tqdm(in_fs.glob(os.path.join(cam_dir, "*", "*.webp"))):
            img_fpath = entry
            meta_fpath = entry.replace(".day", ".meta").replace(".night", ".meta").replace(".webp", ".npy")
            # Keep track of the log ID in the index, so we can merge indexes easily.
            img_fpath_in_root = "/".join(img_fpath.split("/")[-5:])

            timestamp_s = -1
            with in_fs.open(meta_fpath) as meta_f:
                try:
                    # The tolist actually extracts a dict...
                    meta = np.load(meta_f, allow_pickle=True).tolist()
                    timestamp_s = meta["capture_seconds"]
                    index.append((timestamp_s, img_fpath_in_root, meta["shutter_seconds"],
                                meta["sequence_counter"], meta["gain_db"]))
                except UnpicklingError as err:
                    # TODO(andrei): Remove this hack once you re-extract with your ETL code
                    # hack for corrupted metadata, which should be fixed in the latest ETL
                    err_msg = f"WARNING: UnpicklingError reading metadata, {str(err)}"
                    status.append((meta_fpath, timestamp_s, err_msg))
                    print(meta_fpath, err_msg)
                    continue
                except ModuleNotFoundError:
                    # TODO(andrei): Remove this one too
                    # seems like corrupted pickles can trigger this, oof
                    err_msg = f"WARNING: ModuleNotFoundError reading metadata {str(err)}"
                    status.append((meta_fpath, timestamp_s, err_msg))
                    print(meta_fpath, err_msg)
                    continue

            if check:
                try:
                    img = Image.open(img_fpath)
                    img.verify()
                    # This will actually read the image data!
                    img_np = np.asarray(img)
                    if img_np.shape != EXPECTED_IMAGE_SIZE:
                        status.append((img_fpath, timestamp_s, f"WARNING: unexpected size {img_np.shape}"))
                    else:
                        status.append((img_fpath, timestamp_s, "OK"))
                except Exception as e:
                    err_msg = f"WARNING: General error reading image {str(e)}"
                    status.append((img_fpath, timestamp_s, err_msg))
                    print(err_msg)
                    continue

        # Sort by the capture time so we can easily search images by a timestamp
        image_index = sorted(index, key=lambda x: x[0])

        imgs_with_pose = []
        imgs_with_wgs84 = []
        unindexed_frames = []
        for img_row in tqdm(image_index):
            img_time = float(img_row[0])
            img_fpath = img_row[1]
            mrp_idx = np.searchsorted(utm_times, img_time, side="left")
            mrp_idx = mrp_idx + 1
            if mrp_idx >= len(utm_poses):
                mrp_idx = len(utm_poses) - 1
            delta_s = abs(img_time - mrp_poses["time"][mrp_idx])
            if delta_s > 0.1:
                # NOTE(andrei): This is just an indexing limitation, not a true error IMO.
                # print(f"WARNING: {img_time = } does not have a valid pose in this log [{delta_s = }]")
                error = f"WARNING: {img_time = } does not have a valid pose in this log [{delta_s = }]"
                status.append((img_fpath, img_time, error))
                unindexed_frames.append(img_row[1])
            else:
                imgs_with_pose.append((utm_poses[mrp_idx], mrp_poses[mrp_idx], img_row))

            wgs84_idx = np.searchsorted(raw_wgs84_times, img_time, side="left")
            wgs84_idx += 1
            if wgs84_idx >= len(raw_wgs84_data):
                wgs84_idx = len(raw_wgs84_data) - 1

            delta_wgs_s = abs(img_time - raw_wgs84_times[wgs84_idx])
            if delta_wgs_s > 0.5:
                print(f"WARNING: {img_time = } does not have a valid WGS84 coordinate in this log [{delta_s = }]")
            else:
                imgs_with_wgs84.append((raw_wgs84_data[wgs84_idx], img_row))


        # TODO(andrei): Don't write CP, because then you get nonsense when aggregating across logs.
        # with open(out_index_fpath, "w", newline="") as csvfile:
        #     spamwriter = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     spamwriter.writerow(["capture_time", "poses_and_differentials_valid", "x", "y", "z", "capture_seconds", "img_fpath_in_cam", "shutter_seconds", "sequence_counter", "gain_db"])
        #     for pose_row, img_row in imgs_with_pose:
        #         spamwriter.writerow(list(pose_row) + list(img_row))

        if not out_fs.exists(out_index_fpath):
            out_fs.mkdir(out_index_fpath)
        with out_fs.open(raw_wgs84_index_fpath, "w", newline="") as csvfile:
            spamwriter = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["timestamp", "longitude", "latitude", "altitude", "heading", "pitch", "roll", "capture_seconds", "img_fpath_in_cam", "shutter_seconds", "sequence_counter", "gain_db"])
            for wgs84_row, img_row in imgs_with_wgs84:
                spamwriter.writerow(list(wgs84_row) + list(img_row))

        # Write a text file with all samples which could not be matched to an accurate pose.
        # NOTE(andrei): Doing the same for WGS84 is a nice-to-have, but not a priority.
        with out_fs.open(unindexed_fpath, "w", newline="") as csvfile:
            spamwriter = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["img_fpath_in_cam"])
            for entry in unindexed_frames:
                spamwriter.writerow([entry])

        with out_fs.open(utm_index_fpath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["timestamp", "utm_e", "utm_n", "mrp_x", "mrp_y", "mrp_z", "mrp_roll", "mrp_pitch",
                "mrp_yaw", "mrp_submap_id", "capture_seconds", "img_fpath_in_cam",
                "shutter_seconds", "sequence_counter", "gain_db"])
            for utm_pose, mrp_pose, img_row in imgs_with_pose:
                utm_e, utm_n = utm_pose
                # timestamp, valid, submap, x,y,z,roll,pitch,yaw = mrp_pose
                # TODO(andrei): Add altitude here.
                writer.writerow([mrp_pose["time"], utm_e, utm_n, -1] +
                [mrp_pose["x"], mrp_pose["y"], mrp_pose["z"],
                mrp_pose["roll"], mrp_pose["pitch"], mrp_pose["yaw"], mrp_pose["submap_id"]] + list(img_row))

        if check:
            # TODO(andrei): Write actual report here.
            # NOTE: The report may have duplicates, since an image may be missing a pose _AND_ be corrupted. The outputs
            # are not sorted by time or anything.
            with out_fs.open(report_fpath, "w") as csvfile:
                writer = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["img_fpath_in_cam", "timestamp", "status"])
                for path, timestamp, message in status:
                    writer.writerow([path, timestamp, message])

        report = ""
        report += "Date: " + datetime.isoformat(datetime.now()) + "\n"
        report += f"(log_root = {log_root})\n"
        report += f"{len(status)} problems found:\n"
        for problem in status:
            report += "\t -" + problem + "\n"
        report += ""

        # TODO(andrei): Should we write a receipt if there were no problems?
        print(report)

        return report

    def aggregate_reports(self, dataset_base: str, log_list_fpath: Optional[str] = None):
        # TODO(andrei): Given a list of logs (or None = all of them), look for reports, read, and aggregate.
        # for instance, for the first batch I could ETL 100 logs and iterate with a txt with their IDs. Once I'm happy,
        # I can proceed with all other logs.
        #
        # If any log does NOT have its receipt, throw an error.
        #
        # In other words, 'index' (or diagnose, still TBD) is the map, possibly running in parallel, and
        # 'aggregate_reports' is the reduce.
        pass


    def diagnose(self, dataset_base: str):
        # TODO(andrei): Check that the dataset is valid---all logs.
        # TODO(andrei): Unify with the indexing functions.
        pass

    def diagnose_log(self, dataset_base: str, log_id: str) -> None:
        self._diagnose_lidar(dataset_base, log_id)
        self._diagnose_misc(dataset_base, log_id)

    def diagnose_log_cameras(self, dataset_base: str, log_id: str) -> None:
        for cam in CamName:
            self.diagnose_log_camera(dataset_base, log_id, cam)

    def diagnose_log_camera(self, dataset_base: str, log_uri: str) -> None:
        fs = fsspec.filesystem(urlparse(dataset_base).scheme)
        log_root_uri = os.path.join(dataset_base, log_uri)
        log_reader = LogReader(log_root_uri)



    def _diagnose_misc(self, dataset_base: str, log_uri: str) -> None:
        """Loads and prints misc data, hopefully raising errors if something is corrupted."""
        dbp = urlparse(dataset_base)
        fs = fsspec.filesystem(dbp.scheme)

        # Active district information is not very important
        with lzma.open(dataset_base + log_uri + "/active_district.npz.xz", "rb") as f:
            ad = np.load(f)["data"]
            assert isinstance(ad, np.ndarray)
            print("Active district OK:", ad.shape, ad.dtype)

        # TODO(andrei): We probably want to provide poses as something uncompressed, since LZMA decoding is a few
        # seconds per log. For training we will cache this in some index files, but in general it's annoying to wait
        # 4-5 seconds on a modern PC to read 50MiB of data...
        with lzma.open(dataset_base + log_uri + "/all_poses.npz.xz", "rb") as f:
            # Poses contain just the data as a structured numpy array. Each pose object contains a continuous, smooth,
            # log-specific pose, and a map-relative pose. The map relative pose (MRP) takes vehicle points into the
            # current submap, whose ID is indicated by the 'submap' field of the MRP.
            #
            # The data also has pose and velocity covariance information, but I have never used directly so I don't know
            # if it's well-calibrated.
            #
            # Be sure to check the 'valid' flag of the poses before using them!
            #
            # Poses are provided at 100Hz.
            all_poses = np.load(f)["data"]
            print(len(all_poses), "poses read")
            print("All poses read OK")

        # TODO(andrei): Support parsing this pseudo YAML. Right now it's not very important but may be useful if people
        # want raw wheel encoder data, steering angle, etc.
        # with lzma.open(dataset_base + log_uri + "/all_vehicle_data.npz.xz", "rb") as f:
        #     all_vd = np.load(f)
        #     print(list(all_vd.keys()))
        #     ipdb.set_trace()
        #     print("All vd OK")

        # NOTE(andrei): The metadata won't be super useful to end users, since it's mostly internal stuff like a list
        # of active sensors, calibration, etc., which is already obvious by looking at the data files.
        with open(dataset_base + log_uri + "/log_metadata.json", "r") as f:
            meta = json.load(f)
            print("Metadata OK")


        with fs.open(dataset_base + log_uri + "/wgs84.npz.xz", "rb") as lzma_f:
            with lzma.open(lzma_f) as f:
                # WGS84 poses with timestamp, long, lat, alt, heading, pitch, roll.
                #
                # ~10Hz
                # TODO(andrei): Check whether these are inferred from MRP or not. I *think* for logs where localization is
                # OK (i.e., most logs -- and we can always check pose validity flags!) these are properly post-processed,
                # and not just biased filtered GPS, but I need to check. Plotting this with leaflet will make it obvious.
                all_wgs84 = np.load(f)["data"]
                print(f"{len(all_wgs84) = }")
                print("All WGS84 poses OK")



    def _diagnose_lidar(self, dataset_base: str, log_uri: str) -> None:
        ld = dataset_base + log_uri + "/lidars/"
        dbp = urlparse(dataset_base)
        fs = fsspec.filesystem(dbp.scheme)

        # TODO(andrei): Diagnose and check shape!

        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            for lid_dir in fs.ls(ld):
                all_pcd = []
                all_int = []
                n_chunks = 20000
                # Sample every 'sample_rate' sweeps.
                sample_rate = 5
                lidar_chunks = sorted(fs.ls(lid_dir))
                print(f"{len(lidar_chunks)} subdirectories of LiDAR found")
                lidar_files = []
                for chunk in tqdm(lidar_chunks[:n_chunks]):
                    if fs.isdir(chunk):
                        lidar_files += sorted(fs.ls(chunk)[::sample_rate])

                print(f"Will load {len(lidar_files)} LiDAR pcds.")
                all_pcd, all_int = unzip(pool.map(_load_lidar, lidar_files))
                print(all_pcd[0].shape)
                print(all_pcd[0].dtype)
                # for sub_sample in tqdm(
                #     with lzma.open(sub_sample) as lzma_file:
                #         lidar = np.load(lzma_file)
                #         # Points are in the continuous frame I think
                #         all_pcd.append(np.array(lidar["points"]))
                #         all_int.append(np.array(lidar["intensity"]))
                #         # other keys: laser_theta, seconds, raw_power, intensity, points, points_H_sensor, laser_id

                print("Loaded points, processing...")
                all_pcd = np.concatenate(all_pcd, axis=0)
                all_int = np.concatenate(all_int, axis=0)
                # TODO(andrei): This likely makes a copy (based on how slow it is), so we may want to use the tensor
                # API?
                vec = o3d.utility.Vector3dVector(all_pcd)
                col = o3d.utility.Vector3dVector(np.tile(all_int[:, None] / 255.0, (1, 3)))
                pcd = o3d.geometry.PointCloud(vec)
                pcd.colors = col
                print("Before filtering:", pcd)
                pcd = pcd.voxel_down_sample(voxel_size=0.05)
                print("After filtering:", pcd)
                o3d.visualization.draw_geometries([pcd])


def _load_lidar(lidar_uri: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a single LiDAR sweep from a URI, reading a LZMA-compressed file."""
    fs = fsspec.filesystem(urlparse(lidar_uri).scheme)
    with fs.open(lidar_uri) as f:
        try:
            with lzma.open(f) as lzma_file:
                lidar = np.load(lzma_file)
                return np.array(lidar["points"]), np.array(lidar["intensity"])
        except LZMAError:
            print("LZMA error reading LiDAR uri: ", lidar_uri)
            raise


if __name__ == "__main__":
    fire.Fire(MonkeyWrench)