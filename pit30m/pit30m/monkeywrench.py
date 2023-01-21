import os
import ipdb
import csv
import json
from pickle import UnpicklingError
from dataclasses import dataclass
import multiprocessing as mp
import logging
from functools import cached_property
from datetime import datetime
from enum import Enum
import numpy as np
import lz4
import fire
import fsspec
import yaml
from PIL import Image
import lzma
from lzma import LZMAError
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Optional, List, Tuple, Union
from urllib.parse import urlparse, urljoin

from pit30m.camera import CamName
from pit30m.data.log_reader import LogReader
from pit30m.data.submap import Map
from pit30m.indexing import associate

EXPECTED_IMAGE_SIZE = (1200, 1920, 3)

CAM_META_MISSING = "cam_meta_missing"
CAM_META_UNPICKLING_ERROR = "cam_meta_unpickling_error"
CAM_UNEXPECTED_SHAPE = "cam_unexpected_shape"
CAM_UNEXPECTED_CORRUPTION = "cam_unexpected_corruption"
# Expected to happen a lot, should be a warning unless the whole log is wrong
CAM_UNMATCHED_POSE = "cam_unmatched_pose"
CAM_UNMATCHED_RAW_WGS84 = "cam_unmatched_raw_wgs84"
CAM_MEAN_TOO_LOW = "cam_mean_too_low"
CAM_MEAN_TOO_HIGH = "cam_mean_too_high"
INVALID_REPORT = "invalid_report"

VALIDATION_CANARY_FNAME = ".pit30m_valid_v0"
ETL_CANARY_FNAME = ".pit30m_done"
PIT30M_IMAGE_CANARY_DONE = ".pit30m_image_done"
PIT30M_NONIMAGE_CANARY_DONE = ".pit30m_nonimage_done"

MIN_LIDAR_FRAMES = 10 * 60 * 3

TQDM_MIN_INTERVAL_S = 3

KNOWN_INCOMPLETE_CAMERAS = [
    # Confirmed 2022-11-24 - only has hdcam_12_middle_front_narrow_left
    "c9e9e7a7-f1cb-4af8-c5c9-3a610cbcc20e",
    # Confirmed 2022-11-24 - only has hdcam_02_starboard_front_roof_wide
    "dc3d9c11-5ec7-41cd-d4e0-ec795cdae27d",
]


@dataclass
class CamReportSummary:
    n_ok: int
    warnings: List[str] = None
    errors: List[str] = None

    @property
    def n_warns(self) -> int:
        return len(self.warnings)

    @property
    def n_errs(self) -> int:
        return len(self.errors)



class LogStatus(Enum):
    NOT_ATTEMPTED = "not_attempted"
    DONE_NOT_INDEXED = "done_not_indexed"
    CRASHED_OR_IN_PROGRESS = "crashed_or_in_progress"
    VALIDATED = "validated"

    NEEDS_FINAL_RECEIPT = "needs_final_receipt"
    ONLY_GPU_DONE = "only_gpu_done"
    ONLY_CPU_DONE = "only_cpu_done"


def query_log_status(root, log_id) -> LogStatus:
    out_fs = fsspec.filesystem(urlparse(root).scheme)
    out_log_root = os.path.join(root, log_id.lstrip("/"))
    if not out_fs.exists(out_log_root):
        return LogStatus.NOT_ATTEMPTED
    elif out_fs.exists(os.path.join(out_log_root, VALIDATION_CANARY_FNAME)):
        # Log validated successfully. We gucci.
        return LogStatus.VALIDATED
    elif out_fs.exists(os.path.join(out_log_root, ETL_CANARY_FNAME)):
        # Log was dumped but not yet indexed
        return LogStatus.DONE_NOT_INDEXED
    else:
        img_ok = False
        non_img_ok = False
        if out_fs.exists(os.path.join(out_log_root, PIT30M_IMAGE_CANARY_DONE)):
            img_ok = True
        if out_fs.exists(os.path.join(out_log_root, PIT30M_NONIMAGE_CANARY_DONE)):
            non_img_ok = True

        if img_ok and non_img_ok:
            # We dumped both big parts of the log, but not the final receipt
            return LogStatus.NEEDS_FINAL_RECEIPT

        if img_ok:
            return LogStatus.ONLY_GPU_DONE
        elif non_img_ok:
            return LogStatus.ONLY_CPU_DONE
        else:
            # File exists but the dump did not finish (yet?)
            return LogStatus.CRASHED_OR_IN_PROGRESS


def qls_it(root, log_ids, batch_size, max):
    pool = Parallel(n_jobs=-1)

    for i in range(0, len(log_ids), batch_size):
        res = pool(
            delayed(query_log_status)(root, log_id)
            for log_id in log_ids[i : i + batch_size]
        )
        # TODO(andrei): Invoke next batch while yielding current.
        for element in res:
            yield res

        if i + batch_size > max:
            break


class MonkeyWrench:
    def __init__(
        self,
        dataset_root: str = "s3://pit30m/",
        metadata_root: str = "file:///mnt/data/pit30m/",
        log_status_fpath: str = "log_status.csv",
        log_list_fpath: str = "all_logs.txt",
        submap_utm_fpath: str = "submap_utm.pkl",
    ) -> None:
        """Dataset administration tool for the Pit30M dataset (geo-indexing, data integrity checks, etc.)

        For regular end users, `pit30m.cli` is almost always what you should use.

        Users of the dataset won't have permissions to modify the official dataset, but they can use this tool to create
        custom indexes and subsets of the dataset.
        """
        self._root = dataset_root
        self._metadata_root = metadata_root
        self._log_status_fpath = log_status_fpath
        self._log_list_fpath = log_list_fpath
        self._submap_utm_fpath = submap_utm_fpath

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self._logger.addHandler(handler)

    @cached_property
    def all_logs(self) -> List[str]:
        """Return a list of all log IDs."""
        with open(self._log_list_fpath, "r") as f:
            return [line.strip() for line in f]

    @cached_property
    def map(self) -> Map:
        return Map.from_submap_utm_uri(self._submap_utm_fpath)

    def index(self, log_id: str, out_index_fpath: Optional[str] = None, check: bool = True):
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
        res = self.diagnose_misc(log_id=log_id, out_index_fpath=out_index_fpath, check=check)
        # XXX(andrei): Turn the misc diagnosis into a mini report too.
        if len(res) > 0:
            print("WARNING: misc non-sensor data had errors. See above for more information.")

        print("Log indexing complete.")

    def merge_indexes(self, log_ids: List[str]):
        """Given a list of log IDs with indexed data, merge the indexes into a single index."""
        # TODO(andrei): Implement this.
        ...

    def get_lidar_dir(self, log_root: str, lidar_name: str) -> str:
        return os.path.join(log_root, "lidars", lidar_name.lstrip("/"))

    def get_cam_dir(self, log_root: str, cam_name: str) -> str:
        return os.path.join(log_root, "cameras", cam_name.lstrip("/"))

    def next_to_validate(self, max: int = 10):
        """Next logs to validate."""
        pass

    def next_to_etl(self, max: int = 10, include_attempted_but_incomplete: bool = False, kind: str = "gpu",
            batch_size: int = 32, n_read_workers: int = 12, webp_out_quality: int = 85):
        all_logs = self.all_logs
        not_attempted = []
        crashed_or_in_progress = []

        need_gpu_job = []
        need_cpu_job = []
        need_only_receipt = []
        effective_list = []

        for log_id in all_logs:
            status = query_log_status(self._root, log_id)
            if status == LogStatus.NOT_ATTEMPTED:
                # not_attempted.append(log_id)
                need_gpu_job.append(log_id)
            elif status == LogStatus.ONLY_CPU_DONE:
                need_gpu_job.append(log_id)
            elif status == LogStatus.ONLY_GPU_DONE:
                need_cpu_job.append(log_id)
            elif status == LogStatus.NEEDS_FINAL_RECEIPT:
                need_only_receipt.append(log_id)
            elif status == LogStatus.CRASHED_OR_IN_PROGRESS and include_attempted_but_incomplete:
                crashed_or_in_progress.append(log_id)

            effective_list = need_gpu_job if kind == "gpu" else need_cpu_job
            if include_attempted_but_incomplete:
                effective_list += crashed_or_in_progress
            # effective_list = crashed_or_in_progress

            if max > 0 and len(effective_list) >= max:
                break

        # print("Not attempted:")
        # if include_attempted_but_incomplete:
        #     print("(or attempted but incomplete logs)")
        for log_id in effective_list:
            if kind == "gpu":
                print(f"python process.py etl_images {log_id} --anonymizer yolo --anonymizer-weights "
                    f"/app/weights/anon-yolo5m-exp21-best.fp16.bs{batch_size}.1280.engine --image-batch-size {batch_size} "
                    f"--meta-batch-size {batch_size * 4} --out-root s3://pit30m/ --n-read-workers {n_read_workers} "
                    f"--webp-out-quality {webp_out_quality}")
            elif kind == "cpu":
                print(f"python process.py etl_non_images {log_id} --out-root s3://pit30m/ " \
                    f"--n-read-workers {int(n_read_workers)}")

    def index_lidar(
        self,
        log_id: str,
        lidar_name: str = "hdl64e_12_middle_front_roof",
        sweep_time_convention: str = "end",
        out_index_fpath: Optional[str] = None,
        check: bool = True,
    ):
        """Same as 'index_all_cameras', except for the LiDAR sweeps."""
        in_fs = fsspec.filesystem(urlparse(self._root).scheme)
        out_fs = fsspec.filesystem(urlparse(out_index_fpath).scheme)
        log_root = os.path.join(self._root, log_id.lstrip("/"))

        if not os.path.isdir(log_root):
            raise RuntimeError(f"Log {log_id} directory does not exist at all. Indexing failed.")

        log_reader = LogReader(log_root_uri=log_root)
        lidar_dir = self.get_lidar_dir(log_root, lidar_name)
        etl_canary = os.path.join(lidar_dir, ETL_CANARY_FNAME)

        if out_index_fpath is None:
            out_index_fpath = os.path.join(lidar_dir, "index")

        wgs84_index_fpath = os.path.join(out_index_fpath, "raw_wgs84.csv")
        utm_index_fpath = os.path.join(out_index_fpath, "utm.csv")
        unindexed_fpath = os.path.join(out_index_fpath, "unindexed.csv")
        dumped_ts_fpath = os.path.join(lidar_dir, "timestamps.npz.lz4")
        report_fpath = os.path.join(out_index_fpath, "report.csv")

        # Non-sorted list of outputs used in reporting if check is True.
        stats = []

        if not os.path.isfile(etl_canary):
            raise RuntimeError(
                f"Log {log_id} was not dumped yet as its 'ETL finished' canary file was not found; " "can't index it."
            )

        def _get_lidar_time(lidar_uri):
            try:
                with in_fs.open(lidar_uri, "rb") as compressed_f:
                    with lz4.frame.open(compressed_f, "rb") as f:
                        lidar_data = np.load(f)
                        point_times = lidar_data["seconds"]

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

                        return (
                            "OK",
                            lidar_uri,
                            point_times.min(),
                            point_times.max(),
                            point_times.mean(),
                            np.median(point_times),
                            lidar_data["points"].shape,
                        )
            except EOFError as err:
                return "Error", lidar_uri, "EOFError", str(err)
            except Exception as err:
                return "Error", lidar_uri, "unknown-error", str(err)

        # TODO(andrei): Directly using the timestamps file seems difficult to leverage as the number of timestamps seems
        # to differ from the number of dumped sweeps, so aligning the two would be challenging. Perhaps I could just use
        # this data to check some of my assumptions later.
        #
        # with in_fs.open(dumped_ts_fpath, "rb") as compressed_f:
        #     with lz4.frame.open(compressed_f, "rb") as f:
        #         timestamps = np.load(f)["data"]

        sample_uris = in_fs.glob(os.path.join(lidar_dir, "*", "*.npz.lz4"))
        print(f"Will analyze and index {len(sample_uris)} samples")
        pool = Parallel(n_jobs=-1, verbose=10)
        time_stats = pool(delayed(_get_lidar_time)(lidar_uri) for lidar_uri in sample_uris)
        raw_wgs84 = log_reader.raw_wgs84_poses_dense

        sweep_times_raw = []
        valid_sample_uris = []
        for sample_uri, result in zip(sample_uris, time_stats):
            status = result[0]
            if status != "OK":
                err_msg = f"error_{str(result[2:])}"
                print(err_msg, sample_uri)
                stats.append(tuple([sample_uri, -1] + list(result[2:])))
                continue

            (min_s, max_s, mean_s, med_s, shape) = result[2:]
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

            stats.append((sample_uri, sweep_times_raw[-1], "OK", "n/A"))

        sweep_times = np.array(sweep_times_raw)
        # del sample_uri

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
        for sweep_uri, sweep_time, wgs84_delta_sample, wgs84_idx in tqdm(
            zip(valid_sample_uris, sweep_times, wgs84_delta, wgs84_corr_idx),
            mininterval=TQDM_MIN_INTERVAL_S,
        ):
            bad_off = wgs84_delta_sample > 0.10
            if bad_off:
                stats.append((sweep_uri, sweep_time, "bad-raw-WGS84-offset", f"{wgs84_delta_sample:.4f}s"))
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
            spamwriter = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                [
                    "timestamp",
                    "longitude",
                    "latitude",
                    "altitude",
                    "heading",
                    "pitch",
                    "roll",
                    f"sweep_seconds_{sweep_time_convention}",
                    "lidar_fpath",
                ]
            )
            for wgs84_row, lidar_row in lidars_with_wgs84:
                spamwriter.writerow(list(wgs84_row) + list(lidar_row))

        if check:
            # NOTE: The report may have duplicates, since an image may be missing a pose _AND_ be corrupted. The outputs
            # are not sorted by time or anything.
            with out_fs.open(report_fpath, "w") as csvfile:
                writer = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["lidar_uri", "timestamp", "status", "details"])
                for path, timestamp, message, details in stats:
                    writer.writerow([path, timestamp, message, details])

        report = ""
        report += "Date: " + datetime.isoformat(datetime.now()) + "\n"
        report += f"(log_root = {log_root})\n"
        report += f"{len(stats)} samples analyzed.\n"
        n_problems = len([x for x in stats if x[2] != "OK"])
        report += f"{n_problems} problems found."
        # for problem in status:
        #     report += "\t -" + problem + "\n"
        report += ""
        print(report)

        if check:
            print("\n\nWrote detailed health report to", report_fpath)
        else:
            print("Did not compute or dump detailed report.")

    def index_all_cameras(
        self, log_id: str, out_index_fpath: Optional[str] = None, check: bool = True, parallel: bool = True
    ):
        """Create an index of the images in the given log.

        This is useful for quickly finding images in a given region, or for finding the closest image to a given GPS
        location.

        Building large indexes from scratch can take a while, even on the order of hours on some machines.

        Args:
            log_id:     ID of the log to process within the established root.
            out_index_fpath:   Path to the output index file. If not given, the index will be written relative to
                                the respective camera roots (highly recommended).
            check:      If True, will check the integrity of the log in great detail, including image properties and
                        shapes, and write a detailed report to the output index
            parallel:   Whether to process each camera in parallel.
        """
        if parallel:
            n_jobs = len(CamName)
            print(f"Using exactly {n_jobs} jobs")
            with mp.Pool(processes=n_jobs) as pool:
                pool.starmap(
                    self.index_camera,
                    [(log_id, cam_name, out_index_fpath, check, index) for index, cam_name in enumerate(CamName)],
                )
        else:
            for cam in CamName:
                self.index_camera(log_id=log_id, cam_name=cam, out_index_fpath=out_index_fpath, check=check)

    def index_camera(
        self,
        log_id: str,
        cam_name: Union[CamName, str],
        out_index_fpath: Optional[str] = None,
        check: bool = False,
        pb_position: int = 0,
    ):
        """Please see `index_all_cameras` for info."""
        scheme = urlparse(self._root).scheme
        in_fs = fsspec.filesystem(scheme)
        if isinstance(cam_name, str):
            cam_name = CamName(cam_name)

        self._logger.info("Setting up log reader to process camera %s", cam_name.value)
        log_root = os.path.join(self._root, log_id.lstrip("/"))
        log_reader = LogReader(log_root_uri=log_root, map=self.map)
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

        # print(pose_times.shape)

        cp_dense = log_reader.continuous_pose_dense
        cp_times = cp_dense[:, 0]

        utm_poses = log_reader.utm_poses_dense
        mrp_poses = log_reader.map_relative_poses_dense
        # Check a dataset invariant as a smoke tests
        assert len(mrp_poses) == len(utm_poses)
        mrp_times = mrp_poses["time"]

        check_status = "deep checking" if check else "NO checking !!!"
        self._logger.info("Starting to index camera data. [%s]", check_status)
        index = []
        n_errors_so_far = 0
        debug_max_errors = 10

        progress_bar = tqdm(
            sorted(in_fs.glob(os.path.join(cam_dir, "*", "*.webp"))),
            mininterval=TQDM_MIN_INTERVAL_S,
            # Supposed to let each process have its own progress bar, but it doesn't work since the progress bars don't
            # start at the same time. Rip.
            position=pb_position,
            desc=f"{cam_name.value:<18}",
        )
        for entry in progress_bar:
            img_fpath = entry
            # XXX(andrei): Similarly, in a sibling method, look for large discrepancies between the number of samples
            # in different sensors. E.g., if one camera has 100 images and the other has 1000, that's a problem we would
            # like to look into.

            # XXX(andrei): Log a status error if meta is missing. May want to also quickly loop through metas and error
            # if no image for a specific meta.
            meta_fpath = entry.replace(".day", ".meta").replace(".night", ".meta").replace(".webp", ".npy")
            # Keep track of the log ID in the index, so we can merge indexes easily.
            img_fpath_in_root = "/".join(img_fpath.split("/")[-5:])
            progress_bar.set_postfix(n_errors_so_far=n_errors_so_far)
            if n_errors_so_far > debug_max_errors and debug_max_errors > 0:
                break

            if not in_fs.exists(meta_fpath):
                status.append((meta_fpath, timestamp_s, CAM_META_MISSING, f"Base entry uri: {entry}"))
                n_errors_so_far += 1
                continue

            timestamp_s = -1.0
            with in_fs.open(meta_fpath) as meta_f:
                try:
                    # The tolist actually extracts a dict...
                    meta = np.load(meta_f, allow_pickle=True).tolist()
                    timestamp_s = float(meta["capture_seconds"])
                    index.append(
                        (
                            timestamp_s,
                            img_fpath_in_root,
                            meta["shutter_seconds"],
                            meta["sequence_counter"],
                            meta["gain_db"],
                        )
                    )
                except UnpicklingError as err:
                    # TODO(andrei): Remove this hack once you re-extract with your ETL code
                    # hack for corrupted metadata, which should be fixed in the latest ETL
                    status.append((meta_fpath, timestamp_s, CAM_META_UNPICKLING_ERROR, str(err)))
                    n_errors_so_far += 1
                    continue
                except ModuleNotFoundError as err:
                    status.append((meta_fpath, timestamp_s, CAM_META_UNPICKLING_ERROR, str(err)))
                    n_errors_so_far += 1
                    continue
                    # TODO(andrei): Remove this one too
                    # seems like corrupted pickles can trigger this, oof
                    # err_msg = f"ERROR: ModuleNotFoundError reading metadata {str(err)}"
                    # status.append((meta_fpath, timestamp_s, err_msg))
                    # print(meta_fpath, err_msg)
                    # continue

            if check:
                try:
                    img_full_fpath = f"{scheme}://" + img_fpath
                    with in_fs.open(img_full_fpath, "rb") as img_f:
                        img = Image.open(img_f)
                        img.verify()
                        # This will actually read the image data!
                        img_np = np.asarray(img)
                        if img_np.shape != EXPECTED_IMAGE_SIZE:
                            status.append((img_full_fpath, timestamp_s, CAM_UNEXPECTED_SHAPE, str(img_np.shape)))
                            n_errors_so_far += 1
                            continue
                        else:
                            # Might indicate bad cases of over/underexposure. Likely won't trigger if the sensor is covered
                            # by snow (mean is larger than 5-10), which is fine since it's valid data.
                            img_mean = img_np.mean()
                            if img_mean < 5:
                                status.append((img_full_fpath, timestamp_s, CAM_MEAN_TOO_LOW, str(img_mean)))
                            elif img_mean > 250:
                                status.append((img_full_fpath, timestamp_s, CAM_MEAN_TOO_HIGH, str(img_mean)))
                            else:
                                status.append((img_full_fpath, timestamp_s, "OK", ""))
                except Exception as err:
                    status.append((img_full_fpath, timestamp_s, CAM_UNEXPECTED_CORRUPTION, str(err)))
                    n_errors_so_far += 1
                    continue

        # Sort by the capture time so we can easily search images by a timestamp
        image_index = sorted(index, key=lambda x: x[0])
        image_times = np.array([entry[0] for entry in image_index])

        # NOTE(andrei): WGS84 data is coarser, 10Hz, not 100Hz.
        self._logger.info("Reading WGS84 data")
        raw_wgs84_data = log_reader.raw_wgs84_poses_dense
        raw_wgs84_times = np.array(raw_wgs84_data[:, 0])

        imgs_with_pose = []
        imgs_with_wgs84 = []
        unindexed_frames = []
        # The 'max_delta_s' is just a logging thing. We will carefully inspect the deltas in the report analysis part,
        # but for now we only want WARNINGs to be printed if there's some serious problems. A couple of minutes of
        # missing poses (WGS84 or MRP-derived UTM) at the start of a log is expected (especially the localizer-based
        # MRP), and not a huge issue.
        utm_and_mrp_index = associate(image_times, mrp_times, max_delta_s=60 * 10)
        matched_timestamps = mrp_times[utm_and_mrp_index]
        deltas = np.abs(matched_timestamps - image_times)
        # invalid_times = deltas > 0.1
        for img_row, pose_idx, delta_s in zip(image_index, utm_and_mrp_index, deltas):
            img_fpath = img_row[1]
            img_time = float(img_row[0])
            if delta_s > 0.1:
                # error = f"WARNING: {img_time = } does not have a valid pose in this log [{delta_s = }]"
                status.append((img_fpath, img_time, CAM_UNMATCHED_POSE, str(delta_s)))
                unindexed_frames.append(img_row)
            else:
                imgs_with_pose.append((utm_poses[pose_idx, :], mrp_poses[pose_idx], img_row))

        raw_wgs84_index = associate(image_times, raw_wgs84_times, max_delta_s=60 * 10)
        matched_raw_wgs84_timestamps = raw_wgs84_times[raw_wgs84_index]
        raw_wgs84_deltas = np.abs(matched_raw_wgs84_timestamps - image_times)
        for img_row, wgs_idx, delta_wgs_s in zip(image_index, raw_wgs84_index, raw_wgs84_deltas):
            img_fpath = img_row[1]
            img_time = float(img_row[0])
            if delta_wgs_s > 0.5:
                # error = f"WARNING: {img_time = } does not have a valid raw WGS84 pose in this log [{delta_wgs_s = }]"
                status.append((img_fpath, img_time, CAM_UNMATCHED_RAW_WGS84, str(delta_wgs_s)))
            else:
                imgs_with_wgs84.append((raw_wgs84_data[wgs_idx, :], img_row))

        # TODO(andrei): Should we write CP? If we do we need clear docs, because using it will get you nonsense when
        # aggregating across logs.
        # with open(out_index_fpath, "w", newline="") as csvfile:
        #     spamwriter = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     spamwriter.writerow(["capture_time", "poses_and_differentials_valid", "x", "y", "z", "capture_seconds", "img_fpath_in_cam", "shutter_seconds", "sequence_counter", "gain_db"])
        #     for pose_row, img_row in imgs_with_pose:
        #         spamwriter.writerow(list(pose_row) + list(img_row))

        if not out_fs.exists(out_index_fpath):
            out_fs.mkdir(out_index_fpath)
        with out_fs.open(raw_wgs84_index_fpath, "w", newline="") as csvfile:
            spamwriter = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(
                [
                    "timestamp",
                    "longitude",
                    "latitude",
                    "altitude",
                    "roll",
                    "pitch",
                    "yaw",
                    "capture_seconds",
                    "img_fpath_in_cam",
                    "shutter_seconds",
                    "sequence_counter",
                    "gain_db",
                ]
            )
            for wgs84_row, img_row in imgs_with_wgs84:
                spamwriter.writerow(list(wgs84_row) + list(img_row))

        # Write a text file with all samples which could not be matched to an accurate pose.
        with out_fs.open(unindexed_fpath, "w", newline="") as csvfile:
            spamwriter = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["img_fpath_in_cam"])
            for entry in unindexed_frames:
                spamwriter.writerow([entry])

        with out_fs.open(utm_index_fpath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                [
                    "pose_timestamp",
                    "utm_e",
                    "utm_n",
                    "utm_alt",
                    "mrp_x",
                    "mrp_y",
                    "mrp_z",
                    "mrp_roll",
                    "mrp_pitch",
                    "mrp_yaw",
                    "mrp_submap_id",
                    "capture_seconds",
                    "img_fpath_in_cam",
                    "shutter_seconds",
                    "sequence_counter",
                    "gain_db",
                ]
            )
            for utm_pose, mrp_pose, img_row in imgs_with_pose:
                utm_e, utm_n = utm_pose
                # timestamp, valid, submap, x,y,z,roll,pitch,yaw = mrp_pose
                # TODO(andrei): Add UTM altitude here.
                writer.writerow(
                    [mrp_pose["time"], utm_e, utm_n, -1]
                    + [
                        mrp_pose["x"],
                        mrp_pose["y"],
                        mrp_pose["z"],
                        mrp_pose["roll"],
                        mrp_pose["pitch"],
                        mrp_pose["yaw"],
                        mrp_pose["submap_id"],
                    ]
                    + list(img_row)
                )

        if check:
            # NOTE: The report may have duplicates, since an image may be missing a pose _AND_ be corrupted. The outputs
            # are not sorted by time or anything.
            with out_fs.open(report_fpath, "w") as csvfile:
                writer = csv.writer(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["img_fpath_in_cam", "timestamp", "status", "details"])
                for idx, (path, timestamp, message, details) in enumerate(status):
                    writer.writerow([path, timestamp, message, details])
                    if idx < 5:
                        print(f"{path} {timestamp} {message} {details}")

        report = ""
        report += "Date: " + datetime.isoformat(datetime.now()) + "\n"
        report += f"(log_root = {log_root})\n"
        report += f"{len(status)} potential problems found ({n_errors_so_far} errors):\n"
        # for problem in status:
        #     report += "\t -" + problem + "\n"
        report += ""
        print(report)

        if check:
            print("\n\nWrote detailed health report to", report_fpath)
        else:
            print("Did not compute or dump detailed report.")

        return report

    def validate_reports(self, logs: Optional[str] = None, check_receipt: bool = True, write_receipt: bool = True):
        """
        Note that this will typically be the command you run locally, as reading a few CSVs for each log is more than
        doable locally.

        Args:
            logs:       Either a file path to a text file with a list of log IDs to validate, or a comma-separated list
                        of log IDs to validate.
            check_receipt:  If True, will check if the log has already been validated and skip it if so.
            write_receipt:  If validation runs and produces no errors, write a receipt.
        """
        if os.path.isfile(logs):
            log_list = []
            with open(logs, "r", encoding="utf-8") as f:
                for line in f:
                    log_list.append(line.strip())
        else:
            log_list = [entry.strip() for entry in logs.split(",")]

        # TODO(andrei): Given a list of logs (or None = all of them), look for reports, read, and aggregate.
        # for instance, for the first batch I could ETL 100 logs and iterate with a txt with their IDs. Once I'm happy,
        # I can proceed with all other logs.
        #
        # If any log does NOT have its receipt, throw an error.
        #
        # In other words, 'index' (or diagnose, still TBD) is the map, possibly running in parallel, and
        # 'aggregate_reports' is the reduce.
        issues = {}
        n_valid = 0

        for log_id in log_list:
            valid, summary = self.validate_log_report(log_id, check_receipt=check_receipt, write_receipt=write_receipt)
            if valid:
                n_valid += 1
            else:
                issues[log_id] = summary

        total = len(log_list)
        print(f"{n_valid} / {total} logs are valid.")
        if len(issues) > 0:
            print(f"{len(issues)} log(s) have issues:")
            for log_id, summary in issues.items():
                print(f"\t{log_id}: {summary}")

    def validate_log_report(self, log_id: str, check_receipt: bool = True, write_receipt: bool = True):
        log_root = os.path.join(self._root, log_id.lstrip("/"))
        out_fs = fsspec.filesystem(urlparse(log_root).scheme)

        if not out_fs.isdir(log_root):
            return (False, f"Log root directory for {log_id} does not exist under {self._root}.")

        # TODO(andrei): Validate camera, lidar, and other report. If things are OK, write receipt but trace back to the
        # reports used.

        camera_summary = ""
        all_cam_ok = True
        for cam in CamName:
            cam_ok, message = self.validate_camera_report(
                log_id, cam.value, check_receipt=check_receipt, write_receipt=write_receipt
            )
            if not cam_ok:
                camera_summary += f"\t{cam.value}: {message}\n"
                all_cam_ok = False
        lidar_ok, lidar_status = self.validate_lidar_report(
            log_id, check_receipt=check_receipt, write_receipt=write_receipt
        )

        other_summary = ""

        if lidar_ok and all_cam_ok:
            if write_receipt:
                receipt_fpath = os.path.join(log_root, VALIDATION_CANARY_FNAME)
                with out_fs.open(receipt_fpath, "w") as f:
                    f.write("OK, 'other' not checked")

            return (True, "")
        else:
            return (False, lidar_status + camera_summary + other_summary)

    def diagnose_misc(self, log_id: str, out_index_fpath: str, check: bool) -> List[Tuple[str, str]]:
        """Diagnoses non-sensor data, like poses, GPS, metadata, etc. Returns a list of errors, empty if all is well."""
        fs = fsspec.filesystem(urlparse(self._root).scheme)
        log_root_uri = os.path.join(self._root, log_id)
        log_reader = LogReader(log_root_uri=log_root_uri)

        errors: List[Tuple[str, str]] = []

        # Active district information is not very important
        try:
            with fs.open(os.path.join(log_root_uri, "active_district.npz.lz4"), "rb") as raw_f:
                with lz4.frame.open(raw_f) as f:
                    ad = np.load(f)["data"]
                    if not isinstance(ad, np.ndarray):
                        errors.append(("active_district", "not_ndarray"))

                    print("Active district OK:", ad.shape, ad.dtype)
        except (RuntimeError, ValueError) as err:
            errors.append(("active district", "invalid" + str(err)))

        # TODO(andrei): We probably want to provide poses as something uncompressed, since LZMA decoding is a few
        # seconds per log. For training we will cache this in some index files, but in general it's annoying to wait
        # 4-5 seconds on a modern PC to read 50MiB of data...
        try:
            mrp = log_reader.map_relative_poses_dense
            assert mrp["time"].min() > 0
        except (RuntimeError, ValueError) as err:
            errors.append(("map_relative_poses_dense", "invalid"))

        try:
            raw_wgs_dense = log_reader.raw_wgs84_poses_dense
            if raw_wgs_dense[:, 0].min() <= 0:
                errors.append(("raw_wgs84_poses_dense", "negative or zero time"))
            min_lat = raw_wgs_dense[:, 2].min()
            max_lat = raw_wgs_dense[:, 2].max()
            if min_lat < 40.00:
                errors.append(("raw_wgs84_poses_dense", f"invalid min latitude {min_lat:.8f}"))
            if max_lat > 40.95:
                errors.append(("raw_wgs84_poses_dense", f"invalid max latitude {max_lat:.8f}"))
        except (RuntimeError, ValueError) as err:
            errors.append(("raw_wgs84_poses_dense", "general error: " + str(err)))

        # NOTE(andrei): Very long story, but semi-valid YAML due to weird syntax quirks. We can probably transpile these
        # files into JSON without too much effort if there is interest.
        if not fs.isfile(os.path.join(log_root_uri, "all_vehicle_data.pseudo_yaml.npy.lz4")):
            errors.append(("all_vehicle_data", "missing"))

        # NOTE(andrei): The metadata won't be super useful to end users, since it's mostly internal stuff like a list
        # of active sensors, calibration, etc., which is already obvious by looking at the data files.
        try:
            with fs.open(os.path.join(self._root, log_id, "log_metadata.json"), "r") as f:
                meta = json.load(f)
                assert meta is not None
        except (RuntimeError, ValueError) as err:
            errors.append(("log_metadata", "invalid" + str(err)))

        detections_uri = os.path.join(log_root_uri, "detections.npz.lz4")
        if fs.isfile(detections_uri):
            try:
                # These are perception outputs but not reliable, don't assume them to be very good, I have no idea what
                # model was used when we dumped this data. It was very likely not a good production one! :(
                with fs.open(detections_uri, "rb") as raw_f:
                    with lz4.frame.open(raw_f, "rb") as f:
                        # Failure to specify the encoding causes an unpickling error.
                        dets = np.load(f, allow_pickle=True, encoding="latin1")["data"]
                        # print(dets.files)
                        assert dets is not None
                print(len(dets), "dets")
            except (RuntimeError, ValueError) as err:
                errors.append(("detections", "invalid" + str(err)))

        else:
            errors.append(("detections", "missing"))

        try:
            with fs.open(os.path.join(log_root_uri, "auto_state.npz.lz4"), "rb") as raw_f:
                with lz4.frame.open(raw_f, "rb") as f:
                    auto_state = np.load(f, allow_pickle=True, encoding="latin1")["data"]
                    assert auto_state is not None

        except (RuntimeError, ValueError) as err:
            errors.append(("auto_state", "invalid" + str(err)))

        try:
            # Probably useless dump data. We should remove this as it's auxiliary data that's not really useful to end
            # users.
            with fs.open(os.path.join(log_root_uri, "fitted_trajectory.npz.lz4"), "rb") as raw_f:
                with lz4.frame.open(raw_f, "rb") as f:
                    fitted_trajectory = np.load(f, allow_pickle=True, encoding="latin1")["data"]
                    assert fitted_trajectory is not None

        except (RuntimeError, ValueError) as err:
            errors.append(("fitted_trajectory", "invalid" + str(err)))

        try:
            log_reader.calib()
        except (RuntimeError, ValueError) as err:
            errors.append(("monocular_calib", "invalid" + str(err)))

        try:
            log_reader.stereo_calib()
        except (RuntimeError, ValueError) as err:
            errors.append(("stereo_calib", "invalid" + str(err)))

        try:
            with fs.open(os.path.join(log_root_uri, "raw_calibration.yml"), "r") as raw_f:
                cc = yaml.load(raw_f, Loader=SafeLoaderIgnoreUnknown)
                assert cc is not None
        except (RuntimeError, ValueError) as err:
            errors.append(("raw_calibration", "invalid" + str(err)))

        try:
            with fs.open(os.path.join(log_root_uri, "raw_imu.npz.lz4"), "rb") as raw_f:
                with lz4.frame.open(raw_f, "rb") as f:
                    raw_imu = np.load(f, allow_pickle=True, encoding="latin1")["data"]
                    assert raw_imu is not None
                    print(raw_imu.dtype)
                    print(len(raw_imu))
                    assert raw_imu["packet_capture_time_s"].min() > 0
        except RuntimeError as err:
            errors.append(("raw_imu", "invalid" + str(err)))
        except ValueError as err:
            errors.append(("raw_imu", "invalid" + str(err)))

        try:
            with fs.open(os.path.join(log_root_uri, "raw_utms.npz.lz4"), "rb") as raw_f:
                with lz4.frame.open(raw_f, "rb") as f:
                    raw_utm = np.load(f, allow_pickle=True, encoding="latin1")["data"]
                    assert raw_utm is not None
                    # print(raw_utm.dtype)
        except RuntimeError as err:
            errors.append(("raw_utm", "invalid" + str(err)))

        # As far as I remember, traffic light states are 100% the outputs of some camera-based model, and NOT human
        # labeled.
        tl_url = os.path.join(log_root_uri, "traffic_lights.npz.lz4")
        if fs.isfile(tl_url):
            try:
                with fs.open(tl_url, "rb") as raw_f:
                    with lz4.frame.open(raw_f, "rb") as f:
                        tl = np.load(f, allow_pickle=True, encoding="latin1")["data"]
                        assert tl is not None
                        print(tl.dtype)
                        print(len(tl))
            except RuntimeError as err:
                errors.append(("traffic_lights", "invalid" + str(err)))
        else:
            errors.append(("traffic_lights", "missing"))

        vs_url = os.path.join(log_root_uri, "vehicle_state.npz.lz4")
        if fs.isfile(vs_url):
            try:

                with fs.open(vs_url, "rb") as raw_f:
                    with lz4.frame.open(raw_f, "rb") as f:
                        vs = np.load(f, allow_pickle=True, encoding="latin1")["data"]
                        assert vs is not None
                        # print(vs.dtype)
                        # print(len(vs))
            except RuntimeError as err:
                errors.append(("vehicle_state", "invalid" + str(err)))
        else:
            errors.append(("vehicle_state", "missing"))

        # XXX(andrei): Write this as a report entry! Discriminate between WARNING and ERROR. The difference is about
        # which files are missing. E.g., tl states missing is a warning, but core metadata, poses, or calibration
        # missing is an error.
        if check and len(errors) > 0:
            print(f"{len(errors)} errors found in {log_id}!")
            for err in errors:
                print("\t- " + str(err))

        return errors

    def validate_camera_report(
        self,
        log_id: str,
        cam_name: str,
        check_receipt: bool,
        write_receipt: bool,
    ):
        log_root = os.path.join(self._root, log_id.lstrip("/"))
        cam_dir = self.get_cam_dir(log_root, cam_name)

        out_index_fpath = os.path.join(cam_dir, "index")
        report_loc = os.path.join(out_index_fpath, "report.csv")
        canary_loc = os.path.join(out_index_fpath, VALIDATION_CANARY_FNAME)

        fs = fsspec.filesystem(urlparse(self._root).scheme)
        if check_receipt and fs.isfile(canary_loc):
            print("Skipping validation of", cam_dir, "since it has a receipt.")
            return (True, "canary_found")

        if not os.path.isfile(report_loc):
            return (False, "no_report")

        ok = 0
        errors = []
        warnings = []
        unmatched_frames = []

        with open(report_loc, "r") as csvfile:
            reader = csv.reader(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
            header = next(reader)
            if len(header) != 4:
                errors.append(("", -1, INVALID_REPORT, "Invalid header: " + str(header)))
            else:
                for sample_uri, timestamp, status, detail in reader:
                    if status == "OK":
                        ok += 1
                        continue
                    elif status == CAM_UNMATCHED_POSE:
                        delta_s = float(detail)
                        unmatched_frames.append((float(timestamp), delta_s, sample_uri))
                        warnings.append("unmatched frame at " + str(timestamp) + "s, delta_s=" + str(delta_s))
                    else:
                        # print(f"Problem with {sample_uri}: {status}")
                        errors.append((sample_uri, timestamp, status, detail))

        if len(unmatched_frames):
            max_delta = -1
            clusters = []
            cur_start = -1
            cur_start_idx = -1
            prev_ts = -1
            for idx, (ts, delta_s, sample_uri) in enumerate(unmatched_frames):
                # print(f"{ts:.3f}", sample_uri.split("/")[-1], delta_s)

                if cur_start == -1 or ts - prev_ts > 1.0:
                    # Add the previous cluster, if it exists
                    if cur_start != -1:
                        clusters.append((cur_start_idx, idx - 1))

                    # Start new cluster
                    cur_start = ts
                    cur_start_idx = idx
                else:
                    # Extend current cluster
                    pass

                if delta_s > max_delta:
                    max_delta = delta_s

                prev_ts = ts

            print("Max delta was ", max_delta, "sec")
            if cur_start != -1:
                clusters.append((cur_start_idx, idx))

            print(f"Found {len(clusters)} clusters of unmatched frames")
            # for c_start_idx, c_end_idx in clusters:
            #     print(
            #         f"Cluster from {unmatched_frames[c_start_idx][0]:.3f} to {unmatched_frames[c_end_idx][0]:.3f}"
            #     )

            if len(clusters) > 10:
                errors.append(("", -1, INVALID_REPORT, f"Too many clusters of unmatched frames: {len(clusters)}"))

            unmatched_ratio = len(unmatched_frames) / ok
            if ok > 100 and unmatched_ratio > 0.2:
                errors.append(("", -1, INVALID_REPORT, f"Too many unmatched frames compared to OK frames: " \
                    f"{ok=} but {len(unmatched_frames)} unmatched ones"))


        if len(errors) > 0:
            print(f"Found {len(errors)} errors in {log_id} camera data....")
            for err in errors[:10]:
                print("\t", err)
            if len(errors) > 10:
                print("...")

            return (False, f"{len(errors)} errors found")

        # Validation for unexpectedly short logs
        #
        if ok < MIN_LIDAR_FRAMES:
            print(len(errors))
            print(ok, "frames found")
            print(len(warnings))
            return (False, f"not_enough_cam_frames_{ok}")

        if len(warnings) > 0:
            print(f"Found {len(warnings)} warnings in {log_id}, camera {cam_name}.")
            print_list_with_limit(warnings, 15)

        return (True, "checked")

    def validate_lidar_report(
        self,
        log_id: str,
        lidar_name: str = "hdl64e_12_middle_front_roof",
        acceptable_wgs84_delay_s: float = 0.25,
        check_receipt: bool = True,
        write_receipt: bool = True,
    ):
        log_root = os.path.join(self._root, log_id.lstrip("/"))
        lidar_dir = self.get_lidar_dir(log_root, lidar_name)
        out_index_fpath = os.path.join(lidar_dir, "index")
        report_loc = os.path.join(out_index_fpath, "report.csv")
        canary_loc = os.path.join(out_index_fpath, VALIDATION_CANARY_FNAME)

        ok = 0
        errors = []
        warnings = []

        fs = fsspec.filesystem(urlparse(self._root).scheme)
        if check_receipt and fs.isfile(canary_loc):
            print("Skipping validation of", lidar_dir, "since it has a receipt.")
            return (True, "canary_found")

        if not os.path.isfile(report_loc):
            return (False, "no_report")

        with open(report_loc, "r") as csvfile:
            reader = csv.reader(csvfile, quotechar="|", quoting=csv.QUOTE_MINIMAL)
            header = next(reader)
            if len(header) != 4:
                errors.append(("", -1, INVALID_REPORT, "Invalid header: " + str(header)))
            else:
                for sample_uri, timestamp, status, detail in reader:
                    if status == "OK":
                        ok += 1
                        continue
                    elif status == "bad-raw-WGS84-offset":
                        offset_s = float(detail.rstrip("s"))
                        if offset_s > acceptable_wgs84_delay_s:
                            # Be a little lenient
                            warnings.append((sample_uri, timestamp, status, detail))
                    else:
                        # print(f"Problem with {sample_uri}: {status}")
                        errors.append((sample_uri, timestamp, status, detail))

        if len(errors) > 0:
            print(f"Found {len(errors)} errors in {log_id}.")
            for err in errors[:10]:
                print("\t", err)
            if len(errors) > 10:
                print("...")

            return (False, f"{len(errors)} errors found")

        # Validation for unexpectedly short logs
        #
        if ok < MIN_LIDAR_FRAMES:
            print(len(errors))
            print(ok)
            print(len(warnings))
            return (False, f"not_enough_lidar_frames_{ok}")

        if len(warnings) > 0:
            print(f"Found {len(warnings)} warnings in {log_id}.")
            to_show = 15
            for i in range(len(warnings[:to_show])):
                print(f"\t - {warnings[i]}")
            if len(warnings) > to_show:
                print(f"\t - ... and {len(warnings) - to_show} more.")

        print("OK samples: ", ok)
        return (True, "checked")

    def feh(self, log_id: str, cam_name: str = "hdcam_12_middle_front_roof_wide", frame: int = 0):
        pit30m_fs = fsspec.filesystem(urlparse(self._root).scheme)
        log_root = os.path.join(self._root, log_id.lstrip("/"))
        cam_root = self.get_cam_dir(log_root, cam_name)
        top = pit30m_fs.ls(cam_root)
        for entry in top:
            sub = pit30m_fs.ls(entry)
            for subentry in sub:
                if subentry.endswith(".webp"):
                    print(subentry)
                    break




def print_list_with_limit(lst, limit: int) -> None:
    for entry in lst[:limit]:
        print(f"\t - {entry}")
    if len(lst) > limit:
        print(f"\t - ... and {len(lst) - limit} more.")


# Trick to bypass !binary parts of YAML files before we can support them. (They are low priority and not
# very important anyway.)
class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None


SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)


if __name__ == "__main__":
    fire.Fire(MonkeyWrench)
