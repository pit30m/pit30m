"""Internal tooling for data management, validation, and indexing."""

import csv
import json
import logging
import multiprocessing as mp
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import cached_property
from typing import List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse
from uuid import UUID

import fire
import fsspec
import ipdb
import lz4
import numpy as np
import yaml
from joblib import Memory, Parallel, delayed
from PIL import Image
from tqdm import tqdm

from pit30m.camera import CamName
from pit30m.data.log_reader import VELODYNE_NAME, LogReader
from pit30m.data.submap import Map
from pit30m.indexing import build_camera_index, build_lidar_index
from pit30m.util import print_list_with_limit, safe_zip

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
DEFAULT_INDEX_FNAME = "index/index_v0.npy"
ETL_CANARY_FNAME = ".pit30m_done"
PIT30M_IMAGE_CANARY_DONE = ".pit30m_image_done"
PIT30M_NONIMAGE_CANARY_DONE = ".pit30m_nonimage_done"

MIN_LIDAR_FRAMES = 10 * 60 * 3
DEFAULT_CAM_CHECK_FRACTION = 0.0025

TQDM_MIN_INTERVAL_S = 3

KNOWN_INCOMPLETE_CAMERAS = [
    # Confirmed 2022-11-24 - only has hdcam_12_middle_front_narrow_left
    "c9e9e7a7-f1cb-4af8-c5c9-3a610cbcc20e",
    # Confirmed 2022-11-24 - only has hdcam_02_starboard_front_roof_wide
    "dc3d9c11-5ec7-41cd-d4e0-ec795cdae27d",
]

cache = Memory(location=os.path.expanduser("~/.cache/pit30m"), verbose=0)


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
    INDEXED = "INDEXED"

    NEEDS_FINAL_RECEIPT = "needs_final_receipt"
    ONLY_GPU_DONE = "only_gpu_done"
    ONLY_CPU_DONE = "only_cpu_done"


def query_log_status(root, log_id: str) -> LogStatus:
    log_id = log_id.strip().strip("/")
    out_fs = fsspec.filesystem(urlparse(root).scheme)
    out_log_root = os.path.join(root, log_id)
    if not out_fs.exists(out_log_root):
        return LogStatus.NOT_ATTEMPTED
    # elif out_fs.exists(os.path.join(out_log_root, DEFAULT_INDEX_FNAME)):
    #     # TODO(andrei): Aggregate indexes or at least check they're all there unless a camera is missing.
    #     return LogStatus.INDEXED
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


def stat_sensors_for_log(root: str, log_id: str, index_version: int = 0):
    log_id = log_id.strip().strip("/")
    fs = fsspec.filesystem(urlparse(root).scheme)
    log_root = os.path.join(root, log_id)
    if not fs.exists(log_root):
        # n/A for a non-attempted log
        return None
    if not fs.exists(os.path.join(log_root, ETL_CANARY_FNAME)):
        # Dumping is incomplete (use 'stat' to get dumping status differentiation)
        return None

    if not fs.exists(os.path.join(log_root, "all_poses.npz.lz4")):
        # Safeguard: Don't make Andrei have a heart attack if the log didn't have poses to being with.
        # These were skipped for some logs INTENTIONALLY.
        return None

    cam_images = {}
    for cam in CamName:
        cam_dir = os.path.join(log_root, "cameras", cam.value)
        # Load the 'npz' since we are OK with CPU overhead for saving network bandwidth.
        # As of 2023-02-04, stat_sensors 0..750
        #  - Before:                8.2 min
        #  - w/ npz:                2.8 min
        #  - w/ npz + 2x processes: 2.2 min (1Gbps net is saturated)
        #
        index_fpath = os.path.join(cam_dir, "index", f"index_v{index_version}.npz")

        if not fs.exists(index_fpath):
            # Camera not dumped
            cam_images[cam] = -1
            continue

        with fs.open(index_fpath, "rb") as f:
            index = np.load(f, allow_pickle=False)["index"]
            cam_images[cam] = index.shape[0]

    counts = list(cam_images.values())
    if all(v == -1 for v in counts):
        # No indexing done yet, let's not make the mistake of counting this as a failure.
        return None

    return cam_images


def qls_it(root: str, log_ids: list[str], batch_size: int):
    """Parallelized iterator version of query_log_status."""
    # Over-subscribing is fine for network-bound tasks.
    pool = Parallel(n_jobs=mp.cpu_count() * 10)

    for i in range(0, len(log_ids), batch_size):
        res = pool(delayed(query_log_status)(root, log_id) for log_id in log_ids[i : min(i + batch_size, len(log_ids))])
        for element in res:
            yield element


class MonkeyWrench:
    def __init__(
        self,
        dataset_root: str = "s3://pit30m/",
        metadata_root: str = "file:///mnt/data/pit30m/",
        log_status_fpath: str = "log_status.csv",
        log_list_fpath: str = os.path.join(os.path.dirname(__file__), "all_logs.txt"),
        submap_utm_fpath: str = "s3://pit30m/submap_utm.pkl",
    ) -> None:
        """Dataset administration tool for the Pit30M dataset (geo-indexing, data integrity checks, etc.)

        Primarily designed for dataset maintainers and advanced users. For regular end users, `pit30m.cli` is almost
        always what you should use.

        Non-administrator users of the dataset won't have permissions to modify the official dataset, but they can use
        this tool to create custom indexes and subsets of the dataset, as well as using it as a reference for how the
        dataset is organized, indexed, etc.
        """
        self._root = dataset_root
        self._metadata_root = metadata_root
        self._log_status_fpath = log_status_fpath
        self._log_list_fpath = log_list_fpath
        self._submap_utm_fpath = submap_utm_fpath

        self._pool = Parallel(n_jobs=-1, batch_size=8)
        # Oversubscribed pool for I/O heavy tasks
        self._over_pool = Parallel(n_jobs=mp.cpu_count() * 4, backend="threading")

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self._logger.addHandler(handler)

    @cached_property
    def all_logs(self) -> List[UUID]:
        """Return a list of all log IDs."""
        with open(self._log_list_fpath, "r") as f:
            return [UUID(line.strip()) for line in f]

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
        # XXX(andrei): This function is deprecated.
        self.index_all_cameras(log_id=log_id, out_index_fpath=out_index_fpath, check=check)
        self.index_lidar(log_id=log_id, out_index_fpath=out_index_fpath, check=check)
        res = self.diagnose_misc(log_id=log_id, out_index_fpath=out_index_fpath, check=check)
        # XXX(andrei): Turn the misc diagnosis into a mini report too.
        if len(res) > 0:
            print("WARNING: misc non-sensor data had errors. See above for more information.")

        print("Log indexing complete.")

    # def merge_indexes(self, log_ids: List[str]):
    #     """Given a list of log IDs with indexed data, merge the indexes into a single index."""
    #     # TODO(andrei): Implement this.
    #     ...

    def get_lidar_dir(self, log_root: str, lidar_name: str) -> str:
        return os.path.join(log_root, "lidars", lidar_name.lstrip("/"))

    def get_cam_dir(self, log_root: str, cam_name: str) -> str:
        return os.path.join(log_root, "cameras", cam_name.lstrip("/"))

    # def next_to_validate(self, max: int = 10):
    #     """Next logs to validate."""
    #     pass

    def stat(self, max: int = 100, quiet: bool = False):
        all_logs = self.all_logs[:max]

        statuses = []
        for log, status in zip(all_logs, qls_it(self._root, all_logs, batch_size=250)):
            statuses.append(status)
            if not quiet:
                print(log, status)

        counts = Counter(statuses)
        print(f"Stats for up to {max} dumped Pit30M logs:")
        for status, count in counts.items():
            print(f"{status.name}: {count}")
        print("=" * 80)

    def stat_log(self, log_id: str) -> None:
        log_id = log_id.strip().strip("/")
        print("=" * 80)
        print(f"Log ID: {log_id}")
        print(query_log_status(self._root, log_id))
        print("=" * 80)
        for sensor, count in stat_sensors_for_log(self._root, log_id).items():
            print(f"{sensor.value + ':':<40} {count: 6d}")
        print("=" * 80)

    def stat_sensors(self, start: int = 0, max: int = 100, min_img: int = 10, out_root: str = "/tmp"):
        """Gets statistics over the sensors in the dataset.

        Args:
            start:      The index of the first log to analyze.
            max:        The index of the last log to analyze (exclusive).
            min_img:    The minimum number of images a sensor must have to count as "present".
            out_root:   Write detailed information to this directory.
        """
        all_logs = self.all_logs[start:max]
        pool = Parallel(n_jobs=mp.cpu_count() * 2, verbose=10)
        results = pool(delayed(stat_sensors_for_log)(self._root, log) for log in all_logs)

        total = len(results)
        no_index = 0
        none_missing = 0
        just_one_missing = []
        two_plus_missing = []
        has_stereo_total = 0
        has_surround = 0
        all_missing = []
        has_middle_front_wide = 0
        mismatched_counts = []
        assert len(results) == len(all_logs)
        for r, log_id in zip(results, all_logs):
            if r is None:
                no_index += 1
            else:
                has_stereo = (
                    r.get(CamName.MIDDLE_FRONT_NARROW_LEFT, 0) >= min_img
                    and r.get(CamName.MIDDLE_FRONT_NARROW_RIGHT) >= min_img
                )
                has_non_stereo = (
                    r[CamName.MIDDLE_FRONT_WIDE] >= min_img
                    and r[CamName.PORT_FRONT_WIDE] >= min_img
                    and r[CamName.STARBOARD_FRONT_WIDE] >= min_img
                    and r[CamName.PORT_REAR_WIDE] >= min_img
                    and r[CamName.STARBOARD_REAR_WIDE] >= min_img
                )

                missing = 0
                for cam in CamName:
                    if r[cam] < min_img:
                        missing += 1

                if missing == 0:
                    assert has_non_stereo and has_stereo
                    none_missing += 1
                elif missing == 1:
                    just_one_missing.append(log_id)
                else:
                    two_plus_missing.append(log_id)
                    if missing == 7:
                        print(f"WARNING: Log {log_id} has all missing cameras!")
                        all_missing.append(log_id)

                if has_non_stereo:
                    has_surround += 1
                if has_stereo:
                    has_stereo_total += 1

                if r[CamName.MIDDLE_FRONT_WIDE] >= min_img:
                    has_middle_front_wide += 1

                mismatched = False
                counts = np.array([count for count in r.values() if count >= min_img])
                if len(counts) > 0:
                    # print(counts.max(), counts.min(), "!!!")
                    if counts.max() - counts.min() > counts.max() * 0.1:
                        mismatched = True

                    if mismatched:
                        mismatched_counts.append(log_id)

        indexed = total - no_index
        print(f"Stats for up {start}:{max} dumped Pit30M logs:")
        print(f"Total checked:          {total}")
        print(f"Not indexed yet:        {no_index}")
        print("-" * 80)
        print(f"Indexed:                {indexed}")
        print(f"None missing:           {none_missing}")
        print(f"Just one missing:       {len(just_one_missing)}")
        print(f"Two or more missing:    {len(two_plus_missing)}")
        print(f"All missing:            {len(all_missing)}")
        print("-" * 80)
        print(f"Has stereo:             {has_stereo_total}")
        print(f"Has surround:           {has_surround}")
        print(f"Has middle front wide:  {has_middle_front_wide}")
        print("-" * 80)
        # These are out of the PRESENT cameras.
        print(f"Mismatched counts:      {len(mismatched_counts)}")
        print_list_with_limit(mismatched_counts, 10)
        print("=" * 80)

        out_one_missing_fpath = os.path.join(out_root, "one_missing_ids.txt")
        with open(out_one_missing_fpath, "w") as f:
            f.write("\n".join(map(str, just_one_missing)))

        out_two_plus_missing_fpath = os.path.join(out_root, "two_plus_missing_ids.txt")
        with open(out_two_plus_missing_fpath, "w") as f:
            f.write("\n".join(map(str, two_plus_missing)))

        all_missing_fpath = os.path.join(out_root, "all_missing_ids.txt")
        with open(all_missing_fpath, "w") as f:
            f.write("\n".join(map(str, all_missing)))

        print(f"Wrote statistic lists to {out_root}")

    def next_to_index(self, max: int = 10, input_limit: int = 300):
        """Prints internal index commands for indexing logs which need indexing."""
        all_logs = self.all_logs[:input_limit]
        to_index = []

        for log_id, status in zip(all_logs, qls_it(self._root, all_logs, batch_size=50)):
            if status == LogStatus.DONE_NOT_INDEXED:
                to_index.append(log_id)

            if len(to_index) >= max:
                break

        for log_id in to_index:
            # TODO(andrei): Iron out this command.
            print(f"python monkeywrench.py index_all_cameras {log_id} --root s3://pit30m/")

    def next_to_etl(
        self,
        max: int = 10,
        include_attempted_but_incomplete: bool = False,
        kind: str = "gpu",
        batch_size: int = 32,
        n_read_workers: int = 14,
        webp_out_quality: int = 85,
        input_limit: int = 300,
    ):
        """Computes internal dump commands for the next logs to process.

        Args:
            max:    Maximum number of logs to RETURN.
            ...
            input_limit: Maximum number of logs to CONSIDER. Useful for working our way through the dataset.
        """
        all_logs = self.all_logs[:input_limit]
        not_attempted = []
        crashed_or_in_progress = []

        attempted_but_need_gpu_job = []
        attempted_but_need_cpu_job = []
        need_only_receipt = []
        effective_list = []

        for log_id, status in zip(all_logs, qls_it(self._root, all_logs, batch_size=50)):
            if status == LogStatus.NOT_ATTEMPTED:
                not_attempted.append(log_id)
            elif status == LogStatus.ONLY_CPU_DONE:
                attempted_but_need_gpu_job.append(log_id)
            elif status == LogStatus.ONLY_GPU_DONE:
                attempted_but_need_cpu_job.append(log_id)
            elif status == LogStatus.NEEDS_FINAL_RECEIPT:
                need_only_receipt.append(log_id)
            elif status == LogStatus.CRASHED_OR_IN_PROGRESS:
                crashed_or_in_progress.append(log_id)

            effective_list = not_attempted
            if include_attempted_but_incomplete:
                effective_list += crashed_or_in_progress
                if kind == "gpu":
                    effective_list += attempted_but_need_gpu_job
                elif kind == "cpu":
                    effective_list += attempted_but_need_cpu_job
                else:
                    raise ValueError()
            # effective_list = crashed_or_in_progress

            effective_list = list(set(effective_list))
            if max > 0 and len(effective_list) >= max:
                break

        # print("Not attempted:")
        # if include_attempted_but_incomplete:
        #     print("(or attempted but incomplete logs)")
        for log_id in effective_list:
            if kind == "gpu":
                print(
                    f"python process.py etl_images {log_id} --anonymizer yolo --anonymizer-weights "
                    f"/app/weights/anon-yolo5m-exp21-best.fp16.bs{batch_size}.1280.engine --image-batch-size {batch_size} "
                    f"--meta-batch-size {batch_size * 4} --out-root s3://pit30m/ --n-read-workers {n_read_workers} "
                    f"--webp-out-quality {webp_out_quality}"
                )
            elif kind == "cpu":
                print(
                    f"python process.py etl_non_images {log_id} --out-root s3://pit30m/ "
                    f"--n-read-workers {int(n_read_workers)}"
                )

    def backup_specific_files(self, original_base_uri: str, log_id: str, out_root: str, files: List[str]):
        in_fs = fsspec.filesystem(urlparse(original_base_uri).scheme)
        out_fs = fsspec.filesystem(urlparse(out_root).scheme)
        out_fs.makedirs(os.path.join(out_root, log_id), exist_ok=True)

        log_id = log_id.strip("/")

        NO_WGS_LOG_IDS = [
            "5fab9d4e-7a12-49d5-f0f6-5da87b64f3f3",
        ]

        for file in files:
            in_uri = os.path.join(original_base_uri, log_id, file)
            out_uri = os.path.join(out_root, log_id, file)

            if out_fs.exists(out_uri):
                # print(f"File {out_uri} already exists. Skipping.")
                continue

            if not in_fs.exists(in_uri):
                # Fitted trajectories are OK to be missing.
                if "fitted" in in_uri:
                    print(f"File {in_uri} does not exist (expected). Skipping.")
                    continue
                elif "wgs84" in in_uri and log_id in NO_WGS_LOG_IDS:
                    print(f"WGS file {in_uri} does not exist (expected). Skipping.")
                    continue

                print(in_uri)
                print(log_id)
                print(type(log_id))

            with fsspec.open(in_uri, "rb") as in_f:
                with fsspec.open(out_uri, "wb") as out_f:
                    out_f.write(in_f.read())

    def backup_all_specific_files(self, original_base_uri: str, out_root: str, files: Union[str, List[str]]):
        if isinstance(files, str):
            files = files.split(",")

        print(f"Backing up files: {files}")
        for log_id in tqdm(self.all_logs):
            self.backup_specific_files(original_base_uri, log_id, out_root, files)

    # def index_all_cameras(
    #     self, log_id: str, out_index_fpath: Optional[str] = None, check: bool = True, parallel: bool = True
    # ):
    #     if parallel:
    #         n_jobs = len(CamName)
    #         print(f"Using exactly {n_jobs} jobs")
    #         with mp.Pool(processes=n_jobs) as pool:
    #             pool.starmap(
    #                 self.index_camera,
    #                 [(log_id, cam_name, out_index_fpath, check, index) for index, cam_name in enumerate(CamName)],
    #             )
    #     else:
    #         for cam in CamName:
    #             self.index_camera(log_id=log_id, cam_name=cam, out_index_fpath=out_index_fpath, check=check)

    def index_all_cameras_debug(self, idx, reindex=False, index_version: int = 0):
        log_id = self.all_logs[idx]
        print("=" * 80)
        print(f"Indexing log {log_id} ({idx + 1} / {len(self.all_logs)})")
        print("=" * 80)
        self.index_all_cameras(log_id, reindex=reindex, index_version=index_version)

    def index_all_cameras(
        self,
        log_id: str,
        out_index_dir: Optional[str] = None,
        reindex: bool = False,
        index_version: int = 0,
    ):
        """Create an index of the images in the given log.

        This is useful for quickly finding images in a given region, or for finding the closest image to a given GPS
        location.

        Building large indexes from scratch can take a while, even on the order of hours on some machines.

        Args:
            log_id:             ID of the log to process within the established root.
            out_index_fpath:    Path to the output index file. If not given, the index will be written relative to
                                the respective camera roots (highly recommended).
            reindex:            If True, the index will be rebuilt even if it already exists.
            index_version:      The kind of indexing to use.
        """
        for cam_name in CamName:
            self._logger.info(f"\n==========\n{cam_name.name}\n==========")
            self.index_camera_v2(
                log_id=log_id,
                # _root=_root,
                out_index_dir=out_index_dir,
                # submap_utm_fpath=submap_utm_fpath,
                reindex=reindex,
                cam_name=cam_name,
                index_version=index_version,
            )

    def index_camera_v2(
        self,
        log_id: str,
        cam_name: Union[CamName, str],
        out_index_dir: Optional[str] = None,
        reindex: bool = False,
        index_version: int = 0,
    ):
        """v2 indexer - parallel reading and no image loading. Please see `index_all_cameras` for info."""
        assert index_version == 0, "v0 is the only currently supported DTYPE"
        map = self.map

        if isinstance(cam_name, str):
            cam_name = CamName(cam_name)

        self._logger.info("Setting up log reader to process camera %s", cam_name.value)
        log_root = os.path.join(self._root, log_id.lstrip("/"))
        log_reader = LogReader(log_root_uri=log_root, map=map)
        cam_dir = os.path.join(log_root, "cameras", cam_name.value.lstrip("/"))

        if out_index_dir is None:
            out_index_dir = os.path.join(cam_dir, "index")

        in_scheme = urlparse(self._root).scheme
        out_scheme = urlparse(out_index_dir).scheme
        # in_fs = fsspec.filesystem(in_scheme)
        out_fs = fsspec.filesystem(out_scheme)
        #     _logger.info("out fs scheme: %s", out_scheme)

        out_index_fpath = os.path.join(out_index_dir, f"index_v{index_version}.npy")
        out_index_fpath_npz = out_index_fpath.replace(".npy", ".npz")

        self._logger.info("Checking if index %s is there... (incl npz version)", out_index_fpath)
        exists_npy = out_fs.exists(out_index_fpath)
        exists_npz = out_fs.exists(out_index_fpath_npz)
        exists = exists_npy and exists_npz
        if exists and not reindex:
            self._logger.info("Index already exists at %s and %s", out_index_fpath, out_index_fpath_npz)
            return

        self._logger.info(
            "Will build index... reindex = %s, exists_npy = %s, exists_npz = %s",
            str(reindex),
            str(exists_npy),
            str(exists_npz),
        )
        index = build_camera_index(self._root, log_reader, cam_dir, self._logger)

        # For a rather hefty log (1h20) a v0 index would be ~17MiB uncompressed per camera.
        #
        # npz makes this 1/3 of the size, so I think it's worth it. I'll save both so users can pick.
        out_fs.makedirs(out_index_dir, exist_ok=True)
        with out_fs.open(out_index_fpath, "wb") as out_f:
            np.save(out_f, index)
        with out_fs.open(out_index_fpath_npz, "wb") as out_f:
            np.savez_compressed(out_f, index=index)
        print(f"Wrote index(es) to: {out_index_fpath}")

    def index_lidar_debug(self, log_index, reindex=False, index_version: int = 0):
        log_id = self.all_logs[log_index]
        print("=" * 80)
        print(f"Indexing log LiDAR {log_id} ({log_index + 1} / {len(self.all_logs)})")
        print("=" * 80)
        return self.index_lidar(log_id, reindex=reindex, index_version=index_version, out_index_dir=None)

    def index_lidar(
        self,
        log_id: str,
        lidar_name: str = VELODYNE_NAME,
        out_index_dir: Optional[str] = None,
        reindex: bool = False,
        index_version: int = 0,
    ):
        assert index_version == 0, "v0 is the only currently supported DTYPE"
        map = self.map

        self._logger.info("Setting up log reader to process LiDAR %s", lidar_name)
        log_root = os.path.join(self._root, log_id.lstrip("/"))
        log_reader = LogReader(log_root_uri=log_root, map=map)
        lidar_dir = self.get_lidar_dir(log_root, lidar_name)

        if out_index_dir is None:
            out_index_dir = os.path.join(lidar_dir, "index")

        out_scheme = urlparse(out_index_dir).scheme
        out_fs = fsspec.filesystem(out_scheme)

        out_index_fpath = os.path.join(out_index_dir, f"index_v{index_version}.npy")
        out_index_fpath_npz = out_index_fpath.replace(".npy", ".npz")

        self._logger.info("Checking if LiDAR index %s is there... (incl npz version)", out_index_fpath)
        exists_npy = out_fs.exists(out_index_fpath)
        exists_npz = out_fs.exists(out_index_fpath_npz)
        exists = exists_npy and exists_npz
        if exists and not reindex:
            self._logger.info("LiDAR index already exists at %s and %s", out_index_fpath, out_index_fpath_npz)
            return

        self._logger.info(
            "Will build index... reindex = %s, exists_npy = %s, exists_npz = %s",
            str(reindex),
            str(exists_npy),
            str(exists_npz),
        )
        index = build_lidar_index(self._root, log_reader, lidar_dir, self._logger)

        out_fs.makedirs(out_index_dir, exist_ok=True)
        with out_fs.open(out_index_fpath, "wb") as out_f:
            np.save(out_f, index)
        with out_fs.open(out_index_fpath_npz, "wb") as out_f:
            np.savez_compressed(out_f, index=index)
        print(f"Wrote LiDAR index(es) to: {out_index_fpath}")

    def check_all_cameras_by_index(
        self,
        idx: int,
        sample_fraction: float = DEFAULT_CAM_CHECK_FRACTION,
        in_index_dir: Optional[str] = None,
        index_version: int = 0,
    ):
        return self.check_all_cameras(
            self.all_logs[idx], sample_fraction=sample_fraction, in_index_dir=in_index_dir, index_version=index_version
        )

    def check_all_cameras(
        self,
        log_id: str,
        sample_fraction: float = DEFAULT_CAM_CHECK_FRACTION,
        in_index_dir: Optional[str] = None,
        index_version: int = 0,
    ):
        for cam_name in CamName:
            self.check_camera(
                log_id,
                cam_name,
                sample_fraction=sample_fraction,
                in_index_dir=in_index_dir,
                index_version=index_version,
            )

    def check_camera(
        self,
        log_id: str,
        cam_name: Union[str, CamName],
        sample_fraction: float = DEFAULT_CAM_CHECK_FRACTION,
        in_index_dir: Optional[str] = None,
        index_version: int = 0,
        min_num_samples: int = 500,
    ):
        """Samples camera data and checks its integrity. Camera needs to have been indexed first."""
        lr = LogReader(os.path.join(self._root, log_id.strip("/")))

        scheme = urlparse(self._root).scheme
        in_fs = fsspec.filesystem(scheme)
        log_root = os.path.join(self._root, log_id.lstrip("/"))
        cam_name = CamName(cam_name) if isinstance(cam_name, str) else cam_name
        cam_dir = self.get_cam_dir(log_root, cam_name.value)

        meta = {
            "log_id": log_id,
            "cam_name": cam_name.value,
            "sample_fraction": sample_fraction,
            "in_index_dir": in_index_dir,
            "index_version": index_version,
        }
        problems = []

        try:
            idx = lr.get_cam_geo_index(cam_name)
        except FileNotFoundError:
            problems.append(f"Camera index for {cam_name} not found in log {log_id}")
            return problems, meta

        if in_index_dir is None:
            in_index_dir = os.path.join(cam_dir, "index")

        report_dir = os.path.join(os.path.dirname(in_index_dir), "reports")
        report_fpath = os.path.join(report_dir, f"report_v{index_version:02d}.json")
        report_meta_fpath = os.path.join(report_dir, f"report_v{index_version:02d}.meta.json")

        self._logger.info("Inferred report fpath to be %s", report_fpath)
        if in_fs.exists(report_fpath):
            self._logger.info("Report already exists at %s. Not checking.", report_fpath)
            return

        self._logger.info("No report. Performing randomized image checking with sample fraction %.4f", sample_fraction)
        # out_scheme = urlparse(out_index_dir).scheme
        # out_fs = fsspec.filesystem(out_scheme)

        # out_index_fpath = os.path.join(out_index_dir, f"index_v{index_version}.npy")
        # out_index_fpath_npz = out_index_fpath.replace(".npy", ".npz")

        if len(idx) < min_num_samples:
            self._logger.info("Will just check all images")
            all_idx = range(len(idx))
        else:
            start_end = 60 * 10
            i_st = range(0, start_end)
            i_end = range(len(idx) - start_end, len(idx))
            step = int(1 / sample_fraction)
            mid = range(start_end, len(idx) - start_end, step)

            all_idx = list(i_st) + list(mid) + list(i_end)

        # print(f"Will check {all_idx}")
        # print(len(all_idx) / len(idx))
        print(f"Will check {len(all_idx)}, {len(idx)}")

        pool = Parallel(n_jobs=mp.cpu_count() * 2, batch_size=16)

        def check_image(row_idx):
            idx_entry = idx[row_idx]
            try:
                cam_image = lr.get_image(cam_name, row_idx)
                img_np = cam_image.image
                if img_np.shape != EXPECTED_IMAGE_SIZE:
                    print(f"Bad image shape: {img_np.shape} (expected {EXPECTED_IMAGE_SIZE})")
                    return (
                        log_id,
                        cam_name.value,
                        idx_entry["rel_path"],
                        CAM_UNEXPECTED_SHAPE,
                        str(cam_image.image.shape),
                    )
                else:
                    # Might indicate bad cases of over/underexposure. Likely won't trigger if the sensor is covered
                    # by snow (mean is larger than 5-10), which is fine since it's valid data.
                    img_mean = img_np.mean()
                    if img_mean < 5:
                        return (log_id, cam_name.value, idx_entry["rel_path"], CAM_MEAN_TOO_LOW, str(img_mean))
                    elif img_mean > 250:
                        return (log_id, cam_name.value, idx_entry["rel_path"], CAM_MEAN_TOO_HIGH, str(img_mean))
            except Exception as err:
                return (log_id, cam_name.value, idx_entry["rel_path"], CAM_UNEXPECTED_CORRUPTION, str(err))

            return None

        progress_bar = tqdm(all_idx, desc="Checking images")
        res = pool(delayed(check_image)(row_idx) for row_idx in progress_bar)
        problems = [r for r in res if r is not None]

        if 0 != len(problems):
            print(f"{len(problems)} bad status(es) found!")
            print_list_with_limit(problems, 20)
        else:
            print("No problems found!")

        self._logger.info("Writing report to %s", report_fpath)
        with in_fs.open(report_fpath, "w") as f:
            json.dump(problems, f)
        with in_fs.open(report_meta_fpath, "w") as f:
            json.dump(meta, f)

        return problems, meta

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

    def logs_without_pose(self) -> list[str]:
        """Returns a list of log IDs that do not have a pose file."""

        def has_poses(root, log_id):
            uri = os.path.join(root, log_id, "all_poses.npz.lz4")
            fs = fsspec.filesystem(urlparse(uri).scheme, anon=True)
            return fs.exists(uri)

        log_has_pose = self._over_pool(delayed(has_poses)(self._root, log_id) for log_id in self.all_logs)
        return [log_id for log_id, has_pose in safe_zip(self.all_logs, log_has_pose) if not has_pose]

    def one_off_copy_back_poses(self, backup_root: str):
        log_ids_without_pose = self.logs_without_pose()
        self._logger.info("Found %d logs without poses.", len(log_ids_without_pose))

        backup_fs = fsspec.filesystem(urlparse(backup_root).scheme, anon=True)
        # This can't be anonymous, as we actually need maintainer credentials to write to the bucket.
        out_fs = fsspec.filesystem(urlparse(self._root).scheme)

        for log_id in log_ids_without_pose:
            source_uri = os.path.join(backup_root, log_id, "all_poses.npz.lz4")

            target_uri = os.path.join(self._root, log_id, "all_poses.npz.lz4")

            if not backup_fs.exists(source_uri):
                source_uri = os.path.join(backup_root, log_id, "all_poses.npz.xz")
                if not backup_fs.exists(source_uri):
                    print("!!! Source file does not exist: %s", source_uri)

            if out_fs.exists(target_uri):
                print("!!! Target file already exists: %s", target_uri)

            # LZMA specifically required for dealing with data backups. Long story.
            import lzma

            with backup_fs.open(source_uri, "rb") as compressed_f_in:
                with out_fs.open(target_uri, "wb") as compressed_f_out:
                    with lzma.open(compressed_f_in, "rb") as f_in:
                        with lz4.frame.open(compressed_f_out, "wb") as f_out:
                            self._logger.info("%s", source_uri)
                            self._logger.info("%s", "vvvvvv".center(80, " "))
                            self._logger.info("%s", target_uri)
                            bts = f_in.read()
                            f_out.write(bts)

                self._logger.info("Write done.")

        self._logger.info("Done.")

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
                errors.append(
                    (
                        "",
                        -1,
                        INVALID_REPORT,
                        f"Too many unmatched frames compared to OK frames: "
                        f"{ok=} but {len(unmatched_frames)} unmatched ones",
                    )
                )

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


# Trick to bypass !binary parts of YAML files before we can support them. (They are low priority and not
# very important anyway.)
class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None


SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)


if __name__ == "__main__":
    fire.Fire(MonkeyWrench)
