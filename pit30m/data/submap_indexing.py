"""Code for computing submap-to-log-chunk indexes. Exposed via the Pit30M CLI."""

import multiprocessing as mp
import os
import random
import time
from collections import defaultdict
from typing import Any
from uuid import UUID

import numpy as np
from aiohttp.client_exceptions import ServerTimeoutError
from joblib import Memory, Parallel, delayed
from tqdm import tqdm

from pit30m.config import get_pit30m_cache_dir
from pit30m.data.log_reader import LogReader

_mem = Memory(location=get_pit30m_cache_dir(), verbose=0)


@_mem.cache(verbose=0)
def compute_submap_index(
    log_ids: list[str],
    mrp_subsample: int,
    max_jobs: int = 0,
) -> dict[str, list[tuple[str, tuple, tuple]]]:
    """Please refer to the CLI's 'create_submap_index' function for detailed documentation."""
    coarse_data = _load_submap_index_inputs(log_ids, mrp_subsample, max_jobs)
    submap_id_to_chunks = defaultdict(list)
    skipped = 0

    for log_id, (submap_ids, subsampled_mrp) in tqdm(coarse_data.items()):
        if len(subsampled_mrp) <= 1:
            skipped += 1
            continue

        assert isinstance(submap_ids, list)

        # This will give us all the contiguous chunk in a log which are within some submap.
        # We then add these to our index dict so we can look up specific log chunks (log_id,
        # start time, end time) which go through a specific submap ID.
        chunks = _log_submap_info_to_chunks(submap_ids, subsampled_mrp)
        for cur_submap, cur_start, cur_end in chunks:
            # We use string UUIDs to make JSON serialization easy.
            submap_id_to_chunks[str(cur_submap)].append((log_id, cur_start, cur_end))

    print(f"Finished. Total processed {len(coarse_data)} out of which {skipped} were skipped (too short).")
    return dict(submap_id_to_chunks)


@_mem.cache(verbose=0)
def _load_subsampled_poses(log_id: str, subsample_factor: int) -> tuple[list, np.ndarray]:
    """For the given log, returns a tuple with a list of N submap UUIDs and an N x 4 array of map-relative poses.

    The array has the following columns: time, x, y, z - each relative to the corresponding map.
    """
    if subsample_factor < 1 or subsample_factor > 1_000:
        raise ValueError(f"Invalid subsample factor: {subsample_factor}")

    try:
        log_reader = LogReader(log_root_uri=os.path.join("s3://pit30m/", log_id))
        subsampled = log_reader.map_relative_poses_dense[::subsample_factor]
    except (ServerTimeoutError, IndexError) as err:
        # TODO(andrei): Rewrite with 'tenacity' if the pattern ends up getting repeated.
        print(f"Failed to load {log_id}: {err} w/ timeout error")
        time.sleep(5)
        time.sleep(random.randint(1, 5))
        try:
            # TODO(andrei): Try out with tenacity if this works.
            subsampled = log_reader.map_relative_poses_dense[::subsample_factor]
        except (ServerTimeoutError, IndexError):
            print(f"Failed to load {log_id} again. Giving up.")
            raise

    del log_reader
    subsampled_valid = subsampled[subsampled["valid"]]
    sv = subsampled_valid
    del subsampled

    all_uuid_bytes = sv["submap_id"]
    sids = [UUID(bytes=uuid_bytes.ljust(16, b"\x00")) for uuid_bytes in all_uuid_bytes]

    compact = np.stack((sv["time"], sv["x"], sv["y"], sv["z"]), axis=1)
    assert len(sids) == len(compact)
    return sids, compact


def _log_submap_info_to_chunks(submap_ids: list[UUID], mrp_compact: np.ndarray) -> list[tuple[UUID, tuple, tuple]]:
    """Returns a list of contiguous chunks of map-relative poses.

    Designed to be used with an N x 4 MRP array, where MRP is (time, mrp_x, mrp_y, mrp_z).
    TODO(andrei): Zip this with WGS84 poses for extra convenience.
    """
    chunks = []
    if len(submap_ids) == 0:
        return []

    cur_submap = submap_ids[0]
    cur_start = mrp_compact[0, :]
    last_row = None
    assert len(submap_ids) == len(mrp_compact)
    for submap_id, mrp_row in zip(submap_ids, mrp_compact):
        if submap_id != cur_submap:
            assert last_row is not None
            chunks.append((cur_submap, tuple(cur_start.tolist()), tuple(last_row.tolist())))
            cur_submap = submap_id
            cur_start = mrp_row

        last_row = mrp_row

    if last_row is not None:
        chunks.append((cur_submap, tuple(cur_start.tolist()), tuple(last_row.tolist())))
    return chunks


def _load_submap_index_inputs(
    log_ids: list[str],
    mrp_subsample: int,
    max_jobs: int = 0,
) -> Dict[str, Tuple[List, np.ndarray]]:
    """Loads all index data for processing.

    As of 2024-05-22 the code used only 64GiB of RAM for all logs at 1Hz sampling.
    """
    print(f"Will load data for {len(log_ids)} log IDs")
    oversubscribe = 2.0
    n_jobs = int(mp.cpu_count() * oversubscribe)
    if max_jobs > 0:
        n_jobs = min(n_jobs, max_jobs)

    pool = Parallel(n_jobs=n_jobs, prefer="processes")
    # A full progress bar means all jobs were submitted, not that they are done. 'mpire' supports the latter but I found
    # it to be somewhat flaky in practice.
    results = pool(delayed(_load_subsampled_poses)(log_id, mrp_subsample) for log_id in log_ids)

    assert len(log_ids) == len(results)
    return dict(zip(log_ids, results))
