"""Utility functions for timing, data association, and indexing."""

import numpy as np


def associate(query_timestamps: np.ndarray, target_timestamps: np.ndarray, max_delta_s: float = 0.5) -> np.ndarray:
    """Associates timestamps from two arrays, returning the indices of the closest matches.

    The result will contain an index into 'db_ts' for each entry in 'q_ts' (query timestamps).
    All timestamps are expected to be float64, seconds.

    Args:
        query_timestamps:   The timestamps to find matches for.
        target_timestamps:  The timestamps to match against. Must be sorted!!
        max_delta_s:        Warn if the gap between the query and the result is bigger than this. Note that we will
                            still return this index, so the end user is responsible for filtering out the results.
    """
    # TODO(andrei): Within a few ms at worst, poses are evenly spaced. We can use
    # arithmetic to dramatically constrain our search range to turn log n binary search
    # into a constant time look-up.
    #
    # TODO(andrei): Interpolate poses linearly. Poses are 100Hz so nearest-neighbor lookups can
    # be at most 10-ish ms off which can cause 0.33m of error for 33 mps driving (120kph) in
    # a worst-case scenario, so not trivial.
    #
    # Fortunately, for such small intervals, linear interpolation should be OK.
    result = np.zeros(query_timestamps.shape, dtype=np.int64)

    # XXX(andrei): np.interpolate, yo!
    for q_idx, q_ts in enumerate(query_timestamps):
        target_idx = np.searchsorted(target_timestamps, q_ts, side="left")
        prev = target_idx - 1
        next = target_idx

        if next < len(target_timestamps):
            delta_prev = abs(q_ts - target_timestamps[prev])
            delta_next = abs(q_ts - target_timestamps[next])
            if delta_prev < delta_next:
                target_idx = prev
            else:
                target_idx = next
        else:
            target_idx = target_idx - 1

        delta_s = abs(q_ts - target_timestamps[target_idx])
        if delta_s > max_delta_s:
            print(f"WARNING: Timestamp association gap is {delta_s:.3f}s for query {q_idx}.")
        result[q_idx] = target_idx

    return result


