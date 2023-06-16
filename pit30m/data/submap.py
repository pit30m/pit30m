import os
import pickle as pkl
from typing import Any
from uuid import UUID

import numpy as np
from pyproj import CRS, Transformer

# See: https://epsg.io/32617
UTM_ZONE_IN_PITTSBURGH_CODE = 32617
WGS84_CODE = 4326
DEFAULT_SUBMAP_INFO_PREFIX = "submap_utm.pkl"


class SubmapPoseNotFoundException(Exception):
    """Exception raised when we do not have the pose of a requested submap."""

    def __init__(self, submap_id):
        super().__init__(f"Pose not found for submap {submap_id}")


# These logs are known to contain at least one submap for which we do not have UTM.
LOG_ID_TO_MISSING_SUBMAPS = {
    # Crash on 2023-01-29 -- 5dedc51d-5bd3-4026-fd93-dabb4744ab23 submap UUID was missing
    # Visual inspection seems to indicate this is in the city of pittsburgh, so not a highway.
    "327b7948-4e5f-4d8c-f08b-4fc44a067996": [
        UUID("5dedc51d-5bd3-4026-fd93-dabb4744ab23"),
    ],
    # Found on 2023-02-20 -- 1a8132d0-2634-4ad0-e433-ce5987ef0120 submap UUID was missing
    "c9e9e7a7-f1cb-4af8-c5c9-3a610cbcc20e": [
        UUID("1a8132d0-2634-4ad0-e433-ce5987ef0120"),
    ],
}
# Flattens (k -> [v]) to a flat list of all v's
KNOWN_MISSING_SUBMAP_IDS = [submap_id for submap_ids in LOG_ID_TO_MISSING_SUBMAPS.values() for submap_id in submap_ids]

BLANK_UUID = UUID("00000000-0000-0000-0000-000000000000")

# test-query poses have been sanitized from the dataset and are not available. The sanitization includes the submap ID,
# which for these entries has been replaced with a null UUID.
EXPECTED_INVALID_SUBMAPS = KNOWN_MISSING_SUBMAP_IDS + [BLANK_UUID]


class Singleton(type):
    _instances: dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Map(metaclass=Singleton):
    def __init__(self):
        """Singleton that represents the Pit30M topometric map.

        To get a "global" pose from a log pose, you need to take its map-relative pose, and then compose that with the
        map's UTM coordinates.

        This would break if the map spanned multiple UTM zones, in which case we'd be forced to use WGS84 spherical
        coordinates or ECEF Euclidean coordinates for everything but it doesn't. Pittsburgh and its surrounding areas
        fit comfortably within UTM Zone 17N, i.e., EPSG:32617.

        NOTE(andrei): Using the map yields accurate poses across logs but will have artifacts near some submap
        boundaries due to inaccuracies with the submap poses themselves. This is a known issue and we are hoping
        to improve the consistency in future releases of the devkit.

        For SLAM-style trajectory level evaluation (e.g., AMV-SLAM) the GT is generated with relative transforms
        so the logs will have smooth GT but not perfectly consistent between logs, or after major loop closures.
        (The loop closure part was not a major issue in AMV-SLAM upon careful manual inspection.)

        For global localization, e.g., training, we simply take care not to sample triplets (or generalizations
        thereof) which straddle submap boundaries. While not perfect, this nevertheless provides more than enough
        data to train and evaluate a wide range of global localizers.

        If you believe this interferes with a specific use case you had in mind, please don't hesitate to contact
        Andrei Barsan or any of the benchmark and SDK maintainers and we will be happy to help!
        """

        # NOTE(julieta): We've bundled this with the codebase because it is very small < 200 KB.
        # We also have a copy under s3://pit30m-data/submap_utm.pkl
        fpath = os.path.join(os.path.dirname(__file__), "submap_utm.pkl")
        with open(fpath, "rb") as f:
            submap_utm = pkl.load(f)

        submap_to_utm = {UUID(submap_id): utm_coords for submap_id, utm_coords in submap_utm.items()}
        self._submap_to_utm = submap_to_utm

        # Construct the projection object which takes UTM coordinates and converts them into WGS84 (lat, lon).
        crs_utm = CRS.from_epsg(UTM_ZONE_IN_PITTSBURGH_CODE)
        crs_wgs84 = CRS.from_epsg(WGS84_CODE)
        self._pit30m_utm_to_wgs84 = Transformer.from_crs(crs_utm, crs_wgs84)

    def to_utm(self, map_poses_xyz: np.ndarray, submap_ids: list[UUID], strict: bool = True) -> np.ndarray:
        """Returns corresponding UTM coordinates for the given pose + submap ID combinations.

        Assumes all poses are within the same UTM zone, which holds for all of Pit30M.

        Args:
            map_poses_xyz:  n-by-3 array of map-relative poses (x, y, z).
            submap_ids:     n-element array or list with each pose's submap ID.
            strict:         If True, known-bad submap IDs will raise an exception. If False, they will be ignored and
                            results will contain NaNs for those poses, leaving the caller to deal with them. Unexpected
                            missing submaps, i.e., those NOT in 'LOG_ID_TO_MISSING_SUBMAPS' will still raise an
                            exception.

        Returns:
            An n-by-3 array of MRP poses transformed to UTM coordinates (x, y, alt). x and y represent the easting and
            northing, while the altitude is the original altitude in the map.
        """
        assert len(map_poses_xyz) == len(submap_ids)
        assert (
            map_poses_xyz.ndim == 2 and map_poses_xyz.shape[1] == 3
        ), f"Must pass N x 3 map pose array, got: {map_poses_xyz.shape}"

        if len(map_poses_xyz) == 0:
            return np.empty((0, 3))

        off_utm = []
        for map_uuid in submap_ids:
            if not isinstance(map_uuid, UUID):
                raise ValueError(f"Invalid submap ID type: {type(map_uuid)})")
            try:
                off_utm.append(self._submap_to_utm[map_uuid])
            except KeyError:
                if not strict and map_uuid in EXPECTED_INVALID_SUBMAPS:
                    off_utm.append([np.nan, np.nan])
                else:
                    raise SubmapPoseNotFoundException(map_uuid) from KeyError

        off_utm_np = np.array(off_utm)
        result = np.array(map_poses_xyz)
        result[:, :2] += off_utm_np
        # Make sure NaN rows are all NaNs
        nan_mask = np.isnan(off_utm_np[:, 0])
        result[nan_mask, :] = np.nan

        return result

    def to_wgs84(self, map_poses, submap_ids: list[UUID], strict: bool = True) -> np.ndarray:
        """Converts map-relative poses to WGS84 (lat, lon, alt) coordinates.

        Args:
            map_poses:      N-dimensional array of map-relative poses ('x' and 'y' fields required).
            submap_ids:     N-element array or list with each poses's submap ID.
            strict:         Please see 'to_utm'.

        Returns:
            An (N, 3) array of WGS84 (lat, lon, alt).
        """
        utm_poses = self.to_utm(map_poses, submap_ids, strict=strict)
        # 20x faster than manually looping over the coords in Python
        # pylint: disable=unpacking-non-sequence
        wgs_lat, wgs_lon = self._pit30m_utm_to_wgs84.transform(utm_poses[:, 0, np.newaxis], utm_poses[:, 1, np.newaxis])
        return np.hstack((wgs_lat, wgs_lon, utm_poses[:, 2, np.newaxis]))
