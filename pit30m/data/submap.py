import pickle as pkl
import uuid
import ipdb
from uuid import UUID
from typing import Mapping, Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import fsspec
from pyproj import CRS, Transformer
from joblib import Memory

from pit30m.indexing import _FSSPEC_ARGS

# See: https://epsg.io/32617
UTM_ZONE_IN_PITTSBURGH_CODE = 32617
WGS84_CODE = 4326

memory = Memory(location="/tmp/pit30m", verbose=0)

class Map:

    @staticmethod
    @memory.cache()
    def from_submap_utm_uri(uri: str) -> "Map":
        """Loads a submap index from the given URI."""
        with fsspec.open(uri, "rb", **_FSSPEC_ARGS) as f:
            submap_utm = pkl.load(f)

        # Store actual UUID objects for efficiency
        submap_to_utm = {UUID(submap_id): utm_coords for submap_id, utm_coords in submap_utm.items()}
        return Map(submap_to_utm=submap_to_utm)

    def __init__(self, submap_to_utm: Mapping[UUID, Tuple[float, float]]):
        """Represents the Pit30M topometric map.

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

        TODO(andrei): Should we bundle this info in the devkit itself for fast retrieval? It should be pretty tiny.
        At the very least we should cache it somewhere like in "~/.cache/pit30m".
        """
        self._submap_to_utm = submap_to_utm

        # Construct the projection object which takes UTM coordinates and converts them into WGS84 (lat, lon).
        crs_utm = CRS.from_epsg(UTM_ZONE_IN_PITTSBURGH_CODE)
        crs_wgs84 = CRS.from_epsg(WGS84_CODE)
        self._pit30m_utm_to_wgs84 = Transformer.from_crs(crs_utm, crs_wgs84)


    def to_utm(self, map_poses_xyz, submap_ids):
        """Returns corresponding UTM coordinates for the given pose + submap ID combinations.

        TODO(andrei): We probably want altitude as well.

        Assumes all poses are within the same UTM zone, which holds for all of Pit30M.
        """
        assert len(map_poses_xyz) == len(submap_ids)
        # print(map_poses_xyz.shape)
        # print(map_poses_xyz.dtype)

        off_utm = []
        for map_uuid in submap_ids:
            off_utm.append(self._submap_to_utm[map_uuid])

        off_utm = np.array(off_utm)
        return map_poses_xyz[:, :2] + off_utm

    def to_wgs84(self, map_poses, submap_ids):
        """Converts map-relative poses to WGS84 (lat, lon) tuples.

        Args:
            map_poses:    N-dimensional array of map-relative poses ('x' and 'y' fields required).
            submap_ids:   N-element array or list with each poses's submap ID.

        Returns:
            An (N, 2) array of WGS84 (lat, lon).

        """
        utm_poses = self.to_utm(map_poses, submap_ids)
        # 20x faster than manually looping over the coords in Python
        wgs_lat, wgs_lon = self._pit30m_utm_to_wgs84.transform(
            utm_poses[:, 0, np.newaxis],
            utm_poses[:, 1, np.newaxis]
        )
        return np.hstack((wgs_lat, wgs_lon))