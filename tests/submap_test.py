import uuid
from uuid import UUID

import numpy as np
import pytest

from pit30m.data.log_reader import LogReader
from pit30m.data.submap import (
    LOG_ID_TO_MISSING_SUBMAPS,
    Map,
    SubmapPoseNotFoundException,
)


def test_singleton():
    map_instance_1 = Map()
    map_instance_2 = Map()
    np.testing.assert_(map_instance_1 is map_instance_2)


def test_utm():
    map = Map()

    fake_submap_ids = [
        UUID("c343709f-f520-41c8-c626-01df08b26100"),
        UUID("c343709f-f520-41c8-c626-01df08b26101"),
        UUID("c343709f-f520-41c8-c626-01df08b26102"),
    ]
    map._submap_to_utm[fake_submap_ids[0]] = (0, 0)
    map._submap_to_utm[fake_submap_ids[1]] = (10, 10)
    map._submap_to_utm[fake_submap_ids[2]] = (20, 20)

    fake_poses = np.array(
        [
            [7, 7, 0],
            [1, 1, 0],
            [-2, 0.3, 16],
        ]
    )

    utm_coords = map.to_utm(fake_poses, fake_submap_ids)
    np.testing.assert_allclose(
        utm_coords,
        np.array(
            [
                [7, 7,0],
                [11, 11,0],
                [18, 20.3,16],
            ]
        ),
    )


def test_wgs84():
    map = Map()

    fake_poses = np.array(
        [
            [0.33, -0.33, 0.0],
        ]
    )
    fake_submap_ids = [uuid.uuid3(uuid.NAMESPACE_URL, "pratt")]
    map._submap_to_utm[fake_submap_ids[0]] = (632315.689399, 4821086.845690)

    wgs84_coords = map.to_wgs84(fake_poses, fake_submap_ids)
    # GT computed using epsg.io
    np.testing.assert_allclose(wgs84_coords, np.array([[43.531002, -79.3625012, 0.0]]))


def test_missing_submap_utm():
    """Integration test for handling the very few submaps with no UTM coordinates."""
    map_ = Map()

    for log_id in LOG_ID_TO_MISSING_SUBMAPS.keys():
        lr = LogReader(log_root_uri=f"s3://pit30m/{log_id}/")
        sids = []
        # sids = [UUID(bytes=submap_uuid_bytes) for submap_uuid_bytes in lr.map_relative_poses_dense["submap_id"]]
        for submap_uuid_bytes in lr.map_relative_poses_dense["submap_id"]:
            submap_uuid_bytes = submap_uuid_bytes.ljust(16, b"\x00")
            sids.append(UUID(bytes=submap_uuid_bytes))

        mrp_xyz = np.hstack((lr.map_relative_poses_dense["x"][:, np.newaxis],
                         lr.map_relative_poses_dense["y"][:, np.newaxis],
                         lr.map_relative_poses_dense["z"][:, np.newaxis],
                         ))
        with pytest.raises(SubmapPoseNotFoundException):
            utm_poses = map_.to_utm(mrp_xyz, sids)
            assert np.isnan(utm_poses.ravel()).sum() == 0
        with pytest.raises(SubmapPoseNotFoundException):
            utm_poses = map_.to_utm(mrp_xyz, sids, strict=True)
            assert np.isnan(utm_poses.ravel()).sum() == 0

        utm_poses = map_.to_utm(mrp_xyz, sids, strict=False)
        assert np.isnan(utm_poses.ravel()).sum() > 0
        assert np.isnan(utm_poses.ravel()).sum() % 3 == 0   # A row is either all NaN or all non-NaN


