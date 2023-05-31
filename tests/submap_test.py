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
                [7, 7, 0],
                [11, 11, 0],
                [18, 20.3, 16],
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


def test_anonymized_entry():
    """Ensure that handling an entry with a null submap ID yields empty UTMs."""
    blank_uuid = UUID("00000000-0000-0000-0000-000000000000")
    map_ = Map()
    mrp_xyz_zeros = np.array([
        [0.0, 0.0, 0.0]
    ])
    utm_poses = map_.to_utm(mrp_xyz_zeros, [blank_uuid], strict=False)
    assert np.allclose(utm_poses, np.array([[np.nan, np.nan, np.nan]]), equal_nan=True)

    mrp_xyz_nonzeros = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    utm_poses = map_.to_utm(mrp_xyz_nonzeros, [blank_uuid, blank_uuid], strict=False)
    assert np.allclose(utm_poses, np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]), equal_nan=True)

    mrp_xyz_nans = np.array([
        [np.nan, np.nan, np.nan],
    ])
    utm_poses = map_.to_utm(mrp_xyz_nans, [blank_uuid], strict=False)
    assert np.allclose(utm_poses, np.array([[np.nan, np.nan, np.nan]]), equal_nan=True)

def test_missing_submap_utm():
    """Integration test for handling the very few submaps with no UTM coordinates."""
    map_ = Map()

    completely_unexpected_submap_uuid = UUID("c343709f-ffff-ffff-c626-01df08b26100")

    for log_id in LOG_ID_TO_MISSING_SUBMAPS.keys():
        lr = LogReader(log_root_uri=f"s3://pit30m/{log_id}/")
        sids = [
            UUID(bytes=submap_uuid_bytes.ljust(16, b"\x00"))
            for submap_uuid_bytes in lr.map_relative_poses_dense["submap_id"]
        ]
        mrp_xyz = np.stack(
            (
                lr.map_relative_poses_dense["x"],
                lr.map_relative_poses_dense["y"],
                lr.map_relative_poses_dense["z"],
            ),
            axis=1,
        )
        with pytest.raises(SubmapPoseNotFoundException):
            utm_poses = map_.to_utm(mrp_xyz, sids)
        with pytest.raises(SubmapPoseNotFoundException):
            utm_poses = map_.to_utm(mrp_xyz, sids, strict=True)

        # Completely unexpected submap IDs should always raise
        mutated_sids = list(sids)
        mutated_sids[0] = completely_unexpected_submap_uuid
        with pytest.raises(SubmapPoseNotFoundException):
            utm_poses = map_.to_utm(mrp_xyz, mutated_sids, strict=True)
        with pytest.raises(SubmapPoseNotFoundException):
            utm_poses = map_.to_utm(mrp_xyz, mutated_sids, strict=False)

        # Finally, make sure we get sensible NaN rows in the non-strict case
        utm_poses = map_.to_utm(mrp_xyz, sids, strict=False)
        assert np.isnan(utm_poses.ravel()).sum() > 0
        assert np.isnan(utm_poses.ravel()).sum() % 3 == 0  # A row is either all NaN or all non-NaN
