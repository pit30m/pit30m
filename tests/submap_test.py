import uuid
from uuid import UUID

import numpy as np

from pit30m.data.submap import Map


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
            [7, 7],
            [1, 1],
            [-2, 0.3],
        ]
    )

    utm_coords = map.to_utm(fake_poses, fake_submap_ids)
    np.testing.assert_allclose(
        utm_coords,
        np.array(
            [
                [7, 7],
                [11, 11],
                [18, 20.3],
            ]
        ),
    )


def test_wgs84():
    map = Map()

    fake_poses = np.array(
        [
            [0.33, -0.33],
        ]
    )
    fake_submap_ids = [uuid.uuid3(uuid.NAMESPACE_URL, "pratt")]
    map._submap_to_utm[fake_submap_ids[0]] = (632315.689399, 4821086.845690)

    wgs84_coords = map.to_wgs84(fake_poses, fake_submap_ids)
    # GT computed using epsg.io
    np.testing.assert_allclose(wgs84_coords, np.array([[43.531002, -79.3625012]]))
