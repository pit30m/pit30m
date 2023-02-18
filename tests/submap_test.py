import uuid

import numpy as np
import pytest

from pit30m.data.submap import Map


@pytest.fixture(name="dummy_submap_to_utm")
def _dummy_submap_to_utm():
    return {
        "a": (0, 0),
        "b": (10, 10),
        "c": (20, 20),
    }


@pytest.fixture(name="dummy_real_submap_to_utm")
def _dummy_real_submap_to_utm():
    return {
        # D. L. Pratt Building at UofT
        uuid.uuid3(uuid.NAMESPACE_URL, "pratt"): (632315.689399, 4821086.845690),
    }


def test_utm(dummy_submap_to_utm):
    map = Map(submap_to_utm=dummy_submap_to_utm)

    fake_poses = np.array(
        [
            [7, 7],
            [1, 1],
            [-2, 0.3],
        ]
    )
    fake_submap_ids = ["a", "b", "c"]

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


def test_wgs84(dummy_real_submap_to_utm):
    map = Map(submap_to_utm=dummy_real_submap_to_utm)

    fake_poses = np.array(
        [
            [0.33, -0.33],
        ]
    )
    fake_submap_ids = [uuid.uuid3(uuid.NAMESPACE_URL, "pratt")]

    wgs84_coords = map.to_wgs84(fake_poses, fake_submap_ids)
    # GT computed using epsg.io
    np.testing.assert_allclose(wgs84_coords, np.array([[43.531002, -79.3625012]]))
