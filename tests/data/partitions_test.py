import numpy as np

from pit30m.data.partitions import (
    PreProcessPartition,
)


def test_PreProcessPartition():
    np.testing.assert_equal(
        PreProcessPartition.value_to_index(PreProcessPartition.VALID, np.array([True, False])),
        np.array([True, False]),
    )

    np.testing.assert_equal(
        PreProcessPartition.value_to_index(PreProcessPartition.INVALID, np.array([True, False])),
        np.array([False, True]),
    )
