import numpy as np
from pit30m.indexing import associate

def test_associate_1to1():
    qq = np.array([1, 2, 3, 4])
    tt = np.array([1, 2, 3, 4])
    np.testing.assert_equal(
        np.array([0, 1, 2, 3]),
        associate(qq, tt),
    )


def test_associate_1to2():
    qq = np.array([1, 2, 3, 4])
    tt = np.array([1, 2, 3, 4, 5])
    np.testing.assert_equal(
        np.array([0, 1, 2, 3]),
        associate(qq, tt),
    )

def test_associate_2to1():
    qq = np.array([1, 2, 3, 4, 5])
    tt = np.array([1, 2, 3, 4])
    np.testing.assert_equal(
        np.array([0, 1, 2, 3, 3]),
        associate(qq, tt),
    )


def test_associate_right_side():
    qq = np.array([1, 1.9, 2, 2.1, 2.5, 2.9, 3.0, 3.1])
    tt = np.array([1, 2, 3, 4])
    np.testing.assert_equal(
        np.array([0, 1, 1, 1, 2, 2, 2, 2]),
        associate(qq, tt),
    )

