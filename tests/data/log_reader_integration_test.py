import numpy as np
import pytest
from pytest import fixture

from pit30m.camera import CamName
from pit30m.data.log_reader import LogReader
from pit30m.data.partitions import GeoPartition, PreProcessPartition, QueryBasePartition
from pit30m.data.submap import DEFAULT_SUBMAP_INFO_PREFIX, Map


@fixture(name="real_map", scope="session")
def _real_map() -> Map:
    return Map()


@fixture(name="real_log_reader", scope="session")
def _real_log_reader(real_map: Map) -> LogReader:
    # Creates a real log reader for a cool log with lots of snow
    return LogReader("s3://pit30m/7e9b5978-0a52-401c-dcd1-65c8d9930ad8/", map=real_map)


@fixture(name="real_log_reader_with_test_partition", scope="session")
def _real_log_reader_test_query(real_map: Map) -> LogReader:
    # Creates a real log reader for a cool log with lots of snow
    return LogReader(
        "s3://pit30m/7e9b5978-0a52-401c-dcd1-65c8d9930ad8/", partitions={GeoPartition.TEST, QueryBasePartition.QUERY}
    )


@fixture(name="real_log_reader_with_partition", scope="session")
def _real_log_reader_with_partition() -> LogReader:
    # Creates a real log reader for a cool log with lots of snow
    return LogReader(
        "s3://pit30m/7e9b5978-0a52-401c-dcd1-65c8d9930ad8/",
        partitions={PreProcessPartition.VALID},
    )


@fixture(name="real_log_reader_with_two_partitions", scope="session")
def _real_log_reader_with_two_partition() -> LogReader:
    # Creates a real log reader for a cool log with lots of snow
    return LogReader(
        "s3://pit30m/7e9b5978-0a52-401c-dcd1-65c8d9930ad8/",
        partitions={PreProcessPartition.VALID, GeoPartition.TEST, QueryBasePartition.QUERY},
    )


def test_real_log_can_fetch_raw_pose_data(real_log_reader: LogReader):
    poses = real_log_reader.raw_pose_data
    assert len(poses) > 0


def test_real_log_can_fetch_raw_wgs84_poses(real_log_reader: LogReader):
    poses = real_log_reader.raw_wgs84_poses
    assert len(poses) > 0


def test_real_log_with_test_partition_can_fetch_dense_utm(
    real_log_reader_with_test_partition: LogReader,
):
    mask, poses = real_log_reader_with_test_partition.utm_poses_dense
    assert np.sum(~np.isnan(poses)) > 0
    assert len(poses) > 1000, "Expected to have a reasonable number of UTM poses."
    assert len(poses) == len(mask)
    assert len(poses) == np.sum(~np.isnan(mask)), "All UTMs must be invalid on a log reading test query data!"


def test_lidar_index_is_sorted(real_log_reader: LogReader):
    index = real_log_reader.get_lidar_geo_index()
    image_times = index["lidar_time"]
    assert np.all(np.diff(image_times) >= 0)


def test_camera_index_is_sorted(real_log_reader: LogReader):
    index = real_log_reader.get_cam_geo_index(CamName.PORT_FRONT_WIDE)
    image_times = index["img_time"]
    assert np.all(np.diff(image_times) >= 0)


def test_real_log_read_index(real_log_reader: LogReader):
    index = real_log_reader.get_cam_geo_index(CamName.PORT_REAR_WIDE)
    assert len(index) > 1000
    assert "webp" in index["rel_path"][42]


def test_real_log_can_fetch_partition_index(
    real_log_reader: LogReader,
    real_log_reader_with_partition: LogReader,
    real_log_reader_with_two_partitions: LogReader,
):
    """A log without arguments should filter out nothing"""
    index = real_log_reader.partitions_mask
    assert sum(index) == len(index)

    # If loading a partition, some measurements should be filtered out
    index = real_log_reader_with_partition.partitions_mask
    assert sum(index) < len(index)

    # If loading more partitions, even fewer measurements should be available
    smallest_index = real_log_reader_with_two_partitions.partitions_mask
    assert sum(smallest_index) < sum(index)
