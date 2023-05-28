import numpy as np
from pytest import fixture

from pit30m.camera import CamName
from pit30m.data.log_reader import LogReader
from pit30m.data.partitions import GeoPartition, PreProcessPartition, QueryBasePartition
from pit30m.data.submap import DEFAULT_SUBMAP_INFO_PREFIX, Map


@fixture(name="real_map")
def _real_map() -> Map:
    return Map()


@fixture(name="real_log_reader")
def _real_log_reader(real_map: Map) -> LogReader:
    # Creates a real log reader for a cool log with lots of snow
    return LogReader("s3://pit30m/7e9b5978-0a52-401c-dcd1-65c8d9930ad8/", map=real_map)


@fixture(name="real_log_reader_with_partition")
def _real_log_reader_with_partition() -> LogReader:
    # Creates a real log reader for a cool log with lots of snow
    return LogReader(
        "s3://pit30m/7e9b5978-0a52-401c-dcd1-65c8d9930ad8/",
        partitions={PreProcessPartition.VALID},
    )


@fixture(name="real_log_reader_with_two_partitions")
def _real_log_reader_with_two_partition() -> LogReader:
    # Creates a real log reader for a cool log with lots of snow
    return LogReader(
        "s3://pit30m/7e9b5978-0a52-401c-dcd1-65c8d9930ad8/",
        partitions={PreProcessPartition.VALID, GeoPartition.TEST, QueryBasePartition.QUERY},
    )


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


def test_real_log_with_partitions(
    real_log_reader_with_partition: LogReader,
    real_log_reader_with_two_partitions: LogReader,
):
    # A log with some partitions should return a mask with some False entries
    partitions_mask = real_log_reader_with_partition._partitions_mask
    assert sum(partitions_mask["mask"]) < len(partitions_mask)

    # A log with more partitions should return a mask with even more False entries
    partitions_mask_2 = real_log_reader_with_two_partitions._partitions_mask
    assert sum(partitions_mask_2["mask"]) < sum(partitions_mask["mask"])


def test_real_log_without_partitions(real_log_reader: LogReader):
    # A log with no partitions should return a mask of all True
    partitions_mask = real_log_reader._partitions_mask
    assert sum(partitions_mask["mask"]) == len(partitions_mask)


def test_index_gets_filtered(
    real_log_reader: LogReader,
    real_log_reader_with_partition: LogReader,
):
    # A log with no partitions should return a mask of all True
    index_full = real_log_reader.get_cam_geo_index(CamName.PORT_REAR_WIDE)
    index_short = real_log_reader_with_partition.get_cam_geo_index(CamName.PORT_REAR_WIDE)

    assert len(index_full) > len(index_short)
