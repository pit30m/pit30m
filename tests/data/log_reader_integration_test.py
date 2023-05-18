from pytest import fixture

from pit30m.camera import CamName
from pit30m.data.log_reader import LogReader
from pit30m.data.partitions import (
    GeoPartition,
    PreProcessPartition,
    QueryBasePartition,
    SizePartition,
)
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
        partitions={PreProcessPartition: PreProcessPartition.VALID},
    )


@fixture(name="real_log_reader_with_partition")
def _real_log_reader_with_two_partition() -> LogReader:
    # Creates a real log reader for a cool log with lots of snow
    return LogReader(
        "s3://pit30m/7e9b5978-0a52-401c-dcd1-65c8d9930ad8/",
        partitions={PreProcessPartition: PreProcessPartition.VALID, GeoPartition: GeoPartition.TEST},
    )


def test_real_log_read_index(real_log_reader: LogReader):
    index = real_log_reader.get_cam_geo_index(CamName.PORT_REAR_WIDE)
    assert len(index) > 1000
    assert "webp" in index["rel_path"][42]


def test_real_log_can_fetch_partition_index(
    real_log_reader: LogReader,
    real_log_reader_with_partition: LogReader,
):
    """A log without arguments should filter out nothing"""
    index = real_log_reader.partitions_index
    assert sum(index) == len(index)

    # If loading a partition, some measurements should be filtered outs
    index = real_log_reader_with_partition.partitions_index
    assert sum(index) < len(index)
