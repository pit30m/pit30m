import os
from enum import Enum
from typing import Dict, Iterable, Union
from urllib.parse import urlparse
from uuid import UUID

import fsspec
import numpy as np


# Preprocessing filters invalid poses and limits the number of poses per squared metre
class PreProcessPartition(Enum):
    VALID = True
    INVALID = False


# Geograohic partition into train/val/test
class GeoPartition(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2


# Query/base partitioning for retrieval/based localization
class QueryBasePartition(Enum):
    QUERY = 0
    BASE = 1


# Size partitioning for mid/tiny/full
class SizePartition(Enum):
    TINY = 0
    MID = 1
    FULL = 2


PARTITION_TO_NAME = {
    PreProcessPartition: "preprocessed",
    GeoPartition: "train_val_test",
    QueryBasePartition: "query_base",
    SizePartition: "size",
}


def fetch_partitions(
    log_id: UUID,
    partitions_to_fetch: Iterable[Union[PreProcessPartition, GeoPartition, QueryBasePartition, SizePartition]],
) -> Dict[Union[PreProcessPartition, GeoPartition, QueryBasePartition, SizePartition], np.ndarray]:
    """Fetch partitions from s3 for a given log"""

    dir = "s3://pit30m/partitions/"
    fs = fsspec.filesystem(urlparse(dir).scheme, anon=True)

    partitions = {}
    for partition in partitions_to_fetch:
        partition_fpath = os.path.join(dir, PARTITION_TO_NAME[partition], f"{log_id}.npz")
        if not fs.exists(partition_fpath):
            raise ValueError(f"Partition file not found: {partition_fpath}")

        with fs.open(partition_fpath, "rb") as f:
            partitions[partition] = np.load(f)["partition"]

    return partitions
