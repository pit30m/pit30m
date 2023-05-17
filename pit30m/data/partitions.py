from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Iterable, Tuple, Union
from urllib.parse import urlparse
from uuid import UUID

import fsspec
import numpy as np


# Preprocessing filters invalid poses and limits the number of poses per squared metre
class PreProcessPartition(Enum):
    VALID = True
    INVALID = False

    @staticmethod
    def value_to_index(val: PreProcessPartition.value, index: np.ndarray) -> int:
        return index == val.value


# Geographic partition into train/val/test
class GeoPartition(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    # Some values migth be NaN, meaning they are outside the three geopartitions

    @staticmethod
    def value_to_index(val: GeoPartition.value, index: np.ndarray) -> int:
        return index == val.value


# Query/base partitioning for retrieval/based localization
class QueryBasePartition(Enum):
    QUERY = 0
    BASE = 1

    @staticmethod
    def value_to_index(val: QueryBasePartition.value, index: np.ndarray) -> int:
        return index == val.value


# Size partitioning for mid/tiny/full
class SizePartition(Enum):
    TINY = 0
    MID = 1
    FULL = 2

    @staticmethod
    def value_to_index(val: SizePartition.value, index: np.ndarray) -> int:
        if val == SizePartition.FULL:
            return np.full(len(index), True)
        elif val == SizePartition.MID:
            return np.logical_or(index == SizePartition.MID.value, index == SizePartition.FULL.value)
        elif val == SizePartition.TINY:
            return index == val.value


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


def combine_partitions(
    partititons_dict: Dict[
        Union[PreProcessPartition, GeoPartition, QueryBasePartition, SizePartition], Tuple[np.ndarray, Enum.value]
    ]
) -> np.ndarray:
    """Converts a bunch of partitions to a single boolean index that can be used for filtering in the logreader"""

    indices = []
    for partition, (index, value) in partititons_dict.items():
        processes_index = partition.value_to_index(value, index)
        indices.append(processes_index)

    combined_indices = np.logical_and.reduce(indices)
    return combined_indices
