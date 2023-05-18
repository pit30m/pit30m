from __future__ import annotations

from abc import abstractmethod
from enum import Enum

import numpy as np


class PartitionEnum(Enum):
    @staticmethod
    @abstractmethod
    def value_to_index(val: PartitionEnum, index: np.ndarray) -> int:
        ...

    @property
    @abstractmethod
    def path_name(self) -> str:
        ...


# Preprocessing filters invalid poses and limits the number of poses per squared metre
class PreProcessPartition(PartitionEnum):
    VALID = True
    INVALID = False

    @staticmethod
    def value_to_index(val: PreProcessPartition, index: np.ndarray) -> int:
        assert isinstance(val, PreProcessPartition), f"val must be a PreProcessPartition, not {type(val)=}"
        return index == val.value

    @property
    def path_name(self) -> str:
        return "preprocessed"


# Geographic partition into train/val/test
class GeoPartition(PartitionEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    # Some values migth be NaN, meaning they are outside the three geopartitions

    @staticmethod
    def value_to_index(val: GeoPartition, index: np.ndarray) -> int:
        assert isinstance(val, GeoPartition), f"val must be a GeoPartition, not {type(val)=}"
        return index == val.value

    @property
    def path_name(self) -> str:
        return "train_val_test"


# Query/base partitioning for retrieval/based localization
class QueryBasePartition(PartitionEnum):
    QUERY = 0
    BASE = 1

    @staticmethod
    def value_to_index(val: QueryBasePartition, index: np.ndarray) -> int:
        assert isinstance(val, QueryBasePartition), f"val must be a QueryBasePartition, not {type(val)=}"
        return index == val.value

    @property
    def path_name(self) -> str:
        return "query_base"


# Size partitioning for mid/tiny/full
class SizePartition(PartitionEnum):
    TINY = 0
    MID = 1
    FULL = 2

    @staticmethod
    def value_to_index(val: SizePartition, index: np.ndarray) -> int:
        assert isinstance(val, SizePartition), f"val must be a SizePartition, not {type(val)=}"
        if val == SizePartition.FULL:
            return np.full(len(index), True)
        if val == SizePartition.MID:
            return np.logical_or(index == SizePartition.MID.value, index == SizePartition.FULL.value)
        if val == SizePartition.TINY:
            return index == val.value

        raise ValueError(f"Invalid value for SizePartition: {val=}")

    @property
    def path_name(self) -> str:
        return "size"
