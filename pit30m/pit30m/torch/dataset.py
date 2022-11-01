
import os
from typing import Sequence

import torch
from torch.utils.data import Dataset

from pit30m.data.log_reader import LogReader


class Pit30MDataset(Dataset):

    def __init__(self, root_uri: str, log_ids: Sequence[str]) -> None:
        super().__init__()

        # TODO: Load UTMs so we can expose MRP in UTM.
        # self._submap_utm = ...

        self._log_readers = {
            log_id: LogReader(os.path.join(root_uri, log_id)) for log_id in log_ids
        }

        indexes = [
            (log_id, reader.get_cam_geo_index())
            for log_id, reader in self._log_readers.items()
        ]


    def __len__(self):
        pass

    def __getitem__(self, index: int) -> torch.Tensor:
        pass



class TripletPit30MDataset(Dataset):
    # TODO(andrei): Implement based on the official Pit30M benchmark split.
    pass