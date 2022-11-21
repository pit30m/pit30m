
import os
from typing import Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.datapipes import S3FileLister, S3FileLoader, IterDataPipe

from pit30m.data.log_reader import LogReader

# TODO(andrei): Configure shard-aware shuffling.
# TODO(andrei): This should be right but update as needed if we end up getting the Open Data bucket somewhere else.
PIT30M_AWS_REGION = "us-east-1"

class Pit30MLogFilePipe(IterDataPipe):
    def __init__(self, root: str, logs: Sequence[str]):
        """Root data pipe which just lists all images in the given logs."""
        pass



def demo_datapipe():
    pass

# TODO(andrei): Explore using DataLoader2 since it's better suited for data pipes.
def demo_dataloader():
    datapipe = demo_datapipe()
    return DataLoader(
        dataset=datapipe, batch_size=5, num_workers=4, shuffle=True
    )



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