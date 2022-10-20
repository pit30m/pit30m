
import torch
from torch.utils.data import Dataset


class LogReader:
    pass



class Pit30MDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()

        # TODO: Load UTMs so we can expose MRP in UTM.
        self._submap_utm = ...


    def __len__(self):
        pass

    def __getitem__(self, index: int) -> torch.Tensor:
        pass



class TripletPit30MDataset(Dataset):
    # TODO(andrei): Implement based on the official Pit30M benchmark split.
    pass