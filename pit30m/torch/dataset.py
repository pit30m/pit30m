import os
import time
from functools import cached_property
from typing import Sequence, Union
from uuid import UUID

import fire
import numpy as np

try:
    import torch.multiprocessing as mp
    from torch.utils.data import DataLoader, Dataset
except ImportError as err:
    raise ImportError(
        "PyTorch is required to import PyTorch Pit30M functionality. Please install PyTorch and try again."
    ) from err


from pit30m.camera import CamName
from pit30m.data.log_reader import CameraImage, LogReader
from pit30m.data.submap import Map
from pit30m.indexing import CAM_INDEX_V0_0_DTYPE


class Pit30MLogDataset(Dataset):
    def __init__(
        self, log_ids: Sequence[str], cam_name: CamName = CamName.MIDDLE_FRONT_WIDE, root_uri: str = "s3://pit30m/"
    ) -> None:
        """A low-level interface dataset for Pit30M operating on a per-log single-sensor basis.

        Somewhat inefficient due to limitations in PyTorch when it comes to high-throughput high-latency data sources,
        please see [@svogor2022profiling] for a related analysis.

        References:
            - [@svogor2022profiling]: https://arxiv.org/abs/2211.04908
        """
        super().__init__()

        self._cam_name = cam_name
        self._root_uri = root_uri
        self._log_ids = log_ids

    @cached_property
    def _map(self):
        return Map()

    @cached_property
    def _log_readers(self):
        return {log_id: LogReader(os.path.join(self._root_uri, str(log_id)), map=self._map) for log_id in self._log_ids}

    @cached_property
    def _lengths(self):
        return [len(index) for _, index in self._indexes]

    @cached_property
    def _indexes(self):
        return [
            (log_id, reader.get_cam_geo_index(cam_name=self._cam_name)) for log_id, reader in self._log_readers.items()
        ]

    @cached_property
    def _len_cdf(self):
        return np.cumsum(self._lengths)

    @cached_property
    def n_samples(self) -> int:
        return sum(self._lengths)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[np.ndarray, tuple, tuple]:
        """Returns the image, its metadata and the MRP at the given index."""
        # print(f"Getting item {index}...")
        log_idx = np.searchsorted(self._len_cdf, index, side="right")
        idx_in_log = index - self._len_cdf[log_idx] if log_idx > 0 else index
        cur_log_id, cur_index = self._indexes[log_idx]

        # Refer to the dtype definition for more information on what is available.
        cur_sample: np.ndarray = cur_index[idx_in_log]
        assert cur_sample.dtype == CAM_INDEX_V0_0_DTYPE

        image: CameraImage = self._log_readers[cur_log_id].get_image(self._cam_name, idx_in_log)

        image_metadata = (
            cur_sample["mrp_present"],
            cur_sample["mrp_valid"],
            cur_sample["mrp_time"],
            cur_sample["img_time"],
        )

        mrp_xyz_rpw = (
            cur_sample["mrp_x"],
            cur_sample["mrp_y"],
            cur_sample["mrp_z"],
            cur_sample["mrp_roll"],
            cur_sample["mrp_pitch"],
            cur_sample["mrp_yaw"],
        )

        return image.image, image_metadata, mrp_xyz_rpw


class TripletPit30MDataset(Dataset):
    # TODO(andrei, julieta): Implement based on the official Pit30M benchmark split.
    pass


def demo_dataloader(
    root_uri: str = "s3://pit30m/",
    logs: Union[Sequence[str], str] = ["log1", "log2"],
    batch_size: int = 32,
    num_workers: int = 8,
    max_batches: int = 25,
):
    # Failure to do this may cause fsspec to hang in workers
    mp.set_start_method("forkserver")

    if isinstance(logs, str):
        logs = [entry.strip() for entry in logs.split(",")]

    print("root_uri", root_uri)
    print("logs:", logs)
    dataset = Pit30MLogDataset(logs, root_uri=root_uri)
    loader = DataLoader(
        # No shuffle for demo/benchmarking purposes.
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        # prefetch_factor=8 if num_workers > 0 else None,
        pin_memory=True,
        drop_last=False,
    )
    start_time = time.time()
    for batch_idx, (image, metadata, pose_6dof) in enumerate(loader):
        print(f"Batch {batch_idx + 1} of {len(loader)}; image shape: {image.shape}")
        if (batch_idx + 1) >= max_batches:
            break

    dt_s = time.time() - start_time
    n_images = batch_size * max_batches
    print(f"Loaded {n_images} images in {dt_s:.2f} s, {n_images / dt_s:.2f} images/s.")
    print(f"bench,{num_workers},{batch_size},{max_batches},{n_images},{dt_s:.2f},{n_images / dt_s:.2f}")
    print("Done")


if __name__ == "__main__":
    fire.Fire(demo_dataloader)
