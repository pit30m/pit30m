
import os
import fire
import time
from typing import Sequence, Union, Optional

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchdata.datapipes as dp
from PIL import Image

from pit30m.data.log_reader import LogReader
from pit30m.camera import CamName

# TODO(andrei): Configure shard-aware shuffling.

PIT30M_AWS_REGION = "us-east-1"



def logwise_s3_cam_lister(root: str, logs: Sequence[str], cameras: Optional[Sequence[str]] = None, region: str = PIT30M_AWS_REGION):
    """Returns an iterator over all the camera files (images and metadata) in the given logs."""
    if cameras is None:
        cameras = [CamName.MIDDLE_FRONT_WIDE]

    bases = [
        os.path.join(root, log, "cameras", cam.value)
        for log in logs
        for cam in cameras
    ]
    print(bases)

    import fsspec
    fs = fsspec.filesystem("s3")
    print(fs.ls(root))
    # raise ValueError("OK")

    import ipdb
    ipdb.set_trace()

    # NOTE(andrei): This will be VERY slow without caching: That is why we need indexing.
    return dp.iter.S3FileLister(
        dp.iter.IterableWrapper(bases),
        region=region
    )

def by_webp(uri):
    return uri.endswith(".webp")

def img_as_np(file_and_path):
    return np.array(Image.open(file_and_path[0]))

def build_demo_datapipe(root_uri: str, logs: Union[str, Sequence[str]]):
    if isinstance(logs, str):
        logs = [entry.strip() for entry in logs.split(",")]

    # TODO(andrei): Rewrite in functional form (recommended).
    if root_uri.startswith("s3"):
        uri_pipe = logwise_s3_cam_lister(root_uri, logs)
        return uri_pipe
        img_uri_pipe = uri_pipe.filter(lambda uri: uri.endswith(".webp"))
        s3_data_pipe = dp.iter.S3FileLoader(img_uri_pipe, region=PIT30M_AWS_REGION)
        img_data_pipe = s3_data_pipe.map(img_as_np)

    else:
        # hack
        cameras = [CamName.MIDDLE_FRONT_WIDE]

        uri_pipe = dp.iter.IterableWrapper([
            os.path.join(root_uri, log, "cameras", cam.value)
            for log in logs
            for cam in cameras
            # List image dirs, then list all images in each dir.
        ]).list_files_by_fsspec().list_files_by_fsspec()

        # We shuffle at the URI level since we can cache a huge pool of URIs, but obviously not the images themselves.
        shuf_uri_pipe = dp.iter.Shuffler(uri_pipe, buffer_size=100000)

        img_uri_pipe = shuf_uri_pipe.filter(by_webp)
        local_data_pipe = dp.iter.FileOpener(img_uri_pipe)
        img_data_pipe = local_data_pipe.map(img_as_np)

    return img_data_pipe



# TODO(andrei): Explore using DataLoader2 since it's better suited for data pipes.
def demo_dataloader(root_uri: str = "s3://my/bucket", logs: Union[Sequence[str], str] = ["log1", "log2"],
                    batch_size: int = 32, num_workers: int = 8, max_batches: int = 25):
    print("root_uri", root_uri)
    print("logs:", logs)

    datapipe = build_demo_datapipe(root_uri, logs)
    loader = DataLoader(
        dataset=datapipe, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        print(len(batch))

        if (batch_idx + 1) >= max_batches:
            break

    dt_s = time.time() - start_time
    n_images = batch_size * max_batches
    print(f"Loaded {n_images} images in {dt_s:.2f} s, {n_images / dt_s:.2f} images/s.")
    print(f"bench,{num_workers},{batch_size},{max_batches},{n_images},{dt_s:.2f},{n_images / dt_s:.2f}")
    print("Done")



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



if __name__ == "__main__":
    fire.Fire(demo_dataloader)