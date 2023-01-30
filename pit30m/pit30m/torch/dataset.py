
import os
import fire
import time
from typing import Sequence, Union, Optional
from uuid import UUID

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchdata.datapipes as dp
from PIL import Image

from pit30m.data.log_reader import LogReader
from pit30m.camera import CamName
from pit30m.data.submap import Map

# TODO(andrei): Configure shard-aware shuffling.

PIT30M_AWS_REGION = "us-east-1"


# As of 2023-01-22 I'm having trouble getting the (more efficient?) datapipe API working with S3. Getting credential
# errors even though the bucket is public.
#
# def logwise_s3_cam_lister(root: str, logs: Sequence[str], cameras: Optional[Sequence[str]] = None, region: str = PIT30M_AWS_REGION):
#     """Returns an iterator over all the camera files (images and metadata) in the given logs."""
#     if cameras is None:
#         cameras = [CamName.MIDDLE_FRONT_WIDE]

#     bases = [
#         os.path.join(root, log, "cameras", cam.value)
#         for log in logs
#         for cam in cameras
#     ]
#     print(bases)

#     import fsspec
#     fs = fsspec.filesystem("s3")
#     print(fs.ls(root))
#     # raise ValueError("OK")

#     import ipdb
#     ipdb.set_trace()

#     # NOTE(andrei): This will be VERY slow without caching: That is why we need indexing.
#     return dp.iter.S3FileLister(
#         dp.iter.IterableWrapper(bases),
#         region=region
#     )

# def by_webp(uri):
#     return uri.endswith(".webp")

# def img_as_np(file_and_path):
#     return np.array(Image.open(file_and_path[0]))

# def build_demo_datapipe(root_uri: str, logs: Union[str, Sequence[str]]):
#     if isinstance(logs, str):
#         logs = [entry.strip() for entry in logs.split(",")]

#     # TODO(andrei): Rewrite in functional form (recommended).
#     if root_uri.startswith("s3"):
#         uri_pipe = logwise_s3_cam_lister(root_uri, logs)
#         return uri_pipe
#         img_uri_pipe = uri_pipe.filter(lambda uri: uri.endswith(".webp"))
#         s3_data_pipe = dp.iter.S3FileLoader(img_uri_pipe, region=PIT30M_AWS_REGION)
#         img_data_pipe = s3_data_pipe.map(img_as_np)

#     else:
#         # hack
#         cameras = [CamName.MIDDLE_FRONT_WIDE]

#         uri_pipe = dp.iter.IterableWrapper([
#             os.path.join(root_uri, log, "cameras", cam.value)
#             for log in logs
#             for cam in cameras
#             # List image dirs, then list all images in each dir.
#         ]).list_files_by_fsspec().list_files_by_fsspec()

#         # We shuffle at the URI level since we can cache a huge pool of URIs, but obviously not the images themselves.
#         shuf_uri_pipe = dp.iter.Shuffler(uri_pipe, buffer_size=100000)

#         img_uri_pipe = shuf_uri_pipe.filter(by_webp)
#         local_data_pipe = dp.iter.FileOpener(img_uri_pipe)
#         img_data_pipe = local_data_pipe.map(img_as_np)

#     return img_data_pipe



class Pit30MLogDataset(Dataset):

    def __init__(self, root_uri: str, log_ids: Sequence[str], submap_utm_uri: str) -> None:
        """A low-level interface dataset for Pit30M operating on a per-log basis."""
        super().__init__()

        self._map = Map.from_submap_utm_uri(submap_utm_uri)
        self._log_readers = {
            log_id: LogReader(os.path.join(root_uri, str(log_id)), map=self._map) for log_id in log_ids
        }

        print("Loading indexes...")
        indexes = [
            (log_id, reader.get_cam_geo_index_utm(cam_name=CamName.MIDDLE_FRONT_WIDE))
            for log_id, reader in self._log_readers.items()
        ]

        print("Done loading indexes...")

    def __len__(self):
        pass

    def __getitem__(self, index: int) -> torch.Tensor:
        pass



class TripletPit30MDataset(Dataset):
    # TODO(andrei, julieta): Implement based on the official Pit30M benchmark split.
    pass


def demo_dataloader(root_uri: str = "s3://pit30m/", logs: Union[Sequence[str], str] = ["log1", "log2"],
                    batch_size: int = 32, num_workers: int = 8, max_batches: int = 25,
                    submap_utm_uri: str = "s3://pit30m/submap_utm.pkl"):
    if isinstance(logs, str):
        logs = [entry.strip() for entry in logs.split(",")]

    logs = [UUID(log) for log in logs]

    print("root_uri", root_uri)
    print("logs:", logs)
    dataset = Pit30MLogDataset(root_uri, logs, submap_utm_uri=submap_utm_uri)
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
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




if __name__ == "__main__":
    fire.Fire(demo_dataloader)