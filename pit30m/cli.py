import json
import multiprocessing as mp
import os
import shlex
import subprocess
import tempfile
from functools import cached_property
from typing import Iterable, Optional, Union
from urllib.parse import urlparse

import fire
import fsspec
from joblib import Parallel, delayed

from pit30m.data.submap_indexing import compute_submap_index


class Pit30MCLI:
    def __init__(
        self,
        data_root: str = "s3://pit30m/",
        log_list_fpath: str = os.path.join(os.path.dirname(__file__), "all_logs.txt"),
    ):
        self._data_root = data_root
        self._read_pool = Parallel(n_jobs=mp.cpu_count() * 4)
        self._log_list_fpath = log_list_fpath

    @cached_property
    def all_log_ids(self) -> list[str]:
        """Return a list of all log IDs in the dataset."""
        with open(self._log_list_fpath, "r") as f:
            return [line.strip() for line in f]

    def compute_submap_index(self, out_fpath: str, log_ids: Optional[list[str]] = None, mrp_subsample: int = 10, max_jobs: int = 0) -> None:
        """Returns a submap-to-log index.

        For each unique submap UUID we get a list of log chunks which indicate approximately when that log entered, then
        exited that submap. A log may re-enter a submap multiple times if it loops around.

        A record of entering / existing a submap is a tuple array containing: UNIX time, x, y, z - coordinates are
        relative to the submap's origin.

        You can then use this index to quickly find all log chunks which go through a specific submap, dump their full
        poses, LiDAR, images, etc.

        One a 24 core machine with a 1Gbps connection to S3, indexing all logs with the default 10 subsampling factor
        will take under 10 minutes. The code is parallelized so faster machine (e.g., a cloud workstation) will scale
        accordingly.

        Args:
            out_fpath:      Output JSON file path for the index.
            log_ids:        List of log IDs to process. Pass None (default) to use all of them.
            mrp_subsample:  Subsampling factor for map-relative poses. 1 means no subsampling, 2 means every other pose,
                            etc. We recommend 10, which means downsampling 100Hz data to 10Hz.
            max_jobs:       Maximum number of parallel jobs to run. Pass 0 (default) to use all available cores,
                            multiplied by a scaling factor to ensure the network is saturated.
        """
        out_dir = os.path.dirname(out_fpath)
        if not os.path.isdir(out_dir):
            raise ValueError(f"Output directory {out_dir} does not exist.")

        index = compute_submap_index(log_ids or self.all_log_ids, mrp_subsample, max_jobs)
        with open(out_fpath, "w") as f:
            json.dump(index, f, indent=4)

        print(f"Saved submap index to {out_fpath}")


    def multicam_demo(
        self,
        out_dir: str,
        log_id: str = "e9511854-f657-47bd-c9d3-047187cfc663",
        chunks: Union[int, Iterable[int]] = (17, 18),
    ) -> None:
        """Bakes a multicam video from a log URI. Requires `ffmpeg` to be installed.

        chunks = 100-image chunks of log to include

        TODO(andrei): Port to use the log reader.
        """
        in_fs = fsspec.filesystem(urlparse(self._data_root).scheme, anon=True)
        out_fs = fsspec.filesystem(urlparse(out_dir).scheme)
        if isinstance(chunks, int):
            chunks = (chunks,)

        # Clockwise wide camera names, with the front wide in the middle
        cams_clockwise = [
            "hdcam_08_port_rear_roof_wide",
            "hdcam_10_port_front_roof_wide",
            "hdcam_12_middle_front_roof_wide",
            "hdcam_02_starboard_front_roof_wide",
            "hdcam_04_starboard_rear_roof_wide",
        ]

        video_fpaths = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for cam in cams_clockwise:
                ld = self._data_root + log_id + "/cameras/" + cam

                sample_img_uris = [
                    entry
                    for chunk in list(chunks)
                    for entry in in_fs.ls(os.path.join(ld, f"{chunk:04d}"))
                    if entry.endswith("webp")
                ]

                def copy_img(img_uri, out_dir):
                    with in_fs.open(img_uri, "rb") as f:
                        with out_fs.open(out_dir + "/" + img_uri.split("/")[-1], "wb") as out_f:
                            out_f.write(f.read())

                out_img_dir = tmp_dir + "/frames/" + cam
                if not out_fs.exists(out_img_dir):
                    out_fs.mkdir(out_img_dir)
                print("ETL for", cam)
                self._read_pool(delayed(copy_img)(img_uri, out_img_dir) for img_uri in sample_img_uris)

                framerate = 10
                subprocess.run(
                    shlex.split(
                        f"ffmpeg -framerate {framerate} -i {out_img_dir}/%*.day.webp -crf 20 -pix_fmt yuv420p "
                        f"-filter:v 'scale=-1:800' {out_img_dir}.mp4"
                    )
                )
                video_fpaths.append(f"{out_img_dir}.mp4")

            # Stack the resulting videos horizontally with ffmpeg
            print(video_fpaths)
            out_fpath = f"{out_dir}/{log_id}-sample-multicam.mp4"
            subprocess.run(
                shlex.split(
                    f"ffmpeg -i {' -i '.join(video_fpaths)} -filter_complex hstack=inputs={len(video_fpaths)} "
                    f"{out_fpath}"
                )
            )
            print("Generated video to:", out_fpath)


# Opposite of zip
def unzip(l):
    return list(zip(*l))


if __name__ == "__main__":
    # Ensures we can use multiprocessing and fsspec together.
    mp.set_start_method("forkserver")
    fire.Fire(Pit30MCLI)
