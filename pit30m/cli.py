import multiprocessing as mp
import os
import shlex
import subprocess
import tempfile
from functools import cached_property
from typing import Union
from urllib.parse import urlparse

import fire
import fsspec
from joblib import Parallel, delayed


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

    def multicam_demo(
        self,
        out_dir: str,
        log_id: str = "e9511854-f657-47bd-c9d3-047187cfc663",
        chunks: Union[int, tuple[int]] = (17, 18),
    ) -> None:
        """Bakes a multicam video from a log URI. Requires `ffmpeg` to be installed.

        chunks = 100-image chunks of log to include

        TODO(andrei): Port to use the log reader.
        """
        in_fs = fsspec.filesystem(urlparse(self._data_root).scheme, anon=True)
        out_fs = fsspec.filesystem(urlparse(out_dir).scheme)
        if isinstance(chunks, int):
            chunks = chunks

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
                    for chunk in chunks
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
                res = self._read_pool(delayed(copy_img)(img_uri, out_img_dir) for img_uri in sample_img_uris)

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
