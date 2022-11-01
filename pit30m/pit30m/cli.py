
import shutil
import fire
import shlex
from typing import Tuple
from lzma import LZMAError

import numpy as np
import json
import open3d as o3d
import subprocess
import fsspec
from urllib.parse import urlparse
import tempfile
import ipdb
import lzma
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
from PIL import Image

from pit30m.camera import CAM_NAMES


# Opposite of zip
def unzip(l):
    return list(zip(*l))

class Pit30MCLI:
    def __init__(self):
        self._read_pool = Parallel(n_jobs=mp.cpu_count())
        pass

    def multicam_demo(self, dataset_base: str, log_uri: str, out_dir: str) -> None:
        in_fs = fsspec.filesystem(urlparse(dataset_base).scheme)
        out_fs = fsspec.filesystem(urlparse(out_dir).scheme)

        # Clockwise camera names, with the front wide in the middle
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
                ld = dataset_base + log_uri + "/cameras/" + cam
                chunks = sorted(in_fs.ls(ld))

                sample_img_uris = [
                    s_uri
                    for s_uri in in_fs.ls(chunks[17]) + in_fs.ls(chunks[18])
                    if s_uri.endswith("webp")
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
                    shlex.split(f"ffmpeg -framerate {framerate} -i {out_img_dir}/%*.day.webp -crf 20 -pix_fmt yuv420p " \
                        f"-filter:v 'scale=-1:800' {out_img_dir}.mp4")
                )
                video_fpaths.append(f"{out_img_dir}.mp4")

            # Stack the resulting videos horizontally with ffmpeg
            print(video_fpaths)
            subprocess.run(
                shlex.split(f"ffmpeg -i {' -i '.join(video_fpaths)} -filter_complex hstack=inputs={len(video_fpaths)} " \
                    f"{out_dir}/{log_uri}-sample-multicam.mp4")
            )


if __name__ == "__main__":
    fire.Fire(Pit30MCLI)