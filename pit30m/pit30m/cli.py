
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

    def diagnose(self, dataset_base: str):
        # TODO(andrei): Check that the dataset is valid---all logs.
        pass

    def diagnose_log(self, dataset_base: str, log_uri: str) -> None:
        self._diagnose_lidar(dataset_base, log_uri)
        self._diagnose_misc(dataset_base, log_uri)

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


    def _diagnose_misc(self, dataset_base: str, log_uri: str) -> None:
        """Loads and prints misc data, hopefully raising errors if something is corrupted."""
        dbp = urlparse(dataset_base)
        fs = fsspec.filesystem(dbp.scheme)

        # Active district information is not very important
        with lzma.open(dataset_base + log_uri + "/active_district.npz.xz", "rb") as f:
            ad = np.load(f)["data"]
            assert isinstance(ad, np.ndarray)
            print("Active district OK:", ad.shape, ad.dtype)

        # TODO(andrei): We probably want to provide poses as something uncompressed, since LZMA decoding is a few
        # seconds per log. For training we will cache this in some index files, but in general it's annoying to wait
        # 4-5 seconds on a modern PC to read 50MiB of data...
        with lzma.open(dataset_base + log_uri + "/all_poses.npz.xz", "rb") as f:
            # Poses contain just the data as a structured numpy array. Each pose object contains a continuous, smooth,
            # log-specific pose, and a map-relative pose. The map relative pose (MRP) takes vehicle points into the
            # current submap, whose ID is indicated by the 'submap' field of the MRP.
            #
            # The data also has pose and velocity covariance information, but I have never used directly so I don't know
            # if it's well-calibrated.
            #
            # Be sure to check the 'valid' flag of the poses before using them!
            #
            # Poses are provided at 100Hz.
            all_poses = np.load(f)["data"]
            print(len(all_poses), "poses read")
            print("All poses read OK")

        # TODO(andrei): Support parsing this pseudo YAML. Right now it's not very important but may be useful if people
        # want raw wheel encoder data, steering angle, etc.
        # with lzma.open(dataset_base + log_uri + "/all_vehicle_data.npz.xz", "rb") as f:
        #     all_vd = np.load(f)
        #     print(list(all_vd.keys()))
        #     ipdb.set_trace()
        #     print("All vd OK")

        # NOTE(andrei): The metadata won't be super useful to end users, since it's mostly internal stuff like a list
        # of active sensors, calibration, etc., which is already obvious by looking at the data files.
        with open(dataset_base + log_uri + "/log_metadata.json", "r") as f:
            meta = json.load(f)
            print("Metadata OK")


        with fs.open(dataset_base + log_uri + "/wgs84.npz.xz", "rb") as lzma_f:
            with lzma.open(lzma_f) as f:
                # WGS84 poses with timestamp, long, lat, alt, heading, pitch, roll.
                #
                # ~10Hz
                # TODO(andrei): Check whether these are inferred from MRP or not. I *think* for logs where localization is
                # OK (i.e., most logs -- and we can always check pose validity flags!) these are properly post-processed,
                # and not just biased filtered GPS, but I need to check. Plotting this with leaflet will make it obvious.
                all_wgs84 = np.load(f)["data"]
                print(f"{len(all_wgs84) = }")
                print("All WGS84 poses OK")



    def _diagnose_lidar(self, dataset_base: str, log_uri: str) -> None:
        ld = dataset_base + log_uri + "/lidars/"
        dbp = urlparse(dataset_base)
        fs = fsspec.filesystem(dbp.scheme)

        # TODO(andrei): Diagnose and check shape!

        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            for lid_dir in fs.ls(ld):
                all_pcd = []
                all_int = []
                n_chunks = 20000
                # Sample every 'sample_rate' sweeps.
                sample_rate = 5
                lidar_chunks = sorted(fs.ls(lid_dir))
                print(f"{len(lidar_chunks)} subdirectories of LiDAR found")
                lidar_files = []
                for chunk in tqdm(lidar_chunks[:n_chunks]):
                    if fs.isdir(chunk):
                        lidar_files += sorted(fs.ls(chunk)[::sample_rate])

                print(f"Will load {len(lidar_files)} LiDAR pcds.")
                all_pcd, all_int = unzip(pool.map(_load_lidar, lidar_files))
                print(all_pcd[0].shape)
                print(all_pcd[0].dtype)
                # for sub_sample in tqdm(
                #     with lzma.open(sub_sample) as lzma_file:
                #         lidar = np.load(lzma_file)
                #         # Points are in the continuous frame I think
                #         all_pcd.append(np.array(lidar["points"]))
                #         all_int.append(np.array(lidar["intensity"]))
                #         # other keys: laser_theta, seconds, raw_power, intensity, points, points_H_sensor, laser_id

                print("Loaded points, processing...")
                all_pcd = np.concatenate(all_pcd, axis=0)
                all_int = np.concatenate(all_int, axis=0)
                # TODO(andrei): This likely makes a copy (based on how slow it is), so we may want to use the tensor
                # API?
                vec = o3d.utility.Vector3dVector(all_pcd)
                col = o3d.utility.Vector3dVector(np.tile(all_int[:, None] / 255.0, (1, 3)))
                pcd = o3d.geometry.PointCloud(vec)
                pcd.colors = col
                print("Before filtering:", pcd)
                pcd = pcd.voxel_down_sample(voxel_size=0.05)
                print("After filtering:", pcd)
                o3d.visualization.draw_geometries([pcd])


def _load_lidar(lidar_uri: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a single LiDAR sweep from a URI, reading a LZMA-compressed file."""
    fs = fsspec.filesystem(urlparse(lidar_uri).scheme)
    with fs.open(lidar_uri) as f:
        try:
            with lzma.open(f) as lzma_file:
                lidar = np.load(lzma_file)
                return np.array(lidar["points"]), np.array(lidar["intensity"])
        except LZMAError:
            print("LZMA error reading LiDAR uri: ", lidar_uri)
            raise


if __name__ == "__main__":
    fire.Fire(Pit30MCLI)