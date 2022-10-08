
import fire
from typing import Tuple

import numpy as np
import json
import open3d as o3d
import fsspec
from urllib.parse import urlparse
import ipdb
import lzma
from tqdm import tqdm
import multiprocessing as mp


# Opposite of zip
def unzip(l):
    return list(zip(*l))

class Pit30MCLI:
    def __init__(self):
        pass

    def diagnose(self, dataset_base: str):
        # TODO(andrei): Check that the dataset is valid---all logs.
        pass

    def diagnose_log(self, dataset_base: str, log_uri: str) -> None:
        self._diagnose_lidar(dataset_base, log_uri)
        self._diagnose_misc(dataset_base, log_uri)

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
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            for lid_dir in fs.ls(ld):
                # sample = lid_dir + "/0032/003203.npz.xz"
                all_pcd = []
                all_int = []
                n_chunks = 99999
                # Sample every 'sample_rate' sweeps.
                sample_rate = 5
                lidar_chunks = sorted(fs.ls(lid_dir))
                print(f"{len(lidar_chunks)} subdirectories of LiDAR found")
                lidar_files = []
                for chunk in tqdm(lidar_chunks[:n_chunks]):
                    if fs.isdir(chunk):
                        lidar_files += sorted(fs.ls(chunk)[::sample_rate])

                print(f"Will load {len(lidar_files)} LiDAR pcds.")
                all_pcd, all_int = unzip(pool.map( _load, lidar_files))
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


    def hello(self, name: str):
        print(f"Hello {name}!")

def _load(lidar_uri: str) -> Tuple[np.ndarray, np.ndarray]:
    fs = fsspec.filesystem(urlparse(lidar_uri).scheme)
    with fs.open(lidar_uri) as f:
        with lzma.open(f) as lzma_file:
            lidar = np.load(lzma_file)
            return np.array(lidar["points"]), np.array(lidar["intensity"])


if __name__ == "__main__":
    fire.Fire(Pit30MCLI)