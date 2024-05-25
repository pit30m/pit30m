"""(Deprecated, WIP) Example plotting using Open3d."""

# mypy: ignore-errors

import multiprocessing as mp
import os
import sys
from urllib.parse import urlparse

import fire
import fsspec
import numpy as np
from tqdm import tqdm

# Pretend we installed the pit30m package
pth = os.path.realpath(os.path.dirname(__file__) + "/../")
sys.path.append(pth)

# This example needs Open3D, but it's not a dependency.
try:
    import open3d as o3d
except ImportError:
    print("Warning: Open3d is required for this example, but not installed.\n" "Install with `pip install open3d`.")
    print(f"Current Python: {sys.executable}")
    exit(0)


from pit30m.data.log_reader import LogReader


def unzip(l):
    return list(zip(*l))


def np_to_o3d_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def open3d_example(log_root_uri: str):
    log_reader = LogReader(log_root_uri)
    lidar = next(log_reader.lidar_iterator())

    o3d.visualization.draw_geometries([np_to_o3d_pcd(lidar.xyz_continuous)])


def open3d_many_sweeps_example(self, log_root_uri: str) -> None:
    # Old code for loading a lot of LiDAR into the same coordinate frame for Open3D visualization.
    # ld = dataset_base + log_uri + "/lidars/"
    # dbp = urlparse(dataset_base)

    _ = LogReader(log_root_uri)  # TODO: update this
    fs = fsspec.filesystem(urlparse(log_root_uri).scheme, anon=True)

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


# TODO(andrei): Remove this since we can just load LiDAR with the LogReader.
#
# def _load_lidar(lidar_uri: str) -> Tuple[np.ndarray, np.ndarray]:
#     """Loads a single LiDAR sweep from a URI, reading a LZMA-compressed file."""
#     fs = fsspec.filesystem(urlparse(lidar_uri).scheme)
#     with fs.open(lidar_uri) as f:
#         try:
#             with lzma.open(f) as lzma_file:
#                 lidar = np.load(lzma_file)
#                 return np.array(lidar["points"]), np.array(lidar["intensity"])
#         except LZMAError:
#             print("LZMA error reading LiDAR uri: ", lidar_uri)
#             raise


if __name__ == "__main__":
    fire.Fire(open3d_example)
