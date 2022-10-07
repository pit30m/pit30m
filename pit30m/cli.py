
import fire

import numpy as np
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
        ld = dataset_base + log_uri + "/lidars/"
        dbp = urlparse(dataset_base)
        fs = fsspec.filesystem(dbp.scheme)



        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            for lid_dir in fs.ls(ld):
                # sample = lid_dir + "/0032/003203.npz.xz"
                all_pcd = []
                all_int = []

                all_pcd, all_int = unzip(
                    pool.map(
                        _load,
                        sorted(fs.ls(lid_dir + "/0032/") + fs.ls(lid_dir + "/0033/") + fs.ls(lid_dir + "/0034/")
                        + fs.ls(lid_dir + "/0035/") + fs.ls(lid_dir + "/0036/") + fs.ls(lid_dir + "/0037/"))
                    )
                )
                # for sub_sample in tqdm(
                #     with lzma.open(sub_sample) as lzma_file:
                #         lidar = np.load(lzma_file)
                #         # Points are in the continuous frame I think
                #         all_pcd.append(np.array(lidar["points"]))
                #         all_int.append(np.array(lidar["intensity"]))
                #         # other keys: laser_theta, seconds, raw_power, intensity, points, points_H_sensor, laser_id

                all_pcd = np.concatenate(all_pcd, axis=0)
                all_int = np.concatenate(all_int, axis=0)
                vec = o3d.utility.Vector3dVector(all_pcd)
                col = o3d.utility.Vector3dVector(np.tile(all_int[:, None] / 255.0, (1, 3)))
                pcd = o3d.geometry.PointCloud(vec)
                pcd.colors = col
                print("Before filtering:", pcd)
                print(pcd)
                pcd = pcd.voxel_down_sample(voxel_size=0.025)
                print("After filtering:", pcd)
                print(pcd)
                o3d.visualization.draw_geometries([pcd])

            # fs.ls(lid_dir)


    def hello(self, name: str):
        print(f"Hello {name}!")

def _load(path):
    fs = fsspec.filesystem(urlparse(path).scheme)
    with fs.open(path) as f:
        with lzma.open(f) as lzma_file:
            lidar = np.load(lzma_file)
            return np.array(lidar["points"]), np.array(lidar["intensity"])


if __name__ == "__main__":
    fire.Fire(Pit30MCLI)