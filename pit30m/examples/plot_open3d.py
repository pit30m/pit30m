"""Example plotting using Open3d."""

import sys
import os

# Pretend we installed the package
pth = os.path.realpath(os.path.dirname(__file__) + "/../")
sys.path.append(pth)

try:
    import open3d as o3d
except ImportError:
    print("Warning: Open3d is required for this example, but not installed.\n"
        "Install with `pip install open3d`.")
    print(f"Current Python: {sys.executable}")
    exit(0)

import fire
from pit30m.data.log_reader import LogReader


def np_to_o3d_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def open3d_example(log_root_uri: str):
    lr = LogReader(log_root_uri)
    lidar = next(lr.lidar_iterator())

    o3d.visualization.draw_geometries([np_to_o3d_pcd(lidar.xyz_continuous)])



if __name__ == "__main__":
    fire.Fire(open3d_example)