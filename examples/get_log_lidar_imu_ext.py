import os

import fsspec
import numpy as np
import yaml
from fire import Fire
from joblib.memory import Memory
from scipy.spatial.transform import Rotation as R

from pit30m.cli import Pit30MCLI
from pit30m.data.log_reader import LogReader
from pit30m.util import SafeLoaderIgnoreUnknown

# Cache S3 requests so we can iterate fast in the notebook.
memory = Memory(location="/tmp/joblib-pit30m", verbose=0)

@memory.cache()
def load_yaml_from_s3(uri: str) -> dict:
    with fsspec.open(uri, "r") as f:
        # Make sure we ignore "!binary" tags, which are not supported by the default loader.
        # They are used for embedding image data into the calibration file, e.g., for self-occlusion masks.
        return yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

def read_raw_calibration(log_reader):
    """Processes the mess that are 'raw_calibration.yml' files into something usable.

    The original files are raw dumps from a very legacy system, so their format is difficult to work with and contains
    a lot of irrelevant information. This is also why some of the keys are a bit strange, e.g.,
    'sensor_config::calib_info'.
    """
    log_root: str = log_reader._log_root_uri
    calib_uri: str = os.path.join(log_root, "raw_calibration.yml")

    raw_data = load_yaml_from_s3(calib_uri)

    # This may be useful for sensor simulation since it stores per-laser information, but for regular applications
    # we don't need to parse it. Deleting it makes is cleaner to, e.g., print these full dicts without too much
    # output.
    del raw_data["lidar"]["hdl64e_12_middle_front_roof"]["lidar_task_config::intrinsics_config"]
    # The aforementioned LiDAR occlusion mask.
    del raw_data["lidar"]["hdl64e_12_middle_front_roof"]["lidar_task_config::occlusion"]

    return raw_data

def parse_transforms_experimental(raw_calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    print("LiDAR")
    print("=" * 40)
    # There is always just one LiDAR
    velodyne_calib = raw_calibration["lidar"]["hdl64e_12_middle_front_roof"]
    # print(velodyne_calib.keys())
    extrinsics = velodyne_calib["sensor_config::extrinsics"]
    # This is t_veh_sensor, i.e. sensor-to-vehicle. This transform will take points from the LiDAR frame and express
    # them w.r.t. the vehicle.
    extrinsics_pose_dict = extrinsics["extrinsics_object::pose_offset"]
    xyz_rpy = np.array([
        float(extrinsics_pose_dict[0]["0"]),
        float(extrinsics_pose_dict[1]["1"]),
        float(extrinsics_pose_dict[2]["2"]),
        float(extrinsics_pose_dict[3]["3"]),
        float(extrinsics_pose_dict[4]["4"]),
        float(extrinsics_pose_dict[5]["5"]),
    ])
    print(f"t_veh_sensor expressed as xyz_rpy: {xyz_rpy}")
    t_veh_lidar = np.eye(4)
    # TODO(andrei): Double-check xyz vs. zyx order.
    EULER_CONVENTION = "xyz"
    t_veh_lidar[:3, :3] = R.from_euler(EULER_CONVENTION, xyz_rpy[3:], degrees=False).as_matrix()
    t_veh_lidar[:3, 3] = xyz_rpy[:3]

    # 'abs_encoder' = wheel encoder
    # 'adis' = IMU
    print()
    print("Position sensors")
    print("=" * 40)
    position_sensor_names = raw_calibration["position"].keys()
    for name in position_sensor_names:
        print(f"  {name}")

    # Just in case you need it
    # print("ABS encoder")
    # pprint(calibration["position"]["abs_encoder"])

    print()
    print("Analog Devices (Adis) IMU")
    print("-" * 40)
    imu_calib = raw_calibration["position"]["adis"]
    print(imu_calib.keys())
    bias_data = imu_calib["imu_task_config::calibration"]
    print(f"{bias_data.keys() = }")
    # Example
    print(f"{bias_data['imu_calibration_output::accel_bias'] = }")

    imu_ext = imu_calib["sensor_config::extrinsics"]
    # This will give you 't_veh_imu'
    imu_ext_pose_dict = imu_ext["extrinsics_object::pose_offset"]
    xyz_rpy = np.array([
        float(imu_ext_pose_dict[0]["0"]),
        float(imu_ext_pose_dict[1]["1"]),
        float(imu_ext_pose_dict[2]["2"]),
        float(imu_ext_pose_dict[3]["3"]),
        float(imu_ext_pose_dict[4]["4"]),
        float(imu_ext_pose_dict[5]["5"]),
    ])
    t_veh_imu = np.eye(4)
    t_veh_imu[:3, :3] = R.from_euler(EULER_CONVENTION, xyz_rpy[3:], degrees=False).as_matrix()
    t_veh_imu[:3, 3] = xyz_rpy[:3]

    return t_veh_lidar, t_veh_imu

def get_log_lidar_imu_ext(log_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Experimental code to get specific extrinsics from a log in Pit30M.

    May fail for the (fraction of) logs that don't have the expected structure and are missing general-purpose
    extrinsics.

    Will print the transforms twice because of how the 'fire' library works. The values are the same, just ignore
    the second copy as it's less readable.
    """
    cli = Pit30MCLI()
    all_logs = cli.all_log_ids

    sample_log_uuid = all_logs[log_idx]
    log_root_uri = os.path.join(cli._data_root, sample_log_uuid)
    log_reader = LogReader(log_root_uri=log_root_uri)


    raw_calibration = read_raw_calibration(log_reader)
    t_veh_lidar, t_veh_imu = parse_transforms_experimental(raw_calibration)
    print(f"{t_veh_lidar = }\n{t_veh_imu = }")

    return t_veh_lidar, t_veh_imu


if __name__ == "__main__":
    Fire(get_log_lidar_imu_ext)