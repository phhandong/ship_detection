import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
import os


def load_transformations(file_path):
    data = np.loadtxt(file_path)
    timestamps = data[:, 0]
    quaternions = data[:, 1:5]
    translations = data[:, 5:8]

    # Normalize translations with respect to the first translation
    reference_translation = translations[0]  # Capture the first translation vector
    translations = (
        translations - reference_translation
    )  # Subtract to make relative to the first

    return timestamps, quaternions, translations


def interpolate_transformations(
    timestamps, quaternions, translations, target_timestamps
):
    timestamps *= 1e9  # Convert from seconds to nanoseconds

    # Initialize the rotations from the quaternions
    rotations = R.from_quat(quaternions)

    # Create a Slerp object for interpolation
    slerp = Slerp(timestamps, rotations)

    # Interpolate the rotations at target timestamps
    # interpolated_rotations = slerp(target_timestamps)
    interpolated_rotations = rotations
    interpolated_quats = interpolated_rotations.as_quat()

    # Linear interpolation for translations
    interp_x = interp1d(
        timestamps, translations[:, 0], kind="cubic", fill_value="extrapolate"
    )
    interp_y = interp1d(
        timestamps, translations[:, 1], kind="cubic", fill_value="extrapolate"
    )
    interp_z = interp1d(
        timestamps, translations[:, 2], kind="cubic", fill_value="extrapolate"
    )
    interpolated_trans = np.vstack(
        (
            interp_x(target_timestamps),
            interp_y(target_timestamps),
            interp_z(target_timestamps),
        )
    ).T

    return interpolated_quats, interpolated_trans


def transform_lidar_data(lidar_data, quats, trans):
    rotation = R.from_quat(quats)
    rotated_points = rotation.apply(lidar_data)
    transformed_points = rotated_points + trans
    return transformed_points




def load_and_transform_lidar_files(lidar_dir, timestamps, quats, trans):
    files = sorted(os.listdir(lidar_dir))[0:number_of_lidar_scans_to_transform]
    combined_points = np.empty((0, 3), dtype=np.float32)

    for file in files:
        file_timestamp = int(file.split(".")[0])
        idx = np.searchsorted(timestamps, file_timestamp)
        # lidar_data = np.fromfile(
        #     os.path.join(lidar_dir, file), dtype=np.float32
        # ).reshape(-1, 3)
        lidar_dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('time', np.uint32), ('reflectivity', np.uint16), ('ambient', np.uint16), ('range', np.uint32)]
        scan = np.fromfile(os.path.join(lidar_dir, file), dtype=lidar_dtype)
        lidar_data = np.stack((scan['x'], scan['y'], scan['z']), axis=-1)
        transformed_points = transform_lidar_data(lidar_data, quats[idx], trans[idx])
        combined_points = np.vstack((combined_points, transformed_points))
    print(np.shape(combined_points))

    return combined_points


def plot_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd], window_name="Combined LiDAR Point Cloud")


number_of_lidar_scans_to_transform = 10
path_to_lidar_files = r"E:\Python_Project\Ship_Detection\Pohang\poang00\lidar_starboard\points"

path_to_baseline = r"E:\Python_Project\Ship_Detection\Pohang\poang00\navigation\baseline.txt"

# Load transformation data
timestamps, quaternions, translations = load_transformations(path_to_baseline)

# Define LiDAR timestamps (based on filenames)
lidar_timestamps = np.array(
    [
        int(file.split(".bin")[0])
        for file in sorted(os.listdir(path_to_lidar_files))[
            0:number_of_lidar_scans_to_transform
        ]
    ]
)

# Interpolate transformations
interpolated_quats, interpolated_trans = interpolate_transformations(
    timestamps, quaternions, translations, lidar_timestamps
)


# Load, transform and combine LiDAR data
combined_lidar = load_and_transform_lidar_files(
    path_to_lidar_files, lidar_timestamps, interpolated_quats, interpolated_trans
)

# Visualize the result
plot_point_cloud(combined_lidar)
