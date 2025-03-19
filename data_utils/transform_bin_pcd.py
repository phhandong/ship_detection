import numpy as np
import open3d as o3d
import struct

def bin_to_pcd(bin_file, pcd_file):
    """
    将 .bin 格式的点云文件转换为 .pcd 格式。

    参数:
    - bin_file: 输入的 .bin 文件路径。
    - pcd_file: 输出的 .pcd 文件路径。
    """
    # 读取 .bin 文件
    pointcloud = []
    with open(bin_file, 'rb') as f:
        while True:
            data = f.read(16)  # 每个点占 16 字节 (4 * float32)
            if not data:
                break
            x, y, z, intensity = struct.unpack('ffff', data)
            pointcloud.append([x, y, z, intensity])

    # 转换为 numpy 数组
    pointcloud = np.array(pointcloud, dtype=np.float32)

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])  # 只使用 x, y, z 坐标
    pcd.colors = o3d.utility.Vector3dVector(np.tile(pointcloud[:, 3:4], (1, 3)))  # 使用强度作为颜色

    # 保存为 .pcd 文件
    o3d.io.write_point_cloud(pcd_file, pcd)
    print(f"Saved .pcd file to {pcd_file}")

# 示例调用
bin_file = r"E:\Python_Project\Ship_Detection\Pohang\poang00\lidar_starboard\points\1625124349327406376.bin"  # 替换为实际的 .bin 文件路径
pcd_file = r"E:\Python_Project\Ship_Detection\Pohang\poang00\lidar_starboard\pcd\1625124349327406376.pcd"  # 替换为实际的 .pcd 文件路径
bin_to_pcd(bin_file, pcd_file)