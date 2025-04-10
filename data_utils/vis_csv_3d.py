import open3d as o3d
import pandas as pd
import numpy as np
import dpkt

def parse_pcap(pcap_path):
    """
    使用 dpkt 解析 .pcap 文件中的点云数据。

    参数:
    - pcap_path: .pcap 文件的路径。

    返回:
    - point_clouds: 一个包含所有帧点云的列表，每帧是一个 (N, 3) 的 NumPy 数组。
    """
    point_clouds = []

    with open(pcap_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)

        for timestamp, buf in pcap:
            # 解析以太网帧
            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue

            # 解析 IP 数据包
            ip = eth.data
            if not isinstance(ip.data, dpkt.udp.UDP):
                continue

            # 解析 UDP 数据包
            udp = ip.data
            data = udp.data

            # 根据 LiDAR 数据格式解析点云数据
            # 假设每个点由 12 字节组成 (x, y, z)，每个坐标为 4 字节浮点数
            points = np.frombuffer(data, dtype=np.float16).reshape(-1, 3)
            point_clouds.append(points)

    return point_clouds


def visualize_point_cloud(point_cloud):
    """
    使用 Open3D 可视化点云数据。

    参数:
    - point_cloud: 点云数据，形状为 (N, 3)。
    """
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])
    
if __name__ == '__main__':
    # 设置 .pcap 文件路径
    pcap_path = r'E:\Python_Project\Ship_Detection\dataset\hand_craft\202410_jimiya\24-10-23-Jimiya-RS128\piece\2024-10-23-14-32-10-RS-128-Data (Frame 100 to 150).pcap'

    point_clouds = parse_pcap(pcap_path)

    # 可视化第一帧点云
    if len(point_clouds) > 0:
        print(f"Loaded {len(point_clouds)} frames from the .pcap file.")
        visualize_point_cloud(point_clouds[0])
    else:
        print("No point cloud data found in the .pcap file.")
    