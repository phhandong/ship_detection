import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
import open3d as o3d
from data_utils.data_utils_pohang import DataGenerator


class Fusion():
    def __init__(self):
        pass
    
    def draw_box(self,ori_img,box,img_name,output_dir):
        img_cp = ori_img.copy()
        
    def points_cluster(points, eps = 0.5, depth_facter=20):
        min_samples = max(5,int(len(points)*0.1))
        depth = np.sqt(points[:, 0] ** 2 + points[:, 1] **2)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(depth.reshape(-1,1))
        
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        cluster_counts_dict = dict(zip(unique_clusters,cluster_counts))
        
        sorted_clusters = sorted(cluster_counts_dict, key=cluster_counts_dict.get, reverse=True)
        
        top_cluster = sorted_clusters[:1]
        
        depth_ranges = {}
        
        for cluster_label in top_cluster:
            if cluster_label == -1:
                continue
            cluster_depth_values = depth[clusters == cluster_label]
            min_depth = np.min(cluster_depth_values)
            max_depth = np.max(cluster_depth_values)
            range_depth = max_depth - min_depth
            min_depth_adjusted = min_depth + (1 - depth_facter) * range_depth / 2
            max_depth_adjusted = max_depth - (1 - depth_facter) * range_depth / 2
            depth_ranges[cluster_label] = (min_depth_adjusted, max_depth_adjusted)
        
        filtered_points_of_object = []
        for cluster_label, depth_range in depth_ranges.items():
            min_depth,max_depth = depth_range
            valid_cluster_mask = clusters == cluster_label
            valid_cluster_indices = np.where(valid_cluster_mask)[0]
            cluster_points = points[valid_cluster_indices]
            depth_mask = (cluster_points[:, 0] >= min_depth) & (cluster_points[:, 0] <= max_depth)
            filtered_points_of_object.extend(cluster_points[depth_mask])
        
        return np.array(filtered_points_of_object)
        
        
        
        
if __name__ == '__main__':
    base_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00'
    data_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00'
    img_dir = os.path.join(data_dir,r'stereo\left_images')
    # img_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\stereo\left_images'
    xband_dir = os.path.join(data_dir,'X-band')
    lidar_dir = os.path.join(data_dir,r'lidar_front\points')
    calib_dir = os.path.join(base_dir,'calibration')
    label_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\labels\all\left'
    dynamic_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\labels\dynamic\00_dynamic_lidar_label'
    data_loader = DataGenerator(data_dir, img_dir, xband_dir,lidar_dir,calib_dir,label_dir,dynamic_dir)
    point_cloud = data_loader.load_pcd_data(r'ship\id_05\008176.pcd')
    data_loader.visualize_lidar(point_cloud)