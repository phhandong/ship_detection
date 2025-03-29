import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
import open3d as o3d
from data_utils.data_utils_pohang import DataGenerator
from ultralytics import YOLO


class Fusion():
    def __init__(self, dataloader, tracking=False, PCA=False):
        self.dataloader = dataloader
        self.pca = PCA
        self.tracking = tracking
        self.last_ground_center_of_id = {}
        self.output_dir = r'E:\Python_Project\Ship_Detection\yolo8\results\wo_train_v8' 
        
    def fuse(self,img,cloud_points,result,img_name=None,lidar_name=None,erosion_factor = 25, depth_factor= 10):
        all_corner,all_points,cam_calib, lidar_calib = self.process(img,cloud_points,result,erosion_factor,depth_factor)
        for corner,point in zip(all_corner,all_points):
            img = self.draw_box(img,corner,(cam_calib,lidar_calib))
            
        # cv2.imwrite(os.path.join(self.output_dir,img_name),img)
        return img
        
        
    def process(self,img,cloud_points,result,erosion_factor=25,depth_factor=10):
        cam_calib, lidar_calib = self.dataloader.load_calibration()
        cam_K, cam_R, cam_T, cam_dis = cam_calib
        lidar_R, lidar_T = lidar_calib
        lidar2camera = cam_R.T @ lidar_R
        boxes = result.boxes
        masks = result.masks
            
        pts_3d,pts_2d = self.dataloader.project_pointcloud_lidar_to_image(cloud_points[:,:3],img, cam_K, cam_R, cam_T, cam_dis, lidar_R, lidar_T, mask=masks.xy, just_project=True)
        all_corners_3D = []
        all_filtered_points_of_object = []
        all_object_IDs = []
        objects3d_data = []
        for j, cls in enumerate(boxes.cls.tolist()):
            conf = boxes.conf.tolist()[j] if boxes.conf is not None else None
            box_id = int(boxes.id.tolist()[j]) if boxes.id is not None else None

            all_object_IDs.append(box_id)

            # Check if the mask is empty before processing
            if masks.xy[j].size == 0:
                continue

            # Pass the segmentation mask to the fusion function
            fusion_result = self.lidar_camera_fusion(pts_3d[j], pts_2d[j], img, masks.xy[j], int(cls), lidar2camera, erosion_factor=erosion_factor, depth_factor=depth_factor, PCA=self.pca)
            
            
            

            # If the fusion is successfull, retrieve relevant bbox data (e.g. for RoboCar)
            if fusion_result is not None:
                filtered_points_of_object, corners_3D, yaw = fusion_result

                all_corners_3D.append(corners_3D)
                all_filtered_points_of_object.append(filtered_points_of_object)

                # Retrieve the ROS data (e.g. relevant for RoboCar)
                ROS_type = int(np.int32(cls))
                bottom_indices = np.argsort(corners_3D[:, 2])[:4]
                ROS_ground_center = np.mean(corners_3D[bottom_indices], axis=0)
                ROS_dimensions = np.ptp(corners_3D, axis=0)                
                ROS_points = corners_3D
                time_between_frames = 0.1

                # Compute the velocity and direction (only available with tracking)
                if box_id in self.last_ground_center_of_id and not np.array_equal(self.last_ground_center_of_id[box_id], ROS_ground_center):
                    ROS_direction, ROS_velocity = self.compute_relative_object_velocity(self.last_ground_center_of_id[box_id], ROS_ground_center, time_between_frames)
                else:
                    ROS_direction = None
                    ROS_velocity = None

                self.last_ground_center_of_id[box_id] = ROS_ground_center
                # Save the ROS information of the current object and append it to an array that contains all information of all objects in the frame
                if ROS_type is not None and ROS_ground_center is not None and ROS_direction is not None and ROS_dimensions is not None and ROS_velocity is not None and ROS_points is not None:
                    objects3d_data.append([ROS_type, ROS_ground_center, ROS_direction, ROS_dimensions, ROS_velocity, ROS_points])   
            return (all_corners_3D,all_filtered_points_of_object,cam_calib, lidar_calib)            
    
    def get_pred_bbox_edges(self,corners_2D):
        edges = []
        
        # Define the edges in the desired order
        edges_indices = [
            (1, 7), (1, 6), (4, 7), (4, 6), # Top edges
            (0, 2), (0, 3), (2, 5), (3, 5), # Bottom eges
            (0, 1), (2, 7), (3, 6), (4, 5), # Vertical edges
        ]
        
        # Map the corner indices to actual points
        for (start_idx, end_idx) in edges_indices:
            edges.append((corners_2D[start_idx], corners_2D[end_idx]))

        return edges
    
    def draw_box(self,ori_img,box,calib,color=(0,0,255)):
        cam_calib, lidar_calib = calib
        cam_K, cam_R, cam_T, cam_dis = cam_calib
        lidar_R, lidar_T = lidar_calib
        img = ori_img.copy()
        pts_3d,pts_2d = self.dataloader.project_pointcloud_lidar_to_image(box, img, cam_K, cam_R, cam_T, cam_dis, lidar_R, lidar_T, just_project=True,just_K=True)
        if pts_2d != []:
            edge_2d = self.get_pred_bbox_edges(pts_2d)
            for pred_edge in edge_2d:
                pt1 = tuple(np.int32(pred_edge[0]))
                pt2 = tuple(np.int32(pred_edge[1]))
                cv2.line(img, pt1, pt2, color, 3)
        return img
        
        
        
    def points_cluster(self,points, eps = 0.5, depth_facter=20):
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

    def filter_points_with_depth_clustering(self, points_of_object, eps=0.5, depth_factor=20, print_info=False):
        """ Return filtered 3D LiDAR Points using DBSCAN """

        if print_info:
            print("\nFilter 3D LiDAR Points using DBSCAN ...")

        # Compute min_samples as a percentage of number of points
        min_samples = max(5, int(len(points_of_object) * 0.01))
        
        # Compute the depth values using the Euclidean Distance
        depths = np.sqrt(points_of_object[:, 0]**2 + points_of_object[:, 1]**2)

        # Density-Based Clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(depths.reshape(-1, 1))

        # Calculate the number of points in each cluster
        unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
        cluster_counts_dict = dict(zip(unique_clusters, cluster_counts))

        # Sort clusters based on the number of points they contain
        sorted_clusters = sorted(cluster_counts_dict, key=cluster_counts_dict.get, reverse=True)

        # Keep only the clusters with the most points
        top_cluster = sorted_clusters[:1]

        # Dynamic Depth Range
        depth_ranges = {}
        for cluster_label in top_cluster:
            if cluster_label == -1:
                continue  # Skip noise points
            cluster_depth_values = depths[clusters == cluster_label]
            min_depth = np.min(cluster_depth_values)
            max_depth = np.max(cluster_depth_values)
            # Adjust the depth range by a factor
            range_length = max_depth - min_depth
            min_depth_adjusted = min_depth + (1 - depth_factor) * range_length / 2
            max_depth_adjusted = max_depth - (1 - depth_factor) * range_length / 2
            depth_ranges[cluster_label] = (min_depth_adjusted, max_depth_adjusted)

        # Filtering
        filtered_points_of_object = []
        for cluster_label, depth_range in depth_ranges.items():
            min_depth, max_depth = depth_range
            valid_cluster_mask = clusters == cluster_label
            valid_cluster_indices = np.where(valid_cluster_mask)[0]
            cluster_points = points_of_object[valid_cluster_indices]
            depth_mask = (cluster_points[:, 0] >= min_depth) & (cluster_points[:, 0] <= max_depth)
            filtered_points_of_object.extend(cluster_points[depth_mask])

        return np.array(filtered_points_of_object)

    def create_bbox_3d(self,pts_3d, lidar2cam):
        min_point = np.min(pts_3d, axis=0)
        max_point = np.max(pts_3d, axis=0)

        # Define the corners of the bounding box
        corners_3D = np.array([
            [min_point[0], min_point[1], min_point[2]],  # Corner 0
            [min_point[0], min_point[1], max_point[2]],  # Corner 1
            [min_point[0], max_point[1], min_point[2]],  # Corner 2
            [max_point[0], min_point[1], min_point[2]],  # Corner 3
            [max_point[0], max_point[1], max_point[2]],  # Corner 4
            [max_point[0], max_point[1], min_point[2]],  # Corner 5
            [max_point[0], min_point[1], max_point[2]],  # Corner 6
            [min_point[0], max_point[1], max_point[2]]   # Corner 7
        ])

        return corners_3D

    def create_bbox_3d_pca(self,pts_3d, lidar2cam):
        pts_3d_cam = lidar2cam @ pts_3d.T
        pt_cloud_2 = o3d.geometry.PointCloud()
        pt_cloud_2.points = o3d.utility.Vector3dVector(pts_3d_cam.T)
        
        obb2 = pt_cloud_2.get_oriented_bounding_box()
        
        R_camera_coord = np.array(obb2.R)
        
        yaw = np.arctan2(R_camera_coord[0, 0], R_camera_coord[2, 0])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pts_3d)
        
        obb = point_cloud.get_oriented_bounding_box()
        
        corners_3d = np.asarray(obb.get_box_points())
        
        return corners_3d, yaw
    
    def create_bbox_3d_pca_no_z_rotation(self,pts_3d,lidar2cam):
        pts_3d_cam = lidar2cam @ pts_3d
        
        pt_cloud_2 = o3d.geometry.PointCloud()
        pt_cloud_2.points = o3d.utility.Vector3dVector(pts_3d_cam)
        
        obb2 = pt_cloud_2.get_oriented_bounding_box()
        
        R_camera_coord = np.array(obb2.R)
        
        yaw = np.arctan2(R_camera_coord[0, 0], R_camera_coord[2, 0])
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pts_3d)

        # Compute the oriented bounding box using PCA
        obb = point_cloud.get_oriented_bounding_box()

        # Get the rotation matrix of the oriented bounding box
        R = np.array(obb.R)

        # Change the rotation matrix
        R[:, 0] = [R[0, 0], R[1, 0], 0]     # Ensure x-axis rotation is only in x-y plane
        R[:, 1] = [R[0, 1], R[1, 1], 0]     # Ensure y-axis rotation is only in x-y plane
        R[:, 2] = [0, 0, 1]                 # Ensure z-axis rotation is discarded
        
        obb_new = o3d.geometry.OrientedBoundingBox(obb.center, R, obb.extent)
        corners_3D = np.asarray(obb_new.get_box_points())
        
        return corners_3D, yaw
    
    def lidar_camera_fusion(self,pts_3D, pts_2D, frame, seg_mask, obj_class, lidar2cam, erosion_factor, depth_factor, PCA):
        """ Fuse 3D LiDAR Points with Camera frame and return filtered 3D LiDAR Points and 3D Bounding Box of current object """

        #print("\nFuse LiDAR points of detected objects with image ...")

        # Erode the segmentation polygon
        eroded_seg = self.erode_polygon(seg_mask, frame, erosion_factor)

        if eroded_seg is None or len(eroded_seg) <= 2:
            #print("\n\tPass object (segmentation is too small)")
            return None
        
        all_points_of_object = []

        """ ----------------- FUSION -----------------"""
        # Create a mask for the eroded polygon
        mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
        cv2.fillPoly(mask, [eroded_seg.astype(np.int32)], color=1)

        # Convert 2D points to integer coordinates
        pts_3D_int = np.array(pts_3D).astype(np.int32)
        
        # Filter points using the mask
        # inside_mask_indices = mask[pts_2D_int[:, 1], pts_2D_int[:, 0]] == 1

        # # Select corresponding 3D points
        # all_points_of_object = pts_3D[inside_mask_indices]
        
        all_points_of_object = pts_3D_int
        """ ------------------------------------------"""

        # Skip objects with no points
        if len(all_points_of_object) == 0:
            return None

        # Filter points based on their distances
        filtered_points_of_object = self.filter_points_with_depth_clustering(np.array(all_points_of_object), depth_factor=depth_factor)

        # Skip objects with insufficient points and if points are coplanar (e.g. all points have the same z coordinate)
        if len(filtered_points_of_object) < 4 or self.is_coplanar(filtered_points_of_object):
            print("\n\tPass object (not enough points after filtering)")
            return None

        # Get the edges and corners of the 3D bounding box
        yaw = 0
        if PCA:
            # Compute the oriented 3D bounding box (discard the rotation along the z-axis for detected objects other than persons)
            # if obj_class == 0:
            bbox_corners_3D, yaw = self.create_bbox_3d_pca(np.array(filtered_points_of_object), lidar2cam)
            # else:
            #     # Use PCA without z-axis rotation for other classes
            #     bbox_corners_3D, yaw = self.create_bbox_3d_pca_no_z_rotation(filtered_points_of_object, lidar2cam)
        
        # Note: Pedestrian class always uses PCA 
        else:
            # Compute the oriented 3D bounding box (discard the rotation along the z-axis for detected objects other than persons)
            # if obj_class == 0:
            bbox_corners_3D, yaw = self.create_bbox_3d_pca_no_z_rotation(np.array(filtered_points_of_object), lidar2cam)
            # else:
            #     # Don't use PCA for classes other than Pedestrians
            #     bbox_corners_3D = self.create_bbox_3d(filtered_points_of_object, lidar2cam)

        return filtered_points_of_object, bbox_corners_3D, yaw
    
    def erode_polygon(self, polygon, img, erosion_factor=25, print_info=False):
        """ Return an Eroded Polygon based on an Erosion Factor """

        if print_info:
            print(f"\nErode Segmentation Mask with erosion_factor = {erosion_factor} ...")

        # Check if the polygon is empty
        if len(polygon) == 0:
            return None
        
        # Calculate the area of the polygon
        polygon_area = cv2.contourArea(polygon.astype(np.int32))

        # Calculate the scaled erosion factor based on the square root of the polygon area
        scaled_erosion_factor = np.sqrt(polygon_area) / erosion_factor

        # Create a mask for the polygon
        mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], color=255)

        # Define the erosion kernel size based on the erosion factor
        kernel_size = int(2 * scaled_erosion_factor + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply erosion to the mask
        eroded_mask = cv2.erode(mask, kernel, iterations=1)

        # Find contours of the eroded mask
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours were found (polygon is too small)
        if len(contours) == 0:
            return None

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Convert the contour to the polygon format
        eroded_polygon = np.squeeze(largest_contour).astype(np.float32)

        return eroded_polygon
    
    def is_coplanar(self, points, print_info=False):
        """ Check if an array of 3D LiDAR Points is coplanar """

        if print_info:
            print("\nCheck if 3D LiDAR Points are coplanar ...")

        # Only check arrays of more than 4 points 
        if len(points) < 4:
            return True
        
        # Compute the covariance matrix and its eigenvalues
        cov_matrix = np.cov(points.T)
        eigenvalues, _ = np.linalg.eig(cov_matrix)
        
        # If one of the eigenvalues is close to zero, points are nearly coplanar
        return np.min(eigenvalues) < 1e-6
    
    def compute_relative_object_velocity(self,ground_center_frame1, ground_center_frame2, time_between_frames):
        # Compute displacement vector between ground centers
        displacement = ground_center_frame2 - ground_center_frame1

        # Compute relative velocity components
        relative_velocity_x = displacement[0] / time_between_frames
        relative_velocity_y = displacement[1] / time_between_frames
        relative_velocity_z = displacement[2] / time_between_frames

        # Return velocity vector
        return displacement, np.array([relative_velocity_x, relative_velocity_y, relative_velocity_z])
        
        
        
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
    fusion_processor = Fusion(dataloader=data_loader,PCA=True)
    model = YOLO(r'weights\yolov8l-seg.pt')
    
    # point_cloud = data_loader.load_pcd_data(r'ship\id_05\008176.pcd')
    # data_loader.visualize_lidar(point_cloud)
    img_name = '011254.png'
    lidar_name = '1625125474647414894.bin'
    img = cv2.imread(os.path.join(img_dir,img_name))
    results = model.predict(img)
    lidar_points = data_loader.load_lidar_data(os.path.join(lidar_dir,lidar_name))
    fusion_processor.fuse(img,lidar_points,results[0].cpu(),img_name,lidar_name)
    
    