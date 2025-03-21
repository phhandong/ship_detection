import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm
import open3d as o3d

class DataGenerator:
    def __init__(self, base_dir, img_dir, xband_dir,lidar_dir,calib_dir,label_dir):
        self.base_dir = base_dir
        self.img_dir = img_dir
        self.xband_dir = xband_dir
        self.lidar_dir = lidar_dir
        self.calib_dir = calib_dir
        self.label_dir = label_dir 
        self.file_list = list(set(f.split('.')[0] for f in os.listdir(img_dir)))
        self.file_list.sort()
        self.labeled = []
        self.img_list = []
        self.lidar_list = []
        
    def sample_loader(self, img_name):
        img = cv2.imread(os.path.join(self.img_dir, img_name + '.png'))
        if not os.path.exists(os.path.join(self.label_dir, img_name + '.txt')):
            return img, []
        label_list = []
        with open(os.path.join(self.label_dir, img_name + '.txt')) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            labels = [list(map(float, line.split())) for line in lines]
            if labels == []:
              return img, []
            label_list.append(labels[0][1:])
        return img, label_list
    
    def draw_bbox(self, img, label, img_name):
        img_w,img_h = img.shape[1],img.shape[0]
        for obj in label:
            x, y, w, h = int(obj[0]*img_w),int(obj[1]*img_h),int(obj[2]*img_w/2),int(obj[3]*img_h/2)
            cv2.rectangle(img, (x-w, y-h), (x+w, y+h), (0, 255, 0), 5)
            cv2.putText(img, 'ID:'+str(0), (x+w, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
        return img

    def visualize_rgb(self, img_name):
        img, label = self.sample_loader(img_name)
        if label != []:
            # self.labeled.append(img_name)
            img = self.draw_bbox(img, label, img_name)
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.show()
        # cv2.putText(img, 'ID:'+ img_name, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        return img
    
    def visualize_rgb_video(self):
        for img_name in tqdm(self.file_list[:2500]):
            self.img_list.append(self.visualize_rgb(img_name))
        self.gen_video(self.img_list, 'Pohang_camright.avi')
    
    def gen_video(self, video_list, out_dir, fps=60):
        w, h = video_list[0].shape[1], video_list[0].shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_dir, fourcc, fps, (w, h))  # 使用 fps 参数
        for img in video_list:
            out.write(img)
        out.release()

    def load_lidar_data(self, file_name):
        """
        加载 LiDAR 点云数据 (.bin 文件)。

        参数:
        - file_name: LiDAR 数据文件的完整路径。

        返回:
        - pointcloud: 点云数据，形状为 (N, 6)
        """
        # pointcloud = []
        # with open(file_name, 'rb') as f:
        #     while True:
        #         # 读取每个点的 x, y, z, intensity (每个 float 占 4 字节)
        #         data = f.read(16)  # 4 * 4 bytes
        #         if not data:
        #             break
                # x, y, z, intensity = struct.unpack('ffff', data)
        lidar_dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('time', np.uint32), ('reflectivity', np.uint16), ('ambient', np.uint16), ('range', np.uint32)]
        scan = np.fromfile(file_name, dtype=lidar_dtype)
        pointscloud = np.stack((scan['x'], scan['y'], scan['z'], scan['intensity']), axis=-1)
        # pointcloud.append([x, y, z, intensity])
        return np.array(pointscloud)

    def visualize_lidar(self, pointcloud):
        """
        可视化 LiDAR 点云数据。

        参数:
        - pointcloud: 点云数据，形状为 (N, 4)，每行是 (x, y, z, intensity)。
        """

        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])  # 只使用 x, y, z
        o3d.visualization.draw_geometries([pcd])
    
    def visualize_lidar_video(self, output_video='lidar_visualization.avi', fps=30):
        """
        遍历 self.label_dir 中的所有文件，加载点云数据并可视化，输出为视频。

        参数:
        - output_video: 输出视频文件名。
        - fps: 视频帧率。
        """
        video_list = []

        # 遍历 self.label_dir 中的所有文件
        lidar_files = sorted(os.listdir(self.lidar_dir))
        for lidar_file in tqdm(lidar_files[:2000], desc="Processing LiDAR files"):
            # 加载点云数据
            lidar_path = os.path.join(self.lidar_dir, lidar_file)
            pointcloud = self.load_lidar_data(lidar_path)

            # 可视化点云
            img = self.visualize_lidar_to_image(pointcloud)

            # 将可视化结果添加到视频列表
            video_list.append(img)

        # 生成视频
        self.gen_video(video_list, output_video, fps=fps)

    def visualize_lidar_to_image(self, pointcloud, width=2048, height=1080):
        """
        将点云可视化为图像。

        参数:
        - pointcloud: 点云数据，形状为 (N, 4)。
        - width: 输出图像的宽度。
        - height: 输出图像的高度。

        返回:
        - img: 可视化后的图像。
        """
        # 创建一个空白图像
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # 将点云的 x, y 坐标映射到图像平面
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        # 归一化到图像范围
        x_img = ((x - x.min()) / (x.max() - x.min()) * (width - 1)).astype(np.int32)
        y_img = ((y - y.min()) / (y.max() - y.min()) * (height - 1)).astype(np.int32)

        # 绘制点云到图像
        for i in range(len(x_img)):
            cv2.circle(img, (x_img[i], y_img[i]), 1, (255, 255, 255), -1)

        return img

    def process_lidar(self, lidar_file_name):
        """
        加载并处理 LiDAR 数据。

        参数:
        - lidar_file_name: LiDAR 数据文件名（不含路径）。
        """
        file_path = os.path.join(self.lidar_dir, lidar_file_name)
        if not os.path.exists(file_path):
            print(f"LiDAR file {file_path} does not exist!")
            return

        # load cloud points
        pointcloud = self.load_lidar_data(file_path)

        # visualize cloud points
        # self.visualize_lidar(pointcloud)

        return pointcloud

    def quaternion_to_rotation_matrix(self,quaternion):
        """
        将四元数转换为旋转矩阵。

        参数:
        - quaternion: 四元数 [q_z, q_x, q_y, q_w]

        返回:
        - R: 旋转矩阵 (3, 3)
        """
        q_x, q_y, q_z, q_w = quaternion
        R = np.array([
            [1 - 2 * (q_y**2 + q_z**2), 2 * (q_x * q_y - q_z * q_w), 2 * (q_x * q_z + q_y * q_w)],
            [2 * (q_x * q_y + q_z * q_w), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y * q_z - q_x * q_w)],
            [2 * (q_x * q_z - q_y * q_w), 2 * (q_y * q_z + q_x * q_w), 1 - 2 * (q_x**2 + q_y**2)]
        ])
        return R

    def parse_camera_parameters(self,stereo):
        """
        从 stereo_left 数据解析相机内参矩阵 K 和畸变参数 cam_dis。

        参数:
        - stereo_left: 包含相机参数的字典。

        返回:
        - K: 相机内参矩阵 (3, 3)。
        - cam_dis: 畸变参数列表。
        """
        # 提取参数
        focal_length = stereo["focal_length"]
        cc_x = stereo["cc_x"]
        cc_y = stereo["cc_y"]
        distortion_coefficients = stereo["distortion_coefficients"]

        # 构造内参矩阵 K
        K = np.array([
            [focal_length, 0, cc_x],
            [0, focal_length, cc_y],
            [0, 0, 1]
        ])

        # 提取畸变参数
        cam_dis = distortion_coefficients

        return (K, cam_dis)

    def load_calibration(self, calib_file_name=None):
        """
        加载标定数据 (.json 文件)。

        参数:
        - calib_file_name: 相机标定数据文件的完整路径。

        返回:
        - K: 相机内参矩阵 (3, 3)。
        - R: 相机旋转矩阵 (3, 3)。
        - T: 相机平移向量 (3, 1)。
        """
        with open(os.path.join(self.calib_dir, 'extrinsics.json')) as f:
            calib = json.load(f)
        with open(os.path.join(self.calib_dir, 'intrinsics.json')) as f:
            intri = json.load(f)                
        cam_K,cam_dis = self.parse_camera_parameters(intri['stereo_left'])
        cam_calib = calib['stereo_left']
        cam_R = self.quaternion_to_rotation_matrix(cam_calib['quaternion'])
        cam_T = np.array(cam_calib['translation']).reshape(-1,1) 
        
        lidar_camlib = calib['lidar_front']
        lidar_R = self.quaternion_to_rotation_matrix(lidar_camlib['quaternion'])
        lidar_T = np.array(lidar_camlib['translation']).reshape(-1,1)
         
        return (cam_K, cam_R, cam_T, cam_dis), (lidar_R, lidar_T)        
    
    def project_pointcloud_lidar_to_image(self, pointcloud, image, cam_K, cam_R, cam_T, cam_dis, lidar_R, lidar_T):
        """
        将点云投影到图像上，并在图像上绘制投影点。

        参数:
        - pointcloud: 点云数据，形状为 (N, 3)，每行是 (X, Y, Z)。
        - image: 图像 (H, W, 3)。
        - cam_K: 相机内参矩阵 (3, 3)。
        - cam_R: 相机旋转矩阵 (3, 3)。
        - cam_T: 相机平移向量 (3, 1)。
        - cam_dis: 相机畸变参数。
        - lidar_R: LiDAR 旋转矩阵 (3, 3)。
        - lidar_T: LiDAR 平移向量 (3, 1)。

        返回:
        - 带有投影点的图像。
        """
        # 构造齐次变换矩阵
        
        # inv(cam_TR)
        cam_R = cam_R.T
        cam_T = - cam_R @ cam_T
        
        cam_TR = np.vstack((np.hstack((cam_R,cam_T.reshape(-1,1))),np.reshape(np.array([0,0,0,1]),(1,-1))))   # (4,4)
        lidar_TR = np.vstack((np.hstack((lidar_R,lidar_T.reshape(-1,1))),np.reshape(np.array([0,0,0,1]),(1,-1))))  # (4,4)

        pointcloud_homogeneous = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1)))).T  # (4, N)
        pointcloud_homogeneous = (cam_TR @ lidar_TR @ pointcloud_homogeneous)[:3,:] # (3,N)
        
        points_2d_homogeneous = cam_K @ pointcloud_homogeneous  # (3, N)

        points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

        # points_2d = points_2d.T.reshape(-1, 1, 2)  # 转换为 OpenCV 格式 (N, 1, 2)
        # points_2d_undistorted = cv2.undistortPoints(points_2d, cam_K, np.array(cam_dis), None, cam_K)  # 校正畸变

        img_with_projection = image.copy()
        # for u, v in points_2d_undistorted[:, 0, :]:
        for u, v in points_2d.T:
            u, v = int(u), int(v)
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                # cv2.circle(img_with_projection, (image.shape[1] - u,image.shape[0] - v), 3, (0, 0, 255), -1)
                cv2.circle(img_with_projection, (u,v), 3, (0, 0, 255), -1)

        return img_with_projection
    
    def create_match_list(self):
        self.img_list = sorted(os.listdir(self.img_dir))
        self.lidar_list = sorted(os.listdir(self.lidar_dir))
        img_timestamp_dict = {}
        with open(os.path.join(os.path.dirname(self.img_dir), 'timestamp.txt')) as f:
            for line in f:
                timestamp,img_name = line.strip().split()
                timestamp = timestamp.split('.')
                timestamp = timestamp[0] + timestamp[1]
                img_timestamp_dict[img_name+'.png'] = int(timestamp)
        
        match_list = []

        def extract_timestamp(filename):
            timestamp_str = filename.split('.')[0]
            try:
                return int(timestamp_str)
            except ValueError:
                return None

        img_timestamp = [(img_name, extract_timestamp(img_name)) for img_name in self.img_list]
        lidar_timestamps = [(lidar_name, extract_timestamp(lidar_name)) for lidar_name in self.lidar_list]

        img_timestamps = []
        for (img_name, img_timestamp) in img_timestamp:
            if img_name in img_timestamp_dict.keys():
                img_timestamps.append((img_name, img_timestamp_dict[img_name]))        
        
        for i, (img_name, img_timestamp) in enumerate(img_timestamps):
            if img_timestamp is None:
                continue
            closest_lidar_name = None
            closest_lidar_diff = None
            for lidar_name, lidar_timestamp in lidar_timestamps:
                if lidar_timestamp is None:
                    continue
                time_diff = abs(img_timestamp - lidar_timestamp)
                if closest_lidar_diff is None or time_diff < closest_lidar_diff:
                    closest_lidar_diff = time_diff
                    closest_lidar_name = lidar_name

            if closest_lidar_name:
                match_list.append([img_name, closest_lidar_name])

        with open('match_list_pohang.txt', 'w') as f:
            for item in match_list:
                f.write("%s\n" % str(item))

        return match_list
    
    def visualize_projection(self, lidar_name, img_name):
        """
        可视化 LiDAR 点云投影到图像上。

        参数:
        - lidar_file_name: LiDAR 数据文件名（不含路径）。
        - img_name: 图像文件名（不含路径）。
        """
        # load image
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        # load pointcloud
        cloud_points = self.process_lidar(lidar_name)
        distances = np.linalg.norm(cloud_points[:, :2], axis=1)  # Compute distance for [x, y, z]
        min_range = 4.0
        max_range = 100.0
        # Filter points that are closer than the minimum range
        cloud_points = cloud_points[(distances >= min_range) & (distances <= max_range)]
        
        # load calibration
        cam_calib, lidar_calib = self.load_calibration()
        cam_K, cam_R, cam_T, cam_dis = cam_calib
        lidar_R, lidar_T = lidar_calib
        # project pointcloud to image
        img_with_projection = self.project_pointcloud_lidar_to_image(cloud_points[:,:3], img, cam_K, cam_R, cam_T, cam_dis, lidar_R, lidar_T)
        # plt.figure(figsize=(img.shape[0] // 20, img.shape[1] // 20))
        # plt.imshow(cv2.cvtColor(np.hstack([img,img_with_projection]), cv2.COLOR_BGR2RGB))
        # plt.imshow(cv2.cvtColor(img_with_projection, cv2.COLOR_BGR2RGB))
        # plt.show()
        # plt.savefig('projection.jpg')
        return img_with_projection
    
    def visualize_projection_video(self):
        video_list = []
        match_list = self.create_match_list()
        for m in tqdm(match_list[:2000]):
            video_list.append(self.visualize_projection(m[1], m[0]))
        self.gen_video(video_list, 'projection.avi')


if __name__ == '__main__':
    base_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00'
    data_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00'
    img_dir = os.path.join(data_dir,r'stereo\left_images')
    # img_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\stereo\left_images'
    xband_dir = os.path.join(data_dir,'X-band')
    lidar_dir = os.path.join(data_dir,r'lidar_front\points')
    calib_dir = os.path.join(base_dir,'calibration')
    label_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\labels\all\left'
    data_loader = DataGenerator(data_dir, img_dir, xband_dir,lidar_dir,calib_dir,label_dir)

    """   
    # convert calibration txt files to json files
    # convert_list = ['Base2camera.txt','Base2lidar.txt','Base2wband.txt','Base2xband.txt']
    # for c in convert_list:
    # data_loader.convert_txt_to_json(c)
    #  
    """
    
    """
    # visualize rgb images
    #
    # data_loader.convert_txt_to_json('Base2camera.txt')
    # lidar_file_name = "1724242392520782502.bin"  # 替换为实际文件名
    # pointcloud = data_loader.process_lidar(lidar_file_name)
    #
    # """
    
    # data_loader.visualize_rgb_video()
    
    # data_loader.visualize_projection('1724240642520185841.bin', '1724240641715529007.jpg')
    # data_loader.visualize_lidar(data_loader.process_lidar('1724240642520185841.bin'))
    # data_loader.visualize_lidar_video()
    data_loader.visualize_projection_video()

    
        
    