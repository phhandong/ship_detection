import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import json
import struct
from tqdm import tqdm
import open3d as o3d
from numpy.linalg import inv

class DataGenerator:
    def __init__(self, base_dir, img_dir, xband_dir,lidar_dir,calib_dir):
        self.base_dir = base_dir
        self.img_dir = img_dir
        self.xband_dir = xband_dir
        self.lidar_dir = lidar_dir
        self.calib_dir = calib_dir
        self.file_list = list(set(f.split('.')[0] for f in os.listdir(img_dir)))
        self.file_list.sort()
        self.labeled = []
        self.img_list = []
        self.lidar_list = []
        
        
    def sample_loader(self, img_name):
        img = cv2.imread(os.path.join(self.img_dir, img_name + '.jpg'))
        label = json.load(open(os.path.join(self.img_dir, img_name + '.json')))
        return img, label
    
    def draw_bbox(self, img, label):
        for obj in label:
            x, y, w, h = obj['xmin'],obj['ymin'],obj['width'],obj['height']
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
            cv2.putText(img, 'ID:'+str(obj['id']), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
        return img

    def visualize_rgb(self, img_name):
        img, label = self.sample_loader(img_name)
        if label['annotations'] != []:
            self.labeled.append(img_name)
            img = self.draw_bbox(img, label['annotations'])
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        self.img_list.append(img)
        return img
    
    def gen_video(self, video_list, out_dir, fps=30):
        w, h = video_list[0].shape[1], video_list[0].shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_dir, fourcc, fps, (w, h)) 
        for img in video_list:
            out.write(img)
        out.release()
        
    def visualize_projection_video(self, output_video_path, fps=30):
        # Generate the match list for images and LiDAR files
        match_list = self.create_match_list()

        # Read the first image to get the size (for video writer)
        first_image_path = os.path.join(self.img_dir, match_list[0][0])
        first_image = cv2.imread(first_image_path)
        height, width = first_image.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Iterate through the match list and create the video
        for img_name, lidar_name in tqdm(match_list):
            img_with_projection = self.visualize_projection(lidar_name, img_name)
            out.write(img_with_projection)  # Write frame to video
            
        out.release()  # Release the video writer

    def load_lidar_data(self, file_name):
        pointcloud = []
        with open(file_name, 'rb') as f:
            while True:
                data = f.read(16)  # 4 * 4 bytes
                if not data:
                    break
                x, y, z, intensity = struct.unpack('ffff', data)
                pointcloud.append([x, y, z, intensity])
        return np.array(pointcloud)

    def visualize_lidar(self, pointcloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        o3d.visualization.draw_geometries([pcd])

    def process_lidar(self, lidar_file_name):
        file_path = os.path.join(self.lidar_dir, lidar_file_name)
        if not os.path.exists(file_path):
            print(f"LiDAR file {file_path} does not exist!")
            return

        # load cloud points
        pointcloud = self.load_lidar_data(file_path)

        # visualize cloud points
        # self.visualize_lidar(pointcloud)

        return pointcloud

    def convert_txt_to_json(self, txt_file_name):
        key,value = '',[]
        with open(os.path.join(self.calib_dir, txt_file_name)) as f:
            lines = f.readlines()
            calib = {}
            for line in lines:
                if line.strip() == '':
                    calib[key] = value
                    key = ''
                    value = []
                elif ':' in line.strip():
                    if key != '':
                        calib[key] = value
                    key,value = line.strip().split(':')
                    value = [list(map(float, value.split()))]
                    calib[key] = value
                else:
                    value.append(list(map(float, line.strip().split())))
        with open(os.path.join(self.calib_dir, txt_file_name.split('.')[0]+'.json'), 'w') as f:
            json.dump(calib, f, indent=4)
    
    def load_calibration(self, calib_file_name=None):
        with open(os.path.join(self.calib_dir, 'Base2camera.json')) as f:
            cam_calib = json.load(f)
        cam_K = np.array(cam_calib['camera_matrix_left'])
        cam_R = np.array(cam_calib['Tr_gnss_to_cam_left'])[:3,:3]
        cam_T = np.array(cam_calib['Tr_gnss_to_cam_left'])[:,3]
        cam_dis_raw = cam_calib['distortion_coefficent_left']

        # Flatten the distortion coefficients if it's a list of lists
        cam_dis = []
        for item in cam_dis_raw:
            if isinstance(item, list):
                cam_dis.extend(item)  # Extend if it's a nested list
            else:
                cam_dis.append(item)  # Add the element if it's a single value

        # Convert the flattened list into a numpy array
        cam_dis = np.array(cam_dis)

        with open(os.path.join(self.calib_dir, 'Base2lidar.json')) as f:
            lidar_camlib = json.load(f)
        lidar_R = np.array(lidar_camlib['Tr_gnss_to_lidar'])[:3,:3]
        lidar_T = np.array(lidar_camlib['Tr_gnss_to_lidar'])[:,3]

        
        return (cam_K, cam_R, cam_T, cam_dis), (lidar_R, lidar_T)        

    def project_pointcloud_lidar_to_image(self, pointcloud, image, cam_K, cam_R, cam_T, cam_dis, lidar_R, lidar_T,normalized_distances):
        cam_R = cam_R.T
        cam_T = - cam_R @ cam_T.reshape(-1,1)
        cam_TR = np.vstack((np.hstack((cam_R,cam_T.reshape(-1,1))),np.reshape(np.array([0,0,0,1]),(1,-1))))   # (4,4)
        lidar_TR = np.vstack((np.hstack((lidar_R,lidar_T.reshape(-1,1))),np.reshape(np.array([0,0,0,1]),(1,-1))))  # (4,4)

        pointcloud_homogeneous = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1)))).T  # (4, N)
        pointcloud_homogeneous = (cam_TR @ lidar_TR @ pointcloud_homogeneous)[:3,:] # (3,N)
        
        points_2d_homogeneous = cam_K @ pointcloud_homogeneous  # (3, N)

        points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

        img_with_projection = image.copy()
        for (u, v),distance in zip(points_2d.T,normalized_distances):
            u, v = int(u), int(v)
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                cv2.circle(img_with_projection, (u, v), 3, (0, int((distance)*255), int((1-distance)*255)), -1)

        return img_with_projection
    
    def create_match_list(self):
        # Get the list of image and LiDAR files
        self.img_list = sorted(os.listdir(self.img_dir))
        self.lidar_list = sorted(os.listdir(self.lidar_dir))

        match_list = []

        # Define a helper function to extract the timestamp (in nsec) from the filename
        def extract_timestamp(filename):
            # Assuming filenames are based on nanosecond timestamps like "1630451234567891234.jpg"
            timestamp_str = filename.split('.')[0]  # Extract the base name without extension
            try:
                return int(timestamp_str)  # Convert to integer (nanosecond timestamp)
            except ValueError:
                return None

        # Create a list of (filename, timestamp) for both image and LiDAR files
        img_timestamps = [(img_name, extract_timestamp(img_name)) for img_name in self.img_list]
        lidar_timestamps = [(lidar_name, extract_timestamp(lidar_name)) for lidar_name in self.lidar_list]

        # For each image, find the LiDAR file with the nearest timestamp
        
        for i, (img_name, img_timestamp) in enumerate(img_timestamps):
            if img_timestamp is None or i % 10 != 0:
                continue  # Skip non-10th images
            # Find the LiDAR file with the closest timestamp
            closest_lidar_name = None
            closest_lidar_diff = None
            for lidar_name, lidar_timestamp in lidar_timestamps:
                if lidar_timestamp is None:
                    continue  # Skip if we can't extract a valid timestamp
                # Calculate the time difference between the image and LiDAR timestamps
                time_diff = abs(img_timestamp - lidar_timestamp)
                if closest_lidar_diff is None or time_diff < closest_lidar_diff:
                    closest_lidar_diff = time_diff
                    closest_lidar_name = lidar_name

            if closest_lidar_name:
                match_list.append([img_name, closest_lidar_name])

        # Save the match list to a text file
        with open('match_list.txt', 'w') as f:
            for item in match_list:
                f.write("%s\n" % str(item))

        return match_list
    
    def visualize_projection(self, lidar_name, img_name):
        # load image
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        # load pointcloud
        cloud_points = self.process_lidar(lidar_name)
        
        distances = np.linalg.norm(cloud_points[:, :2], axis=1)  # Compute distance for [x, y, z]
        min_range = 4.0
        max_range = 75.0
    
        # Filter points that are closer than the minimum range
        cloud_points = cloud_points[distances >= min_range]
        distances = distances[(distances >= min_range) & (distances <= max_range)]
        
        # Normalize distances to [0, 1] for coloring
        normalized_distances = []
        for d in distances:
          normalized_distances.append((min(max(d,min_range),max_range) - min_range) / (max_range - min_range))
        # load calibration
        cam_calib, lidar_calib = self.load_calibration()
        cam_K, cam_R, cam_T, cam_dis = cam_calib
        lidar_R, lidar_T = lidar_calib
        img_with_projection = self.project_pointcloud_lidar_to_image(cloud_points[:,:3], img, cam_K, cam_R, cam_T, cam_dis, lidar_R, lidar_T,normalized_distances)
        return img_with_projection


if __name__ == '__main__':
    base_dir = r'E:\Python_Project\Ship_Detection\dataset\MOANA\Single_Island'
    data_dir = r'E:\Python_Project\Ship_Detection\dataset\MOANA\Single_Island\sensor_data'
    img_dir = os.path.join(data_dir,'Camera_left')
    xband_dir = os.path.join(data_dir,'X-band')
    lidar_dir = os.path.join(data_dir,'LIDAR')
    calib_dir = os.path.join(base_dir,'calibration')
    data_loader = DataGenerator(data_dir, img_dir, xband_dir,lidar_dir,calib_dir)

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
    
    
    
    # data_loader.visualize_projection('1724240642520185841.bin', '1724240641715529007.jpg')
    # data_loader.visualize_lidar(data_loader.process_lidar('1724240642520185841.bin'))
    data_loader.convert_txt_to_json('Base2camera.txt')
    data_loader.convert_txt_to_json('Base2lidar.txt')
    data_loader.visualize_projection_video('output_video.mp4')

    
        
    