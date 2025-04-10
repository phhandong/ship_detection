import os
import cv2
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    data_dir = r'E:\Python_Project\Ship_Detection\dataset\2025-01-19-11-53-27\cam_images'
    dir_list = ['camera_1','camera_2','camera_3','camera_4']
    data_list = []
    for i,d in enumerate(dir_list):
        data_list.append(os.listdir(os.path.join(data_dir,d))[300:])
    for i in tqdm(range(len(data_list[0]))):
        img_list = []
        for j in range(len(data_list)):
            img_list.append(os.path.join(data_dir,dir_list[j],data_list[j][i]))
        img = cv2.imread(img_list[0])[:,:1650,:]
        for j in range(1,len(img_list)):
            img = np.concatenate((img,cv2.imread(img_list[j])[:,:1700,:]),axis=1)
        cv2.imwrite(os.path.join(data_dir,'concat',data_list[0][i]),img)