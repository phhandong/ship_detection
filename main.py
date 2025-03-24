import os
import argparse
import cv2
import numpy as np
import json
import random
import shutil
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics import settings
from ultralytics.utils.downloads import attempt_download_asset
import  matplotlib.pyplot as PIL

class Model_Trainer():
    def __init__(self,data_dir,weight_dir,output_dir):
      self.data_dir = data_dir
      self.weight_dir = weight_dir
      self.output_dir = output_dir
   
    def download():
        # model = YOLO(r"ultralytics\cfg\models\v8\yolov8.yaml")  # build a new model from YAML
      
        assets = [f"yolov8{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "-cls", "-seg")]
        for x in assets:
            attempt_download_asset(f"weights/{x}")
            
            
    def draw_box(self,ori_img,box,img_name):
        img = ori_img.copy()
        for (x1,y1,x2,y2) in box.xyxy:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
        cv2.imwrite(os.path.join(self.output_dir,img_name),img)
        
        
    def draw_mask(self,ori_img,masks,img_name):
        img = ori_img.copy()
        overlay = img.copy()
        for mask_xy in masks:  # masks.xy contains a list of arrays, each representing a mask boundary
            # Generate a random color for the mask
            # color = [random.randint(0, 255) for _ in range(3)]
            color = (0,255,0)

            # Convert the mask boundary to integer coordinates
            points = np.array(mask_xy, dtype=np.int32)

            # Draw the mask boundary as a polygon
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

            # Optionally, fill the mask area (semi-transparent)
            cv2.fillPoly(img, [points], color=color)
        alpha = 0.5  # Transparency factor (0: fully transparent, 1: fully opaque)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        cv2.imwrite(os.path.join(self.output_dir,img_name.split('.')[0]+'-seg.'+img_name.split('.')[1]),img)

        return img
    
                
    def inference_single(self,model_name = 'yolov8l-seg',img_name = '006210.png'):
        model = YOLO(os.path.join(self.weight_dir,model_name+'.pt'))  # load a pretrained model (recommended for training)
        img_dir = os.path.join(self.data_dir,'left_images',img_name)
        results = model(img_dir)
        results[0].show()
        results = results[0].cpu()
        img_n,ext = img_name.split('.')
        
        if '-' in model_name:            
            output_name = img_n + '-' + model_name.split('-')[-1] + '.' + ext
            json_name = img_n + '-' + model_name.split('-')[-1] + '.json'
            
            with open(os.path.join(self.output_dir,json_name), 'w') as f:
                json.dump(results.to_json(), f, indent=4)
            self.draw_mask(results.orig_img,results.masks.xy,img_name)
        else:
            output_name = img_name
            self.draw_box(results.orig_img,results.boxes,img_name)
        # results[0].save(os.path.join(self.output_dir,output_name))
        # results.save_crop(os.path.join(self.output_dir,output_name.replace(ext,'json')))
        
        return results
        
            

    def split_dataset(self):
        img_list = os.listdir(os.path.join(self.data_dir,'left_images'))
        label_list = os.listdir(os.path.join(self.data_dir,'annotations'))
        valid_list = []
        for i,label in enumerate(label_list):
            if label.split('.')[0] + '.png' in img_list:
              valid_list.append(label.split('.')[0] + '.png')
        for i,(img,label) in tqdm(enumerate(zip(valid_list,label_list))):
            r = random.random()
            if r < 0.1:
                target = 'test'
            elif r < 0.2:
                target = 'val'
            else:
                target = 'train'
            shutil.copy(os.path.join(self.data_dir,'left_images',img),os.path.join(self.data_dir,target,'images',img))
            shutil.copy(os.path.join(self.data_dir,'annotations',label),os.path.join(self.data_dir,target,'labels',label))            
  
if __name__ == '__main__':
    model_trainer = Model_Trainer(r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\stereo',r'.\weights',r'E:\Python_Project\Ship_Detection\yolo8\results\wo_train_v8')
    # model_init.split_dataset()
    result = model_trainer.inference_single(model_name = 'yolov8l-seg',img_name='006210.png')
  
    # Train
    # model = YOLO("yolov8l.yaml").load("weights/yolov8l.pt")  # build from YAML and transfer weights
    # results = model.train(data="pohang.yaml", epochs=100, imgsz=640, batch=2)
    # print(settings)
    # print(model.info())
    