import os
import argparse
import cv2
import numpy as np
import json
import random
import time
import shutil
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics import settings
from ultralytics.utils.downloads import attempt_download_asset
import  matplotlib.pyplot as PIL
from data_utils.data_utils_pohang import DataGenerator

class Model_Trainer():
    def __init__(self,data_dir,weight_dir,output_dir,run_dir='./runs',model_config='./yolov8.yaml',data_config='pohang.yaml'):
      self.data_dir = data_dir
      self.weight_dir = weight_dir
      self.output_dir = output_dir
      self.run_dir = run_dir
      self.model_config = model_config
      self.data_config = data_config
      self.model = YOLO(self.model_config)
   
    def download(self):
        # model = YOLO(r"ultralytics\cfg\models\v8\yolov8.yaml")  # build a new model from YAML
      
        assets = [f"yolov8{size}{suffix}.pt" for size in "lx" for suffix in ("", "-cls", "-seg")]
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
        for (img,label) in tqdm(zip(valid_list,label_list)):
            r = random.random()
            if r < 0.1:
                target = 'test'
            elif r < 0.3:
                target = 'val'
            else:
                target = 'train'
            shutil.move(os.path.join(self.data_dir,'left_images',img),os.path.join(self.data_dir,target,'images',img))
            shutil.move(os.path.join(self.data_dir,'annotations',label),os.path.join(self.data_dir,target,'labels',label))
            
    def train(self, epochs=100, imgsz=640, batch=32, pretrained=False):
        results = self.model.train(data=self.data_config, epochs=epochs, imgsz=imgsz, batch=batch, pretrained=pretrained)

    def eval(self,exp_name):
        self.model = YOLO((os.path.join(self.run_dir,exp_name,'weights','best.pt')))
        metrics = self.model.val(data=self.data_config,save_json=True,plots=True,save_txt=True,name=os.path.join('val',os.path.basename(exp_name)+f'_{time.strftime("%Y%m%d%H%M%S")}'))   
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list contains map50-95 of each category

    def inference(self,model,img_list,output_dir):
        self.model = YOLO((os.path.join(self.run_dir,model,'weights','best.pt')))
        img_listdir = []
        for img_name in img_list:
            if os.path.exists(os.path.join(self.data_dir,'left_images',str(img_name).zfill(6)+'.png')):
                img_listdir.append(os.path.join(self.data_dir,'left_images',str(img_name).zfill(6)+'.png'))
        # result_list = self.model.predict(img_listdir, save_dir=output_dir,save=True,stream=True)
        result_list = []
        for img in tqdm(img_listdir):
            result_list.append(self.model.predict(img,save_dir=output_dir,save=True))
        # results = self.model(img_listdir)
        # os.makedirs(os.path.join(output_dir,model),exist_ok=True)
        # for result,img_name in zip(results,img_list):
        #     result.save(filename=os.path.join(output_dir,model,str(img_name.zfill(6))+'.png'))
        #     result_list.append(result)s
        #     
        return result_list
  
if __name__ == '__main__':
    base_dir = r'D:\hd\dataset\poang01'
    data_dir = r'D:\hd\dataset\poang01'
    img_dir = os.path.join(data_dir,r'stereo\left_images')
    # img_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\stereo\left_images'
    xband_dir = os.path.join(data_dir,'X-band')
    lidar_dir = os.path.join(data_dir,r'lidar_front\points')
    calib_dir = os.path.join(base_dir,'calibration')
    label_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\labels\all\left'
    output_dir = r'D:\hd\ship_detection\results\pohang01\train5'
    model_trainer = Model_Trainer(r'D:\hd\dataset\poang01\stereo',r'.\weights',r'D:\hd\ship_detection\results\wo_train_v8',data_config='pohang_night.yaml')
    dataloader = DataGenerator(data_dir, img_dir, xband_dir,lidar_dir,calib_dir,label_dir)
    # model_trainer.split_dataset()
    # model_trainer.train(epochs=100)
    # result = model_trainer.inference_single(model_name = 'yolov8l',img_name='006210.png')
    # model_trainer.download()
    # Train
    # model = YOLO("yolov8l.yaml")
    # print(settings)
    # print(model.info())
    # model_trainer.eval('detect/train5')
    model_trainer.inference('detect/train5',list(range(300,2600)),output_dir)
    result_list = []
    for img in sorted(os.listdir(r'D:\hd\ship_detection\runs\detect\predict3')):
        if 'jpg' in img:
            result_list.append(cv2.imread(os.path.join(r'D:\hd\ship_detection\runs\detect\predict3',img)))
    dataloader.gen_video(result_list[:500],r'D:\hd\ship_detection\runs\detect\train5\video.avi')
    