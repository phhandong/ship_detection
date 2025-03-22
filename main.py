import os
import random
import shutil
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics import settings
from ultralytics.utils.downloads import attempt_download_asset
import  matplotlib.pyplot as PIL

class Train_Init():
    def __init__(self,data_dir):
      self.data_dir = data_dir
   
    def download():
        # model = YOLO(r"ultralytics\cfg\models\v8\yolov8.yaml")  # build a new model from YAML
      
        assets = [f"yolov8{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "-cls", "-seg")]
        for x in assets:
            attempt_download_asset(f"weights/{x}")

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
    # train_init = Train_Init(r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\stereo')
    # train_init.split_dataset()
  
  
  
    # model = YOLO(r"E:\Python_Project\Ship_Detection\yolo8\weights\yolov8l.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolov8l.yaml").load("weights/yolov8l.pt")  # build from YAML and transfer weights
    # print(settings)
    # print(model.info())
    # # # Train the model
    results = model.train(data="pohang.yaml", epochs=100, imgsz=640, batch=2)
    # results = model(r"E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\stereo\left_images\011346.png")
    # results[0].show()
    