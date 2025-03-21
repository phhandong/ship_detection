import numpy as np
import json
import os


class Constructdataset:
    def __init__(self,data_dir,label_dir):
      self.data_dir = data_dir
      self.label_dir = label_dir
      self.ship_dir = os.path.join(self.label_dir,'dynamic','image','ship')
      self.buoy_dir = os.path.join(self.label_dir,'dynamic','image','buoy')
    
    def process_concat_annotations(self,dir,label_dict):
        
        for id in os.listdir(dir):
          for frame in os.listdir(os.path.join(dir,id)):
            with open(os.path.join(dir,id,frame)) as f:
              for line in f:
                line = line.strip().split()
                # <obstacle type> <id> <x_center> <y_center> <x_width> <y_height>
            
            f_name = frame.split('.')[0]
            if f_name in label_dict.keys():
              label_dict[f_name].append(line)
            else:
              label_dict[f_name] = []
              label_dict[f_name].append(line)
        return label_dict
    
    def concat_all_annotations(self):
        label_dict = {}
        for dir in [self.buoy_dir,self.ship_dir]:
            label_dict = self.process_concat_annotations(dir,label_dict)
        with open(os.path.join(self.label_dir,'labels_dynamic.json'),'w') as f:
            json.dump(label_dict,f,indent=4)


if __name__ == '__main__':
    data_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\stereo\left_images'
    label_dir = r'E:\Python_Project\Ship_Detection\dataset\Pohang\poang00\labels'
    yolodata_process = Constructdataset(data_dir,label_dir)
    yolodata_process.concat_all_annotations()
      

            
      