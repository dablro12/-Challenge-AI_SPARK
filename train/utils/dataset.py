from torch.utils.data import Dataset
from torchvision import transforms
import torch 
import os 
from PIL import Image
import numpy as np
import pandas as pd 
import rasterio

class CustomDataset(Dataset):
    def __init__(self, csv_path, transform = None, MAX_PIXEL_VALUE = 65535, band = None): #transform 가지고올거있으면 가지고 오기
        self.MAX_PIXEL_VALUE = MAX_PIXEL_VALUE 
        self.band = band
        self.transform = transform
        self.image_paths, self.mask_paths = self.csv_load(csv_path = csv_path, type = csv_path.split('/')[-1].split('_')[0])
    
    def csv_load(self, csv_path, type):
        self.type = type 
        df = pd.read_csv(csv_path)
        # os.pathconf('../')
        image_paths = f'../../dataset/{type}_img/' + df[f'{type}_img']
        mask_paths = f'../../dataset/{type}_mask/' + df[f'{type}_mask']
        
        return  image_paths, mask_paths
        
    def get_img_arr(self, path):
        # img = rasterio.open(path).read().transpose((1, 2, 0))
        img = rasterio.open(path).read(self.band).transpose((1, 2, 0))
        img = np.float32(img)/self.MAX_PIXEL_VALUE

        return img
    
    def get_mask_arr(self, path):
        img = rasterio.open(path).read().transpose((1, 2, 0))
        seg = np.float32(img)
        return seg
    
    def __len__(self):
        return len(self.image_paths) 
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        images = self.get_img_arr(path = image_path)
        masks = self.get_mask_arr(path = mask_path)
        
        # # transform 
        if self.transform:
            images = self.transform(images)
            masks = self.transform(masks)
        
        return images, masks