#!/usr/bin/env python
import sys
sys.path.append('../')

from utils.dataset import  CustomDataset
import os
import warnings
warnings.filterwarnings("ignore")
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
from tqdm import tqdm
import threading
import random
# import rasterio
import os
import numpy as np
import sys
from sklearn.utils import shuffle as shuffle_lists
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

import torch

#default
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

#trasnform
from torchvision import transforms

#dataset
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchsummary import summary
from torchsampler import ImbalancedDatasetSampler


from utils import models 
#metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import torchmetrics.functional as tf

#numeric
import numpy as np
import pandas as pd 

#visualization
import matplotlib.pyplot as plt

#system 
from tqdm import tqdm
import os 
import wandb
import datetime


#default
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#trasnform
from torchvision import transforms

#dataset
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader

#model 
import torchvision.models as models
from network.models import get_pretrained_model
# from network.models import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from utils.models import *
import utils.loss as loss 

#metric
from sklearn.metrics import recall_score, f1_score, accuracy_score

from torchsampler import ImbalancedDatasetSampler


#numeric
import numpy as np
import pandas as pd 

#visualization
import matplotlib.pyplot as plt

#system 
from tqdm import tqdm
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import wandb

#parser
from utils.arg import save_args 


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.autograd.set_detect_anomaly(True)

class Train(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('=' * 100)
        print('=' * 100)
        print("\033[41mStart Initialization\033[0m")
        
        now = datetime.datetime.now()
        formatted_now = now.strftime("%Y%m%d%H%M")
        self.checkpoint_datetime = formatted_now
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\033[41mCUDA Status : {self.device.type}\033[0m")
        
        ########################## Data set & Data Loader ##############################
        # Data set & Data Loader
        self.train_csv = '../../dataset/train_meta.csv'
        
        self.train_transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            # transforms.RandomResizedCrop(size = (224,224), antialias= True),
            # transforms.RandomRotation((0, 30), interpolation=transforms.InterpolationMode.NEAREST),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.25, contrast=0.25),
            # transforms.ToTensor(),
        ])


        dataset = CustomDataset(
            csv_path= self.train_csv,
            # transform= self.train_transform,
            transform= None, #None으로 세팅
            MAX_PIXEL_VALUE= 65535,
            band = (7,6,2) #기존 세팅 
        )
        # 훈련 및 검증 세트 분할
        train_size = int(0.9 * len(dataset))
        valid_size = len(dataset) - train_size 
        self.train_dataset, self.valid_dataset = random_split(dataset, [train_size, valid_size])
        
        self.train_loader = DataLoader(
                                    dataset = self.train_dataset,
                                    batch_size = args.ts_batch_size,
                                    # shuffle = True,
                                    # sampler= ImbalancedDatasetSampler(
                                    #     train_dataset,
                                    #     labels = train_dataset.getlabels()
                                    # ),
                                    pin_memory= True,
                                    )
        self.valid_loader = DataLoader(
                                    dataset = self.valid_dataset,
                                    batch_size = args.vs_batch_size,
                                    shuffle = False,
                                    pin_memory= True,
                                    )
        
        ######################################### Wan DB #########################################
        # wandb 실행여부 yes/그외
        self.w = args.wandb
        if args.wandb == 'yes':
            wandb.init(
                project = 'satellite', 
                entity = 'dablro1232',
                notes = 'baseline',
                # epochs = args.epochs,
                config = args.__dict__,
            )
            name = args.model + f'_{args.version}' + f'_{args.training_date}'
            wandb.run.name = name #name으로 지정 
        else:
            name = args.model #없으면 랜덤으로 지정
        ######################################### Wan DB #########################################
        
        ######################################### Saving File #########################################
        # model save할 경로 설정
        self.save_path = os.path.join(args.save_path, f"{name}")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.arg_path = f"{self.save_path}/{name}.json" #인자 save할 경로 설정
        save_args(self.arg_path)
        ######################################### Saving File #########################################
        
        ############################## Model Initialization & GPU Setting ##############################
        if args.pretrain == 'yes': #pretrained model 사용여부
            pass
            # wandb_name = args.pretrained_model
            # PATH = f"./model/{wandb_name}/{wandb_name}.pt"
            # print(f"Previous model : {PATH} | \033[41mstatus : Pretrained Update\033[0m")
            
            # checkpoint = torch.load(PATH)
            # self.model.to(self.device)
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # self.epochs = checkpoint['epochs']
            # if args.error_signal == 'yes': #-> 에러뜬거면 yes로 지정하기
            #     self.epoch = checkpoint['epoch']
            # else:
            #     self.epoch = 0
            # self.lr = checkpoint['learning_rate']
            # self.loss = checkpoint['loss'].to(self.device)
            # self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler = optim.lr_scheduler.LambdaLR(optimizer = self.optimizer, lr_lambda=lambda epoch: 0.95 ** self.epochs)
            # self.name = checkpoint['model']
            # # self.best_loss = checkpoint['best_loss']
            # # self.t_loss_li = checkpoint['t_loss']
            # # self.v_loss_li = checkpoint['v_loss']
            # self.version = args.version
            # self.ts_batch = args.ts_batch_size
            # self.vs_batch = args.vs_batch_size
            # self.model_name = args.model    
            # self.model_save_path = f"{self.save_path}/{name}_{self.epoch}.pt"
            # self.best_loss = 1000000
            
        else:
            # random seed 고정
            random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            else:
                torch.manual_seed_all(42)
                
            # self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
            self.model = get_pretrained_model('manet').get()
            self.model.to(self.device)

            print(f"Training Model : {args.model} | status : \033[42mNEW\033[0m")

            ############################# Hyper Parameter Setting ################################
            # self.class_weights = torch.Tensor([0.09592459044406017, 0.3648678994488848, 0.539207510107055])    
            # self.loss = nn.CrossEntropyLoss(weight= self.class_weights).to(self.device)
            self.loss = nn.BCELoss().to(self.device)
            # self.loss = custom_loss.FocalLoss(alpha = 0.25, gamma = 2.0).to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr = args.learning_rate)
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda= lambda epoch: 0.95**epoch, last_epoch = -1, verbose = True)

            self.epochs = args.epochs
            self.epoch = 0
            self.ve = args.valid_epoch 
            self.lr = args.learning_rate
            self.name = name 
            self.ts_batch = args.ts_batch_size
            self.vs_batch = args.vs_batch_size
            self.version = args.version
            self.model_name = args.model    
            self.model_save_path = f"{self.save_path}/{self.model_name}.pt"
            self.best_loss = float('inf')
            
            ############################# Hyper Parameter Setting ################################
        self.early_stopping_epochs, self.early_stop_cnt = 30, 0
        ############################### Metrics Setting########################################
        self.metrics = {
            'tr_bce' : [],
            'vl_bce' : [],
            'tr_iou' : [],
            'vl_iou' : [],
        }

        ############################### Metrics Setting########################################
        
        # Training 
        print("\033[41mFinished Initalization\033[0m")
        print("\033[41mStart Training\033[0m")
        
        
    def checkpoint(self, SAVE_DIR, checkpoint_datetime, model, optimizer, lr, loss, metrics, epoch, epochs, images, masks, preds):
        # loss plot
        plt.figure(figsize=(10, 7))
        plt.subplot(121)
        plt.plot(metrics['tr_bce'], label='Train Loss')
        plt.plot(metrics['vl_bce'], label='Valid Loss')
        plt.title("BCE | DOWN GOOD")
        plt.legend()
        plt.subplot(122)
        plt.plot(metrics['tr_iou'], label='Train IoU')
        plt.plot(metrics['vl_iou'], label='Valid IoU')
        plt.title("mIOU | UP GOOD")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, checkpoint_datetime+f'_{epoch}_{epochs}_loss.png'))
        plt.close()
        
        # mask plot
        plt.figure(figsize=(10, 7))
        plt.subplot(131)
        plt.imshow(images[0].cpu().permute(1,2,0))
        plt.title('image')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(masks[0].cpu().permute(1,2,0))
        plt.title('mask')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(preds[0].cpu().permute(1,2,0))
        plt.title('pred')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, checkpoint_datetime+f'_{epoch}_{epochs}_mask.png'))
        plt.close()
        
        torch.save({
            "model" : f"{epoch}",
            "epoch" : epoch,
            "epochs" : epochs,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "learning_rate" : lr,
            "loss" : loss,
            "metric" : metrics,
            "description" : f"segmentation model training status : {epoch}/{epochs}"
        },
        os.path.join(SAVE_DIR, checkpoint_datetime+f'_{epoch}_{epochs}.pt'))
        print(f"#"*30, "model save", "#"*30)
        
    def calculate_iou(self, preds, masks, threshold=0.5):
        # 예측 마스크를 이진 형태로 변환

        preds = (preds > threshold).float()
        
        intersection = torch.sum(preds * masks)
        union = torch.sum((preds + masks) > 0)
        
        iou = (intersection + 1e-7) / (union + 1e-7)  # 0으로 나누는 경우를 방지하기 위해 작은 값(1e-7)을 추가
        
        return iou
    
    def fit(self):
        for epoch in tqdm(range(self.epoch, self.epochs)):
            train_losses, valid_losses = 0., 0.
            train_ious, valid_ious = 0., 0.
            
            self.model.train()
            for _, (images, masks) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                images, masks = images.to(self.device), masks.to(self.device)
                images = images.permute(0,3,1,2)
                masks = masks.permute(0,3,1,2)
                
                preds = self.model(images)
                train_loss = self.loss(preds, masks).to(self.device)
                train_iou = self.calculate_iou(preds, masks).cpu().detach().numpy()

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                
                train_losses += train_loss.item()
                train_ious += train_iou
            self.scheduler.step()

            self.metrics['tr_bce'].append(train_losses / len(self.train_loader))
            self.metrics['tr_iou'].append(train_ious / len(self.train_loader))
            print(f"Epoch : {epoch}/{self.epochs} | Train Loss : {train_losses / len(self.train_loader)}")
            print(f"Epoch : {epoch}/{self.epochs} | Train IOU : {train_ious / len(self.train_loader)}")
                
            lr = self.optimizer.param_groups[0]["lr"]
                
            if self.w == "yes":
                wandb.log({
                    "Learning_Rate" : lr,
                    "tr_bce" : self.metrics['tr_bce'][-1],
                    "tr_mIOU" : self.metrics['tr_iou'][-1],
                }, step = epoch)
                
        
            ################################# valid #################################
            with torch.no_grad():
                self.model.eval()
                for _, (images, masks) in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader)):
                    images, masks = images.to(self.device), masks.to(self.device)
                    images = images.permute(0,3,1,2)
                    masks = masks.permute(0,3,1,2)
                    
                    preds = self.model(images)
                    
                    valid_loss = self.loss(preds, masks).to(self.device)
                    valid_iou = self.calculate_iou(preds, masks).cpu().numpy()
                    
                    valid_losses += valid_loss.item()
                    valid_ious += valid_iou
            
            self.metrics['vl_bce'].append(valid_losses / len(self.valid_loader))
            self.metrics['vl_iou'].append(valid_ious / len(self.valid_loader))
            print(f"Epoch : {epoch}/{self.epochs} | Valid Loss : {valid_losses / len(self.valid_loader)}")
            print(f"Epoch : {epoch}/{self.epochs} | Valid IOU : {valid_ious / len(self.valid_loader)}")
                
            ## Display to Wandb for validation loss
            if self.w == "yes":
                wandb.log({
                    "vl_bce" : self.metrics['vl_bce'][-1],
                    "vl_mIOU" : self.metrics['vl_iou'][-1],
                }, step = epoch)
            
            # Early Sooping
            if valid_losses < self.best_loss:
                self.best_loss = valid_losses
                self.early_stop_cnt = 0
                self.checkpoint(self.save_path, self.checkpoint_datetime, self.model, self.optimizer, lr, self.loss, self.metrics, epoch, self.epochs, images, masks, preds)
            else:
                self.early_stop_cnt += 1
                if self.early_stop_cnt >= self.early_stopping_epochs:
                    print(f"Early Stops!!! : {epoch}/{self.epochs}")
                    self.checkpoint(self.save_path, self.checkpoint_datetime, self.model, self.optimizer, lr, self.loss, self.metrics, epoch, self.epochs, images, masks, preds)
                    break
            try:
                if valid_losses > np.array(self.metrics['vl_bce'])[-1].max():
                    self.checkpoint(self.save_path, self.checkpoint_datetime, self.model, self.optimizer, lr, self.loss, self.metrics, epoch, self.epochs, images, masks, preds)
            except Exception as e:
                print(e)
                pass 

        print("="*100)
        print(f"\033[41mFinished Training\033[0m | Save model PATH : {self.model_save_path}")
        if self.w == "yes":
            wandb.finish()
            
            
    