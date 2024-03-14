#!/usr/bin/env python

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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\033[41mCUDA Status : {self.device.type}\033[0m")
        
        ########################## Data set & Data Loader ##############################
        # Data set & Data Loader
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(size = (224,224), antialias= True),
            transforms.RandomRotation((0, 30), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.ToTensor(),
        ])
        valid_transform = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])


        train_dataset = CustomDataset(root_dir = '../data/data_multi_sono/train', transform= train_transform)
        valid_dataset = CustomDataset(root_dir = '../data/data_multi_sono/valid', transform= valid_transform)

        self.train_loader = DataLoader(
                                    dataset = train_dataset,
                                    batch_size = args.ts_batch_size,
                                    # shuffle = True,
                                    sampler= ImbalancedDatasetSampler(
                                        train_dataset,
                                        labels = train_dataset.getlabels()
                                    ),
                                    pin_memory= True,
                                    )
        self.valid_loader = DataLoader(
                                    dataset = valid_dataset,
                                    batch_size = args.vs_batch_size,
                                    shuffle = False,
                                    pin_memory= True,
                                    )
        
        ######################################### Wan DB #########################################
        # wandb 실행여부 yes/그외
        self.w = args.wandb
        if args.wandb == 'yes':
            wandb.init(
                project = 'PCOS', 
                entity = 'dablro1232',
                notes = 'baseline',
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
            wandb_name = args.pretrained_model
            PATH = f"./model/{wandb_name}/{wandb_name}.pt"
            print(f"Previous model : {PATH} | \033[41mstatus : Pretrained Update\033[0m")
            
            checkpoint = torch.load(PATH)
            
            self.model = pretrained_convnext_multi()
            self.model.to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.epochs = checkpoint['epochs']
            if args.error_signal == 'yes': #-> 에러뜬거면 yes로 지정하기
                self.epoch = checkpoint['epoch']
            else:
                self.epoch = 0
            self.lr = checkpoint['learning_rate']
            self.loss = checkpoint['loss'].to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer = self.optimizer, lr_lambda=lambda epoch: 0.95 ** self.epochs)
            self.name = checkpoint['model']
            # self.best_loss = checkpoint['best_loss']
            # self.t_loss_li = checkpoint['t_loss']
            # self.v_loss_li = checkpoint['v_loss']
            self.version = args.version
            self.ts_batch = args.ts_batch_size
            self.vs_batch = args.vs_batch_size
            self.model_name = args.model    
            self.model_save_path = f"{self.save_path}/{name}_{self.epoch}.pt"
            self.best_loss = 1000000
            
        else:
            self.model = pretrained_convnext_multi()
            self.model.to(self.device)

            print(f"Training Model : {args.model} | status : \033[42mNEW\033[0m")

            ############################# Hyper Parameter Setting ################################
            # self.class_weights = torch.Tensor([0.09592459044406017, 0.3648678994488848, 0.539207510107055])    
            # self.loss = nn.CrossEntropyLoss(weight= self.class_weights).to(self.device)
            self.loss = nn.CrossEntropyLoss().to(self.device)
            # self.loss = custom_loss.FocalLoss(alpha = 0.25, gamma = 2.0).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr = args.learning_rate)
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer = self.optimizer,
                                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                                        last_epoch = -1,
                                                        verbose= True)
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
            'train_loss' : [],
            'valid_loss' : [],
            'train_accuracy' : [],
            'train_f1' : [],
            'train_recall' : [],
            'valid_accuracy' : [],
            'valid_f1' : [],
            'valid_recall' : [],
        }

        ############################### Metrics Setting########################################
        
        # Training 
        print("\033[41mFinished Initalization\033[0m")
        print("\033[41mStart Training\033[0m")
    
    def fit(self):
        for epoch in tqdm(range(self.epoch, self.epochs)):
            train_losses, valid_losses = 0., 0.
            train_target, train_pred, valid_target, valid_pred = [], [], [], [] 
            
            self.model.train()
            for _, (inputs, labels) in tqdm(enumerate(self.train_loader)):
                self.optimizer.zero_grad()
                inputs, labels = inputs.to(self.device), labels.type(torch.LongTensor).to(self.device)
                outputs = self.model(inputs)
                
                train_loss = self.loss(outputs, labels)
                train_loss.backward()
                self.optimizer.step()
                train_losses += train_loss.item()
                
                # 예측 값을 이진 레이블로 변환
                pred = torch.argmax(outputs, dim = 1)
                train_target.extend(labels.detach().cpu().numpy())
                train_pred.extend(pred.detach().cpu().numpy())
                
            self.metrics['train_loss'].append(train_losses/len(self.train_loader))
            self.metrics['train_accuracy'].append(accuracy_score(train_target, train_pred))
            self.metrics['train_f1'].append(f1_score(train_target, train_pred, average= 'weighted'))
            self.metrics['train_recall'].append(recall_score(train_target, train_pred, average= 'weighted'))
            
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
                
            if self.w == "yes":
                wandb.log({
                    "Learning_Rate" : lr,
                    "train_LOSS" : self.metrics['train_loss'][-1],
                    "train_ACC" : self.metrics['train_accuracy'][-1],
                    "train_F1Score" : self.metrics['train_f1'][-1],
                    "train_RECALL" : self.metrics['train_recall'][-1],
                }, step = epoch)
                
        
            ################################# valid #################################
            with torch.no_grad():
                self.model.eval()
                if epoch % 1 == 0: 
                    for _ , (inputs, labels) in enumerate(self.valid_loader):
                        inputs, labels = inputs.to(self.device), labels.type(torch.LongTensor).to(self.device)
                        outputs = self.model(inputs)
                        
                        valid_loss = self.loss(outputs, labels)
                        valid_losses += valid_loss.item()

                        # 예측 값을 이진 레이블로 변환
                        pred = torch.argmax(outputs, dim = 1)
                        valid_target.extend(labels.detach().cpu().numpy())
                        valid_pred.extend(pred.detach().cpu().numpy())
                        
                    self.metrics['valid_loss'].append(valid_losses/len(self.valid_loader))
                    self.metrics['valid_accuracy'].append(accuracy_score(valid_target, valid_pred))
                    self.metrics['valid_f1'].append(f1_score(valid_target, valid_pred, average= 'weighted'))
                    self.metrics['valid_recall'].append(recall_score(valid_target, valid_pred, average= 'weighted'))
                    
                    
            print("#"*100)    
            print(f"LOSS : {self.metrics['train_loss'][-1]} | {self.metrics['valid_loss'][-1]}\n ACC : {self.metrics['train_accuracy'][-1]} | {self.metrics['valid_accuracy'][-1]}\n F1 : {self.metrics['train_f1'][-1]} | {self.metrics['valid_f1'][-1]}\n RECALL : {self.metrics['train_recall'][-1]} | {self.metrics['valid_recall'][-1]}")
            print("#"*100)
                    
            ## Display to Wandb for validation loss
            if self.w == "yes":
                wandb.log({
                    "valid_LOSS" : self.metrics['valid_loss'][-1],
                    "valid_ACC" : self.metrics['valid_accuracy'][-1],
                    "valid_F1Score" : self.metrics['valid_f1'][-1],
                    "valid_RECALL" : self.metrics['valid_recall'][-1],
                }, step = epoch)
            
            # Early Sooping
            if self.metrics['valid_loss'][-1] > self.best_loss:
                self.early_stop_cnt += 1
            else:
                self.best_loss = self.metrics['valid_loss'][-1]
                self.early_stop_cnt = 0

            # #valid loss가 valid_loss 이전까지 Loss보다 작으면 저장하게 만듬.
            if len(self.metrics['valid_loss']) > 1 : #valid loss 리스트가 값이 잇을때
                min_valid_loss = np.min(np.array(self.metrics['valid_loss'][:-1]))
                if self.metrics['valid_loss'][-1] < min_valid_loss:
                    torch.save({
                        "model" : f"{self.model_name}" + f"{self.version}_{epoch}",
                        "epoch" : epoch,
                        "epochs" : self.epochs,
                        "model_state_dict" : self.model.state_dict(),
                        "optimizer_state_dict" : self.optimizer.state_dict(),
                        "learning_rate" : lr,
                        "loss" : self.loss,
                        "best_loss" : self.best_loss,
                        "metric" : self.metrics,
                        "description" : f"Training status : {epoch}/{self.epochs}"
                    },
                    f"{self.save_path}/{self.name}_{epoch}.pt")
                    
                    print(f"BEST VALID LOSS : {self.metrics['valid_loss'][-1]} | SAVE MODEL PATH : {self.save_path}/{self.name}_{epoch}")

            # 조기 종료 조건 확인
            if self.early_stop_cnt >= self.early_stopping_epochs:
                print(f"Early Stops!!! : {epoch}/{self.epochs}")
                torch.save({
                    "model" : f"{self.model_name}" + f"{self.version}_{epoch}",
                    "epoch" : epoch,
                    "epochs" : self.epochs,
                    "model_state_dict" : self.model.state_dict(),
                    "optimizer_state_dict" : self.optimizer.state_dict(),
                    "learning_rate" : lr,
                    "loss" : self.loss,
                    "metric" : self.metrics,
                    "description" : f"Training status : {epoch}/{self.epochs}"
                },
                f"{self.save_path}/{self.name}_{epoch}_e.pt")
                
                print(f"SAVE MODEL PATH : {self.save_path}/{self.name}_{epoch}_e.pt")
                break

            elif epoch % 10 == 0 or epoch +1 == self.epochs:
                torch.save({
                    "model" : f"{self.model_name}" + f"{self.version}_{epoch}",
                    "epoch" : epoch,
                    "epochs" : self.epochs,
                    "model_state_dict" : self.model.state_dict(),
                    "optimizer_state_dict" : self.optimizer.state_dict(),
                    "learning_rate" : lr,
                    "loss" : self.loss,
                    "metric" : self.metrics,
                    "description" : f"Training status : {epoch}/{self.epochs}"
                },
                f"{self.save_path}/{self.name}_{epoch}.pt")
                
                print(f"PER {epoch} | SAVE MODEL PATH : {self.save_path}/{self.name}_{epoch}.pt")
                    

        print("="*100)
        print(f"\033[41mFinished Training\033[0m | Save model PATH : {self.model_save_path}")
        if self.w == "yes":
            wandb.finish()