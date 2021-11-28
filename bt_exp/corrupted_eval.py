import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.models as models
from dataset import pathDataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../')
import math
import os
from barlow_twin.bt import BarlowTwin
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils.transformations import get_transformations
from finetune import val
import torchvision.transforms as T
import random
from utils.adversarial import epsilons,test_fgsm
def get_model(filename = ""):
    
    
    filename = "best_bt.pth.tar" if len(filename) == 0  else filename
    checkpoint  = torch.load(f"results/checkpoints/{filename}",map_location=device)
    bt_ = BarlowTwin()
    backbone = bt_.backbone
    model = nn.Sequential(backbone,nn.Flatten(),nn.Linear(512,9))
    model.load_state_dict(checkpoint)
    print(f"Loaded model from: {filename}")
    return model

def get_config():
    parser = argparse.ArgumentParser(description='Supervised Training')
    parser.add_argument('--method',default = "blur",type = str, 
                     help = 'corruption method: blur, bright, noise')
    parser.add_argument('--filename',default = "best_bt.pth",type = str, 
                     help = 'name of checkpoint file')
    return parser.parse_args()

def main():
    config = get_config()
    print(config)
    model = get_model(config.filename)
    model.eval()
    if(config.method != "fgsm"):corrupt_trans = get_transformations[config.method]
    if(config.method == "blur"):
        blurs = [corrupt_trans(3,i/10) for i in range(2,12,2)]
        for blur in blurs:
            torch.manual_seed(0)
            random.seed(0)
            transforms = T.Compose([T.ToPILImage(), T.ToTensor(), 
            T.Normalize(mean = [0.7405, 0.5330, 0.7058],std = [0.1237, 0.1768, 0.1244]),blur])
            
            dataset = pathDataset(root_dir = "../data", split = "test", transform = transforms)
            dataloader = DataLoader(dataset, batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)
            
            criterion = nn.CrossEntropyLoss()
            test_acc,test_loss = val(model,criterion,dataloader)
            print(f"{blur} acc:{test_acc}, loss:{test_loss}")
    
            with open('corrupt_result.txt', 'a') as f:
                f.write(f"{blur} acc:{test_acc}, loss:{test_loss}")
                
    if(config.method == "noise"):
        strengths = [0.05,0.07,0.09,0.13,0.15]
        noises = [corrupt_trans(0,i) for i in strengths]
        for noise in noises:
            torch.manual_seed(0)
            random.seed(0)
            transforms = T.Compose([T.ToPILImage(), T.ToTensor(), 
            T.Normalize(mean = [0.7405, 0.5330, 0.7058],std = [0.1237, 0.1768, 0.1244]),noise])
            
            dataset = pathDataset(root_dir = "../data", split = "test", transform = transforms)
            dataloader = DataLoader(dataset, batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)
            
            criterion = nn.CrossEntropyLoss()
            test_acc,test_loss = val(model,criterion,dataloader)
            print(f"{noise} acc:{test_acc}, loss:{test_loss}")
    
            with open('corrupt_result.txt', 'a') as f:
                f.write(f"{noise} acc:{test_acc}, loss:{test_loss}\n")
    
    
    if(config.method == "bright" or config.method == "shot"):
        noises = [corrupt_trans(i) for i in range(1,6)]
        for noise in noises:
            torch.manual_seed(0)
            random.seed(0)
            transforms = T.Compose([T.ToPILImage(),noise ,T.ToTensor(), 
            T.Normalize(mean = [0.7405, 0.5330, 0.7058],std = [0.1237, 0.1768, 0.1244])])
            
            dataset = pathDataset(root_dir = "../data", split = "test", transform = transforms)
            dataloader = DataLoader(dataset, batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)
            
            criterion = nn.CrossEntropyLoss()
            test_acc,test_loss = val(model,criterion,dataloader)
            print(f"{noise} acc:{test_acc}, loss:{test_loss}")
    
            with open('corrupt_result.txt', 'a') as f:
                f.write(f"{noise} acc:{test_acc}, loss:{test_loss}\n")
    if(config.method == "fgsm"):
        for epsilon in epsilons:
            torch.manual_seed(0)
            random.seed(0)
            transforms = T.Compose([T.ToPILImage(),T.ToTensor(), 
            T.Normalize(mean = [0.7405, 0.5330, 0.7058],std = [0.1237, 0.1768, 0.1244])])
            dataset = pathDataset(root_dir = "../data", split = "test", transform = transforms)
            dataloader = DataLoader(dataset, batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)
   
            total,correct = test_fgsm(model,dataloader,device,epsilon)
            
            print(f"FGSM epsilon:{epsilon} acc:{correct/total}")
    
            with open('corrupt_result.txt', 'a') as f:
                f.write(f"fgsm epsilon:{epsilon} acc:{correct/total}\n")
    

if __name__ == "__main__":
    main()
    