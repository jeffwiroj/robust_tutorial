import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.models as models
from dataset import pathDataset
from torch.utils.data import Dataset, DataLoader
import argparse
import random
import numpy as np
np.random.seed(0)
import math
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sys
sys.path.append('../../')
from utils.transformations import get_transformations
from train import val
import torchvision.transforms as T

def get_model():
    model = models.resnet34(False)
    model.fc = nn.Linear(512,9)
    model.load_state_dict(torch.load(f"results/checkpoints/best_scratch.pth",map_location=device))
    return model

def get_config():
    parser = argparse.ArgumentParser(description='Supervised Training')
    parser.add_argument('--method',default = "blur",type = str, 
                     help = 'corruption method: blur, bright, noise')
    return parser.parse_args()


def main():
    config = get_config()
    print(config)
    model = get_model()
    
    corrupt_trans = get_transformations[config.method]
    if(config.method == "blur"):
        blurs = [corrupt_trans(3,i/10) for i in range(2,12,2)]
        for blur in blurs:
            torch.manual_seed(0)
            random.seed(0)
            transforms = T.Compose([T.ToPILImage(), T.ToTensor(), 
            T.Normalize(mean = [0.7405, 0.5330, 0.7058],std = [0.1237, 0.1768, 0.1244]),blur])
            
            dataset = pathDataset(root_dir = "../../data", split = "test", transform = transforms)
            dataloader = DataLoader(dataset, batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)
            
            criterion = nn.CrossEntropyLoss()
            test_acc,test_loss = val(model,criterion,dataloader)
            print(f"{blur} acc:{test_acc}, loss:{test_loss}")
    
            with open('corrupt_result.txt', 'a') as f:
                f.write(f"{blur} acc:{test_acc}, loss:{test_loss}\n")
                
    if(config.method == "noise"):
        strengths = [0.05,0.07,0.09,0.13,0.15]
        noises = [corrupt_trans(0,i) for i in strengths]
        for noise in noises:
            torch.manual_seed(0)
            random.seed(0)
            transforms = T.Compose([T.ToPILImage(), T.ToTensor(), 
            T.Normalize(mean = [0.7405, 0.5330, 0.7058],std = [0.1237, 0.1768, 0.1244]),noise])
            
            dataset = pathDataset(root_dir = "../../data", split = "test", transform = transforms)
            dataloader = DataLoader(dataset, batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)
            
            criterion = nn.CrossEntropyLoss()
            test_acc,test_loss = val(model,criterion,dataloader)
            print(f"{noise} acc:{test_acc}, loss:{test_loss}")
    
            with open('corrupt_result.txt', 'a') as f:
                f.write(f"{noise} acc:{test_acc}, loss:{test_loss}\n")

    if(config.method == "shot" or config.method == "bright"):
        
        noises = [corrupt_trans(i) for i in  range(1,6)]
        for noise in noises:
            torch.manual_seed(0)
            random.seed(0)
            transforms = T.Compose([T.ToPILImage(),noise, T.ToTensor(), 
            T.Normalize(mean = [0.7405, 0.5330, 0.7058],std = [0.1237, 0.1768, 0.1244])])
            
            dataset = pathDataset(root_dir = "../../data", split = "test", transform = transforms)
            dataloader = DataLoader(dataset, batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)
            
            criterion = nn.CrossEntropyLoss()
            test_acc,test_loss = val(model,criterion,dataloader)
            print(f"{noise} acc:{test_acc}, loss:{test_loss}")
    
            with open('corrupt_result.txt', 'a') as f:
                f.write(f"{noise} acc:{test_acc}, loss:{test_loss}\n")
                
if __name__ == "__main__":
    main()
    