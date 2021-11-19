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


def get_model():
    bt_ = BarlowTwin()
    backbone = bt_.backbone
    model = nn.Sequential(backbone,nn.Flatten(),nn.Linear(512,9))
    model.load_state_dict(torch.load(f"results/checkpoints/best_bt.pth",map_location=device))
    return model


def get_config():
    parser = argparse.ArgumentParser(description='Supervised Training')
    parser.add_argument('--method',default = "blur",type = str, 
                     help = 'corruption method: blur, bright, noise, all')
    return parser.parse_args()

def main():
    config = get_config()
    print(config)
    model = get_model()
    
    corrupt_trans = get_transformations[config.method]
    if(config.method == "blur" or config.method == "all"):
        blurs = [corrupt_trans(3,i/10) for i in range(1,12)]
        for blur in blurs:
            
            transforms = T.Compose([T.ToPILImage(), blur , T.ToTensor(), 
            T.Normalize(mean = [0.7405, 0.5330, 0.7058],std = [0.1237, 0.1768, 0.1244])])
            
            dataset = pathDataset(root_dir = "../data", split = "test", transform = transforms)
            dataloader = DataLoader(dataset, batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)
            
            criterion = nn.CrossEntropyLoss()
            test_acc,test_loss = val(model,criterion,dataloader)
            print(f"{blur} acc:{test_acc}, loss:{test_loss}")
    
            with open('corrupt_result.txt', 'a') as f:
                f.write(f"{blur} acc:{test_acc}, loss:{test_loss}")
    
    

if __name__ == "__main__":
    main()
    