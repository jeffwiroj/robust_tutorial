import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.models as models
from dataset import get_dataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../')
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os
import shutil
from barlow_twin.bt import BarlowTwin,BTLoss

def get_config():
    
    parser = argparse.ArgumentParser(description='Supervised Training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--batch_size',default = 512,type = int, help = 'Batch Size')
    parser.add_argument('--base_lr',default = 0.05,type = int, help = 'base learning rate')
    parser.add_argument('--wd',default =1e-5,type = float, help = 'Weight Decay')
    parser.add_argument('--lambda',default = 5e-3,type = float, help = 'BT Lambda Parameter')
    parser.add_argument('--epochs',default = 1000, type = int)

    args = vars(parser.parse_args())
    args["init_lr"] = (args["batch_size"]/256)*args["base_lr"]
    return args

def save_checkpoint(model,optimizer, is_best, epoch,loss,filename='checkpoint.pth.tar'):

    torch.save({
            'epoch': epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':loss
            }, f"results/checkpoints/{filename}")
    
    if is_best:
        best_file = 'model_best.pth.tar'
        shutil.copyfile(f"results/checkpoints/{filename}", f"results/checkpoints/{best_file}")
        

def train(model,criterion,optimizer,data_loader,writer,args):
    '''
    Trains BarlowTwin Model
    Criterion: BT Loss, see paper
    '''
    
    epochs,init_lr = args["epochs"],args["init_lr"]
    
    filename = 'checkpoint.pth.tar'
    
    model.train()
    
    best_loss = 100000
    start_epoch = 0
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        is_best = False
        for img,_ in data_loader:
            x1,x2 = img

            x1 = x1.to(device)
            x2 = x2.to(device)

            out1,out2 = model(x1,x2)
            
            
            loss = criterion(out1,out2)
            epoch_loss += (loss.item() / len(data_loader))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if(epoch_loss < best_loss): 
            is_best = True
            best_loss = epoch_loss
            
        save_checkpoint(model,optimizer, is_best, epoch,epoch_loss,filename=filename)
        
        if(epoch in [200,400,600,800]):
            save_checkpoint(model,optimizer,False,epoch,epoch_loss,f"checkpoint_ep_{epoch}.pth.tar")
        
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
        writer.add_scalar('Loss',epoch_loss, epoch)
        
def main():
    
    #Check if results folder and results/checkpoints folder exist else create one
    dirs = ["results","results/checkpoints"]
    for d in dirs:
        if(not os.path.isdir(d)):
            os.makedirs(d)
            print(f"Creating directory: {d}")
    config = get_config()
    
    print(f"Current Configuration: {config}")
    
    
    writer = SummaryWriter(log_dir = "results/log_dir")
   
    #Save Hyperparameter values:
    writer.add_text("WD", str(config["wd"]))
    dataset = get_dataset()
    ssl_loader =  DataLoader(dataset['unlabel_set'], batch_size=512,shuffle = True, pin_memory = True,num_workers = 4)
    model = BarlowTwin()
    model = model.to(device)
    criterion = BTLoss(l_param = config["lambda"])
    optimizer = optim.SGD(model.parameters(), lr = config["init_lr"],weight_decay = config["wd"],momentum=0.9)

    train(model,criterion,optimizer,ssl_loader,writer,config)
    writer.close()
if __name__ == "__main__":
    main()
