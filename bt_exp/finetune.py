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
import os

from barlow_twin.bt import BarlowTwin
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(filename = ""):
    
    
    filename = "checkpoint.pth.tar" if len(filename) == 0  else filename
    checkpoint  = torch.load(f"results/checkpoints/{filename}",map_location=device)
    bt_ = BarlowTwin()
    bt_.load_state_dict(checkpoint['model_state_dict'])
    backbone = bt_.backbone
    model = nn.Sequential(backbone,nn.Flatten(),nn.Linear(512,9))
    return model


def get_config():
    parser = argparse.ArgumentParser(description='Supervised Training')
    parser.add_argument('--lr',default = 0.05,type = float, help = 'Learning Rate')
    parser.add_argument('--wd',default =  0.0001,type = float, help = 'Weight Decay')
    parser.add_argument('--epochs',default = 200, type = int)
    parser.add_argument('--filename',default = "", type = str, help = "checkpoint of bt_pretrain")
    parser.add_argument('--save_name',default = "best_bt.pth", type = str, help = "checkpoint of bt_pretrain")
    return parser.parse_args()




def val(model,criterion,val_loader):
    
    total_acc,total_loss = 0,0
    model.eval()
    total,correct = 0,0
    
    with torch.no_grad():
        for x,y in val_loader:
            B = y.size(0)

            x = x.to(device)
            y = y.to(device)
            y = y.view(B).long()

            logits = model(x)
            preds = torch.argmax(logits,1)

            loss = criterion(logits,y)
            total_loss += (loss.item()/len(val_loader))
            total += y.size(0)
            correct += (preds == y).sum().item()
            
    total_acc = correct/total
    
    return total_acc,total_loss
    
def train_n_val(model,optimizer,criterion,train_loader,val_loader,writer,config):
   
    epochs,lr = config["epochs"],config["lr"]
   
    
    best_acc,best_loss = 0,1000
    for epoch in range(epochs):
        train_acc,train_loss = 0,0
        total,correct = 0,0
        model.train()
        for x,y in train_loader:
            B = y.size(0)
            optimizer.zero_grad()
            
            x = x.to(device)
            y = y.to(device)
            y = y.view(B).long()
            
            logits = model(x)
            preds = torch.argmax(logits,1)
            
            loss = criterion(logits,y)
            train_loss += (loss.item()/len(train_loader))
            total += y.size(0)
            correct += (preds == y).sum().item()
            
            loss.backward()
            optimizer.step()
        
        
        train_acc = (correct/total)
        
        

                    
                    
        val_acc,val_loss = val(model,criterion,val_loader)
        
        if(val_acc > best_acc):
            print(f"Achieved New Best Acc: {val_acc}")
            best_acc = val_acc
            save_name = config["save_name"]
            torch.save(model.state_dict(), f"results/checkpoints/{save_name}")
        best_loss = min(val_loss,best_loss)
 
        print(f"Epoch: {epoch}, Train Acc: {train_acc},  Train Loss: {train_loss}")
        print(f"Epoch: {epoch}, Val Acc: {val_acc},  Val Loss: {val_loss}, BEST ACC:{best_acc}")
        
        writer.add_scalar('Loss/train',train_loss, epoch)
        writer.add_scalar('Loss/val',val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
    return best_acc,best_loss




def main():
    
    
    #Check if results folder and results/checkpoints folder exist else create one
    dirs = ["results","results/checkpoints"]
    for d in dirs:
        if(not os.path.isdir(d)):
            os.makedirs(d)
            print(f"Creating directory: {d}")
    
    
    config = vars(get_config())
    print(f"Current Config: {config}")
    writer = SummaryWriter(log_dir = "results/log_dir")
    
    #Save Hyperparameter values:
    writer.add_text("LR", str(config["lr"]))
    writer.add_text("WD", str(config["wd"]))
    
    dataset = get_dataset()
    train_loader =  DataLoader(dataset['train_set'], batch_size=512,shuffle = True, pin_memory = True,num_workers = 4)
    val_loader =  DataLoader(dataset['val_set'], batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)
    test_loader = DataLoader(dataset['test_set'], batch_size=512,shuffle = False, pin_memory = True,num_workers = 4)

    model = get_model()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= config["lr"],weight_decay = config["wd"],momentum=0.9)
    train_n_val(model,optimizer,criterion,train_loader,val_loader,writer,config)
    save_name = config["save_name"]
    model.load_state_dict(torch.load(f"results/checkpoints/{save_name}"))
    test_acc,test_loss = val(model,criterion,test_loader)
    
    print(f"Test Acc: {test_acc}")
    writer.add_text("Test Acc", f"{test_acc:.3f}")
    writer.close()
    
    
if __name__ == "__main__":
    main()
    
