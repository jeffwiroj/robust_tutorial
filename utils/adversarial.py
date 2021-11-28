from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

epsilons = [0.0001, .001, .005, 0.01, 0.1]
def fgsm(model, X, y, epsilon=0.01):
    """ Construct FGSM adversarial examples on the examples X"""
    model.zero_grad() # clears out previous gradient
    B = X.shape[0]
    y = y.view(B).long()
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()
    
def test_fgsm(model,test_loader,device,epsilon = 0.0001):
    model.to(device)
    model.eval()
    
    total = 0
    correct = 0
    
    for x,y in test_loader:
        B = x.shape[0]
        y = y.view(B).long()
        x,y = x.to(device),y.to(device)

        output = model(x)
        preds = torch.argmax(output,dim=1)
        
        correct_ind = torch.where(preds==y)
        noise = fgsm(model,x[correct_ind],y[correct_ind],epsilon)
        x[correct_ind] = x[correct_ind] + noise
        
        n_out = model(x)
        n_preds = torch.argmax(n_out,dim=1)
        correct += torch.sum(n_preds == y)
        total += B
        
    return total,correct.item()