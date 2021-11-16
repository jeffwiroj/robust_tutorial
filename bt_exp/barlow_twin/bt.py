import torch
import torch.nn as nn
import torchvision.models as models




class BarlowTwin(nn.Module):
    '''
    Implementation of Barlow Twins
    Default: Resnet34 backbone with projector dim = 8912
    '''
    
    def __init__(self,backbone = None, num_fts = 512,out_dims = 8912):
        
        super(BarlowTwin, self).__init__()
        self.backbone = backbone if backbone != None else get_backbone()
        self.projector = projector(num_fts,out_dims)
        self.bn1d = nn.BatchNorm1d(out_dims, affine=False)
    def forward(self, img1, img2):
        
        B = img1.shape[0]
        y1,y2 = self.backbone(img1).view(B,-1),self.backbone(img2).view(B,-1)
        z1,z2 = self.bn1d(self.projector(y1)),self.bn1d(self.projector(y2))
        
        return z1,z2

    
def get_backbone():
    '''
    Returns the Resnet34 model without the fc layer
    '''
    network = models.resnet34(False) #not pretrained
    backbone = torch.nn.Sequential(*(list(network.children())[:-1]))
    return backbone

def projector(in_dims = 512, out_dims = 8912):
    '''
    Returns the projector network, which comprises 3 linear layers
    The first two linear layers are followed by bn and relu
    '''
    l1 = nn.Sequential(nn.Linear(in_dims, out_dims,bias=False),
                       nn.BatchNorm1d(out_dims),
                       nn.ReLU(inplace=True))
    l2 = nn.Sequential(nn.Linear(out_dims, out_dims,bias = False),
           nn.BatchNorm1d(out_dims),
           nn.ReLU(inplace=True))
    l3 = nn.Linear(out_dims, out_dims,bias = False)
    
    return nn.Sequential(l1, l2, l3)

#Loss modified from https://github.com/facebookresearch/barlowtwins/blob/21149b45bda50e579f166a4e261f281924b7c208/main.py#L180   
class BTLoss(torch.nn.Module):

    
    def __init__(self,l_param = 5e-3):
        super(BTLoss,self).__init__()
        self.l_param = l_param
    def forward(self, 
                z1: torch.Tensor, 
                z2: torch.Tensor):
        
        B,D = z1.shape # [batch_size , dimension]
        c = (z1.T@z2).div_(B)
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        loss = on_diag + self.l_param * off_diag

        return loss
    
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

if __name__ == "__main__":
    backbone = get_backbone()
    bt = BarlowTwin(backbone = backbone)
    img1,img2 = torch.rand(2,3,28,28),torch.rand(2,3,28,28)
    
    
    out1,out2 = bt(img1,img2)
    criterion = BTLoss()
    loss = criterion(out1,out2)
    print(f"Img shape: {img1.shape}, output shape: {out1[0].shape}, loss: {loss.item()}")
