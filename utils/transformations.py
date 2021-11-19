import torchvision.transforms as T
import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

    
get_transformations = {"original": lambda x: x,
                  "blur": lambda x,y: T.GaussianBlur(kernel_size=(x,x), sigma=(y)),
                  "bright": lambda x: T.ColorJitter(brightness = (x,x)),
                  "noise": lambda x,y: AddGaussianNoise(x,y)}
