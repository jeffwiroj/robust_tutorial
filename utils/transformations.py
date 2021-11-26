import torchvision.transforms as T
import torch
import numpy as np
import skimage.color
import skimage as sk

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
# Taken from 
# https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
def brightness(x, severity=1):
    c = [.01, .02, .03, .07,1][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255
class Brightness():
    '''
    args: severity the factor of brightness
        type: int
    Can only be applied to numpy images with (c x h x w)
    '''
    def __init__(self,severity = 1):
        self.severity = severity
    def __call__(self,image):
        image = brightness(image,self.severity)
        return image.astype(np.uint8)


def shot_noise(x, severity=1):
    c = [120,100,80,60,40][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255

class ShotNoise():
    '''
    args: severity the factor of brightness
        type: int
    Can only be applied to numpy images with (c x h x w)
    '''
    def __init__(self,severity = 1):
        self.severity = severity
    def __call__(self,image):
        image = shot_noise(image,self.severity)
        return image.astype(np.uint8)
    
get_transformations = {"original": lambda x: x,
                  "blur": lambda x,y: T.GaussianBlur(kernel_size=(x,x), sigma=(y)),
                  "bright": lambda x: Brightness(x),
                  "noise": lambda x,y: AddGaussianNoise(x,y),
                  "shot": lambda x: ShotNoise(x)}
