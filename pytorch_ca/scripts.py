import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pylab as pl


# Utility functions

def RGBAtoFloat(images):
    """Converts images in 0-1 range"""
    return torch.clip(images.float() / 255, 0., 1.)

def FloattoRGBA(images):
    """Converts images in 0-255 range"""
    return torch.clip((images * 255), 0, 255).type(torch.uint8)

def FloattoRGB(images):
    """Converts a 0-1 float image with an alpha channel into RGB"""
    if len(images.size()) < 4:
        images = torch.unsqueeze(images, 0)
    return torch.clip(images[:,:3,:,:] * images[:,3,:,:] * 255 + (1-images[:,3,:,:])*255, 0, 255).type(torch.uint8)

def CenterCrop(images, size):
    """Center crops an image"""
    return T.CenterCrop(size)(images)

def Pad(images, padding):
    return T.Pad(padding//2)(images)

def imshow(image, center_crop=False):
    """Prints an image"""
    if center_crop:
        image = CenterCrop(image)
    pl.imshow(np.asarray(image.cpu().permute(1,2,0)[:,:,:4]))


def MakeSeed(n_images, n_channels, image_size):
    """Makes the seed to start the CA, i.e. a black dot"""
    start_point = torch.zeros((n_images, n_channels, image_size, image_size))
    start_point[:, 3, image_size//2, image_size//2] = 1.
    return start_point

# Sample pool dataloader
class SamplePool(Dataset):
    """Samples the training images"""
    def __init__(self, pool_size, n_channels, image_size):
        self.images = MakeSeed(pool_size, n_channels, image_size)
        self.size = pool_size
        self.n_channels = n_channels
        self.image_size = image_size
    
    
    def __len__(self):
        return self.size

    
    def __getitem__(self, idx):
        return self.images[idx], idx

    
    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, False)
        return self.images[idx], idx
    
    def update(self, new_images, idx):
        self.images[idx] = new_images
        
        
class Pool(Dataset):
    
    def __init__(self, pool_size, n_channels, image_size, transform = None):
        self.images = MakeSeed(pool_size, n_channels, image_size)
        self.size = pool_size
        self.n_channels = n_channels
        self.image_size = image_size
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.transform(self.images[index]), index
        
    def update(self, new_images, idx):
        self.images[idx] = new_images

# Loss avarage over time

def time_avarage(x):
    
    pass
    
    