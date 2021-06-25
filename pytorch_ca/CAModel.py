import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode, write_video
import torchvision.transforms as T
import os.path
from importlib import reload

from random import randint

from scripts import *


class CAModel(nn.Module):
    """Neural cellular automata model"""
    def __init__(self, n_channels=16, device=None, fire_rate=0.5):
        super().__init__()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.n_channels = n_channels
        self.fire_rate = fire_rate
        
        self.layers = nn.Sequential(nn.Conv2d(n_channels*3, 128, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, n_channels, 1))
        
    
    def wrap_edges(self, x):
        return F.pad(x, (1,1,1,1), 'circular', 0)

    
    def get_living_mask(self, x):
        alpha = x[:, 3:4, :, :]
        return F.max_pool2d(self.wrap_edges(alpha), 3, stride=1) > 0.1


    def perceive(self, x, angle=0.):
        identity = torch.tensor([[0.,0.,0.],
                                 [0.,1.,0.],
                                 [0.,0.,0.]])
        dx = torch.tensor([[-0.125,0.,0.125],
                           [-0.25 ,0.,0.25 ],
                           [-0.125,0.,0.125]])
        dy = dx.T
        
        angle = torch.tensor(angle)
        c, s = torch.cos(angle), torch.sin(angle)
        dx, dy = c*dx - s*dy, s*dx + c*dy
        
        
        all_filters = torch.stack((identity, dx, dy))
        all_filters_batch = all_filters.repeat(self.n_channels,1,1).unsqueeze(1)
        all_filters_batch = all_filters_batch.to(self.device)
        return F.conv2d(self.wrap_edges(x), all_filters_batch, groups=self.n_channels)
    

    def forward(self, x, angle=0., step_size=1.):
        pre_life_mask = self.get_living_mask(x)
        
        dx = self.layers(self.perceive(x, angle)) * step_size
        update_mask = torch.rand(x[:,:1,:,:].size(), device=self.device) < self.fire_rate
        x += dx*update_mask.float()
        
        post_life_mask = self.get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask

        return x * life_mask.float()
    
    
    def makeVideo(self, n_iters, video_size, fname="video.mkv", rescaling=8, init_state=None, fps=10):
        if init_state is None:
            init_state = MakeSeed(1, self.n_channels, video_size)

        init_state = init_state.to(self.device)
        
        video_size *= rescaling
        video = torch.empty((n_iters, video_size, video_size, 3), device="cpu")
        rescaler = T.Resize((video_size, video_size), interpolation=T.InterpolationMode.NEAREST)
        with torch.no_grad():
            for i in range(n_iters):
                video[i] = FloattoRGB(rescaler(init_state))[0].permute(1,2,0).cpu()
                init_state = self.forward(init_state)
        
        write_video(fname, video, fps=fps)
        
    
    def evolve(self, x, iters, angle=0., step_size=1.):
        self.eval()
        with torch.no_grad():
            for i in range(iters):
                x = self.forward(x, angle=angle, step_size=step_size)
    
        return x
    

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
        
        
    def save(self, fname, overwrite=False):
        if os.path.exists(fname) and not overwrite:
            message = "The file name already exists, to overwrite it set the "
            message += "overwrite argument to True to confirm the overwrite"
            raise Exception(message)
        torch.save(self.state_dict(), fname)
        
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode, write_video
import torchvision.transforms as T
import os.path
from importlib import reload

from scripts import *



class CellularAutomata(nn.Module):
    """Neural cellular automata model"""
    def __init__(self, n_channels=16, device=None, fire_rate=0.5):
        super().__init__()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.n_channels = n_channels
        self.fire_rate = fire_rate
        
        self.layers = nn.Sequential(nn.Conv2d(n_channels*3, 128, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, n_channels, 1))
        
    
    def wrap_edges(self, x):
        return F.pad(x, (1,1,1,1), 'circular', 0)

    
    def get_living_mask(self, x):
        alpha = x[:, 3:4, :, :]
        return F.max_pool2d(self.wrap_edges(alpha), 3, stride=1) > 0.1


    def perceive(self, x, angle=0.):
        identity = torch.tensor([[0.,0.,0.],
                                 [0.,1.,0.],
                                 [0.,0.,0.]])
        dx = torch.tensor([[-0.125,0.,0.125],
                           [-0.25 ,0.,0.25 ],
                           [-0.125,0.,0.125]])
        dy = dx.T
        
        angle = torch.tensor(angle)
        c, s = torch.cos(angle), torch.sin(angle)
        dx, dy = c*dx - s*dy, s*dx + c*dy
        
        
        all_filters = torch.stack((identity, dx, dy))
        all_filters_batch = all_filters.repeat(self.n_channels,1,1).unsqueeze(1)
        all_filters_batch = all_filters_batch.to(self.device)
        return F.conv2d(self.wrap_edges(x), all_filters_batch, groups=self.n_channels)
    

    def forward(self, x, angle=0., step_size=1.):
        pre_life_mask = self.get_living_mask(x)
        
        dx = self.layers(self.perceive(x, angle)) * step_size
        update_mask = torch.rand(x[:,:1,:,:].size(), device=self.device) < self.fire_rate
        x += dx*update_mask.float()
        
        post_life_mask = self.get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask

        return x * life_mask.float()
    
    
    def makeVideo(self, n_iters, video_size, fname="video.mkv", rescaling=8, init_state=None, fps=10):
        if init_state is None:
            init_state = MakeSeed(1, self.n_channels, video_size)

        init_state = init_state.to(self.device)
        
        video_size *= rescaling
        video = torch.empty((n_iters, video_size, video_size, 3), device="cpu")
        rescaler = T.Resize((video_size, video_size), interpolation=T.InterpolationMode.NEAREST)
        with torch.no_grad():
            for i in range(n_iters):
                video[i] = FloattoRGB(rescaler(init_state))[0].permute(1,2,0).cpu()
                init_state = self.forward(init_state)
        
        write_video(fname, video, fps=fps)
        
    
    def evolve(self, x, iters, angle=0., step_size=1.):
        self.eval()
        with torch.no_grad():
            for i in range(iters):
                x = self.forward(x, angle=angle, step_size=step_size)
    
        return x
    

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
        
        
    def save(self, fname, overwrite=False):
        if os.path.exists(fname) and not overwrite:
            message = "The file name already exists, to overwrite it set the "
            message += "overwrite argument to True to confirm the overwrite"
            raise Exception(message)
        torch.save(self.state_dict(), fname)
        
    def to_train(self, pool, optimizer, criterion, device, cpu, scheduler = None, n_epochs = 50000, evolution_iters = 50, **kwargs):
        
        self.train()
        
        losses = []
        
        for i in range(n_epochs):

            inputs, indexes = pool.sample(kwargs['batch_size'])
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            for j in range(evolution_iters+randint(-10, 10)):
                inputs = self.forward(inputs)
            
            batch_avg_loss, batch_losses = criterion(inputs)
            batch_avg_loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(batch_avg_loss.item())
        
            inputs = inputs.to(cpu)
            # batch_losses = batch_losses.to(cpu)
        
            with torch.no_grad():
                if inputs.shape[0] > 1:
                    where_max_loss = torch.argmax(batch_losses)
                    pool.update(MakeSeed(1, kwargs['n_channels'], kwargs['image_size']), indexes[where_max_loss])
                    pool.update(inputs[np.where(indexes!=indexes[where_max_loss])], np.where(indexes!=indexes[where_max_loss]))
                else:
                    pool.update(inputs, indexes)
        
            if i % 100 == 0: print('it: ', i, ' --- ', 'batch avg loss: ', batch_avg_loss.item())
                
        return losses
            

'''
self.train()
        
        losses = []
        
        for i in range(n_epochs):
            
            data_loss = 0
            norm_factor = 0

            for inputs, indeces in dataloader:
                
                norm_factor += 1

                inputs = inputs.to(device)
                    
                optimizer.zero_grad()
                    
                for j in range(evolution_iters+randint(-10, 10)):
                    inputs = self.forward(inputs)
                    
                batch_avg_loss, batch_losses = criterion(inputs)
                data_loss += batch_avg_loss.item()
                
                batch_avg_loss.backward()
                optimizer.step()
                scheduler.step()
                
                inputs = inputs.to(cpu)
                
                with torch.no_grad():
                    if inputs.shape[0] > 1:
                        where_max_loss = torch.argmax(batch_losses)
                        datapool.update(MakeSeed(1, kwargs['n_channels'], kwargs['image_size']), indeces[where_max_loss])
                        datapool.update(inputs[np.where(indeces!=indeces[where_max_loss])], np.where(indeces!=indeces[where_max_loss]))
                    else:
                        datapool.update(inputs, indeces)
                
                # print('--- batch', norm_factor)    
            
            losses.append(data_loss/norm_factor)
            
            print('iteration: ', i, ' --- ', 'batch avg loss: ', data_loss/norm_factor, '(', norm_factor, ')')
                
        return losses
'''