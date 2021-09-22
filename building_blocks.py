import torch
import torch.nn as nn
import numpy as np
import torchvision   
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import time
import matplotlib.pyplot as plt

class RDB(nn.Module):
    def __init__(self, channel_size, out_size, res_scaling_factor, stride=1, padding=1):
        super().__init__()
    
        self.conv1 = nn.Conv2d(channel_size, out_size, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(channel_size+out_size, out_size, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.lrelu2 = nn.LeakyReLU(0.2)
        
        self.conv3 = nn.Conv2d(channel_size+2*out_size, out_size, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight)
        self.lrelu3 = nn.LeakyReLU(0.2)
        
        self.conv4 = nn.Conv2d(channel_size+3*out_size, out_size, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight)
        self.lrelu4 = nn.LeakyReLU(0.2)
        
        self.conv5 = nn.Conv2d(channel_size+4*out_size, channel_size, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight)

        self.res_scaling_factor = res_scaling_factor


    def forward(self, x):
        out1 = self.lrelu1(self.conv1(x))
        out2 = self.lrelu2(self.conv2(torch.cat((x, out1), dim=1)))
        out3 = self.lrelu3(self.conv3(torch.cat((x, out1, out2), dim=1)))
        out4 = self.lrelu4(self.conv4(torch.cat((x, out1, out2, out3), dim=1)))
        out = self.conv5(torch.cat((x, out1, out2, out3, out4), dim=1))

        out = x + out*self.res_scaling_factor

        return out    


class RRDB(nn.Module):
    def __init__(self, channel_size, out_size, res_scaling_factor):
        super(RRDB, self).__init__()
        self.RDB1 = RDB(channel_size, out_size, res_scaling_factor)
        self.RDB2 = RDB(channel_size, out_size, res_scaling_factor)
        self.RDB3 = RDB(channel_size, out_size, res_scaling_factor)
        
        self.res_scaling_factor = res_scaling_factor

    def forward(self, x):
        out = self.RDB3(self.RDB2(self.RDB1(x)))
        out = x + out*self.res_scaling_factor
        return out
        
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class MyPatchDataset(Dataset):
    def __init__(self, filepath, patch_size):
        self.filepath = filepath
        self.file_list = os.listdir(filepath)
        self.patch_size = patch_size
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_HR = torchvision.transforms.Resize((1000,2000)) (torchvision.transforms.ToTensor()(Image.open(self.filepath+self.file_list[index])))
        img_LR = torchvision.transforms.Resize((250,500)) (torchvision.transforms.ToTensor()(Image.open(self.filepath[:-3]+'LR_bicubic/X4/'+self.file_list[index][:-4]+'x4.png')))

        patch_size = self.patch_size
        max_x = img_LR.shape[1] - patch_size
        max_y = img_LR.shape[2] - patch_size
        x_start = torch.randint(max_x,(1,1))[0,0]
        y_start = torch.randint(max_y,(1,1))[0,0]

        return img_HR[:,4*x_start:(4*x_start+4*patch_size),4*y_start:(4*y_start+4*patch_size)],\
        img_LR[:,x_start:(x_start+patch_size),y_start:(y_start+patch_size)]
