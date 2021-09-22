import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19
import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm

######################################################################

### LOSS FUNCTIONS IN THE ORIGINAL ESRGAN FORMULATION

######################################################################



class vggloss(torch.nn.Module):
  def __init__(self, device):
    super(vggloss, self).__init__()
    vgg19 = torchvision.models.vgg19(pretrained=True)
    vgg_layers = list(vgg19.features.children())[:35]
    with torch.no_grad():
      vgg_features = torch.nn.Sequential(*vgg_layers).eval()
    self.vgg_features = vgg_features.to(device)

  def forward(self, img1, img2):
    with torch.no_grad():
      vgg_img1 = self.vgg_features(img1)
      vgg_img2 = self.vgg_features(img2)
    loss = torch.nn.functional.mse_loss(vgg_img1, vgg_img2)
    return loss

### Perceptual loss ESRGAN
class GeneratorLossESRGAN():
  def __init__(self, content_loss, D, lamb, eta, device):
    self.content_loss = content_loss
    self.D = D.to(device)
    self.lamb = lamb
    self.eta = eta
  
  def forward(self, x_real, x_fake):
    ragan_loss = -torch.mean(torch.log(1-self.D.forward(x_real,x_fake))) - torch.mean(torch.log(self.D.forward(x_fake,x_real)))
    ragan_loss = ragan_loss.cuda()
    content_loss_out = self.content_loss(x_real, x_fake).cuda()
    adversarial_loss_out = torch.sum(sum(-torch.log(self.D.forward(x_real, x_fake)))).cuda()

    loss = content_loss_out + 0.001*adversarial_loss_out + lamb*ragan_loss + eta*torch.nn.functional.l1_loss(x_real, x_fake)
    loss.cuda()
    return loss

### Discriminator loss
class DiscriminatorLoss():
    def __init__(self, D,device):
        self.D = D.to(device)
        
    def forward(self, x_real, x_fake):
        loss = -torch.mean(torch.log(self.D.forward(x_real,x_fake)))- torch.mean(torch.log(1-self.D.forward(x_fake,x_real)))
        loss = loss.cuda()
        return loss
