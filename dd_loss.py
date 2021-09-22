import math
import torch
import numpy as np

### Perceptual loss ESRGAN
class GeneratorLossDualESRGAN():
  def __init__(self, content_loss, D1, D2, lamb, eta, device):
    self.content_loss = content_loss
    self.D1 = D1.to(device)
    self.D2 = D2.to(device)
    self.lamb = lamb
    self.eta = eta
  
  def forward(self, x_real, x_fake):
#    ragan_loss_d1 = -torch.mean(torch.log(1-self.D1.forward(x_real,x_fake))) - torch.mean(torch.log(self.D1.forward(x_fake,x_real)))
#    ragan_loss_d2 = -torch.mean(torch.log(self.D2.forward(x_real,x_fake))) - torch.mean(torch.log(1-self.D2.forward(x_fake,x_real)))
    ragan_loss_d1 = torch.mean(torch.log(self.D1.forward(x_real,x_fake))) - torch.mean(torch.log(self.D1.forward(x_fake,x_real)))
    ragan_loss_d2 = -torch.mean(torch.log(self.D2.forward(x_real,x_fake))) + torch.mean(torch.log(self.D2.forward(x_fake,x_real)))
    content_loss_out = self.content_loss(x_real, x_fake).cuda()
    adversarial_loss_out1 = torch.sum(sum(-torch.log(self.D1.forward(x_real, x_fake)))).cuda()
    adversarial_loss_out2 = torch.sum(sum(-torch.log(1-self.D2.forward(x_real, x_fake)))).cuda()

    loss = content_loss_out + 0.001*(adversarial_loss_out1 + adversarial_loss_out2) + lamb*(ragan_loss_d1 + ragan_loss_d2) + eta*torch.nn.functional.l1_loss(x_real, x_fake)
    loss.cuda()
    return loss
    

### Discriminator loss one
class DiscriminatorLossOne():
    def __init__(self, D,device):
        self.D = D.to(device)
        
    def forward(self, x_real, x_fake):
        ragan_D = self.D.forward(x_real,x_fake).cuda()
        #loss = -torch.mean(torch.log(self.D.forward(x_real,x_fake)))- torch.mean(torch.log(1-self.D.forward(x_fake,x_real)))
        loss = -torch.mean(torch.log(self.D.forward(x_real,x_fake))) + torch.mean(torch.log(self.D.forward(x_fake,x_real)))
        loss = loss.cuda()
        return loss


class DiscriminatorLossTwo():
    def __init__(self, D,device):
        self.D = D.to(device)
        
    def forward(self, x_real, x_fake):
        ragan_D = self.D.forward(x_real,x_fake).cuda()
#        loss = -torch.mean(torch.log(1-self.D.forward(x_real,x_fake)))- torch.mean(torch.log(self.D.forward(x_fake,x_real)))
        loss = torch.mean(torch.log(self.D.forward(x_real,x_fake))) - torch.mean(torch.log(self.D.forward(x_fake,x_real)))
        loss = loss.cuda()
        return loss
