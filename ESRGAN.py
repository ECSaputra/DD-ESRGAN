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
        

class GeneratorESRGAN(nn.Module):
  
    def __init__(self, num_rrdb, res_scaling_factor, rrdb_channel_size, rrdb_out_size):
        super(GeneratorESRGAN, self).__init__()
        self.layer1 = nn.Conv2d(3, rrdb_channel_size, kernel_size=3,padding=1)
        nn.init.kaiming_normal_(self.layer1.weight)

        rrdb_blocks = []
        for i in range(num_rrdb):
            rrdb_blocks.append(RRDB(rrdb_channel_size, rrdb_out_size, res_scaling_factor))
        self.rrdb = nn.Sequential(*rrdb_blocks)

        self.layer2 = nn.Conv2d(rrdb_channel_size, rrdb_channel_size, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.layer2.weight)

        # upsampling layers:
        self.upsampling1 = UpsampleBLock(rrdb_channel_size, 2)
        self.upsampling2 = UpsampleBLock(rrdb_channel_size, 2)

        self.layer3 = nn.Conv2d(rrdb_channel_size, rrdb_channel_size, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.layer3.weight)

        self.layer4 = nn.Conv2d(rrdb_channel_size, 3, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.layer4.weight)


    def forward(self, x):
        conv1out = self.layer1(x)
        rrdbout = self.rrdb(conv1out)
        conv2out = self.layer2(rrdbout)

        out = conv1out + conv2out

        out = self.upsampling1(out)
        out = self.upsampling2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out
        

class DiscriminatorESRGAN(nn.Module):
    def __init__(self):
        super(DiscriminatorESRGAN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x_real, x_fake):
        batch_size = x_real.size(0)
        C_real = self.net(x_real).view(batch_size)
        EC_fake = self.net(x_fake).view(batch_size).mean()
        out = torch.sigmoid(C_real - EC_fake)
        return out


class MyDataset(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.file_list = os.listdir(filepath)
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return torchvision.transforms.Resize((1000,2000)) (torchvision.transforms.ToTensor()(Image.open(self.filepath+self.file_list[index]))),\
        torchvision.transforms.Resize((250,500))(torchvision.transforms.ToTensor()(Image.open(self.filepath[:-3]+'LR_bicubic/X4/'+self.file_list[index][:-4]+'x4.png')))


def main():
    G = GeneratorESRGAN(16, 0.2, 64, 32)
    G.load_state_dict(torch.load('best_weight_Generator.model'))
    D = DiscriminatorESRGAN()
    D.load_state_dict(torch.load('best_weight_Discriminator.model'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = MyDataset('DIV2K_train_HR/')
    train_loader_args = dict(shuffle=True, batch_size=5, num_workers=8, pin_memory=True, drop_last=True) if cuda else dict(shuffle=True, batch_size=1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_loader_args)
#    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    
    val_dataset = MyDataset('DIV2K_valid_HR/')
    val_loader_args = dict(shuffle=False, batch_size=5, num_workers=8, pin_memory=True, drop_last=True) if cuda else dict(shuffle=True, batch_size=1)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_loader_args)
    #val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    G = G.to(device)
    D = D.to(device)
    
    criterion= nn.MSELoss()

    lrate = 2*10**(-4)
    optimizerG = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=lrate)
    optimizerD = torch.optim.Adam(D.parameters(), betas=(0.9, 0.999), lr=lrate)
    numEpochs = 10
    start_time = time.time()
    avg_loss_epoch = []

    for epoch in range(numEpochs):
        print("="*20, " Epoch ", epoch+1)
      
        # start training
        avg_loss = 0.0
        num_correct = 0
        total_num = 0

        for batch_num, (highres, lowres) in enumerate(train_dataloader):
            highes, lowres = highres.to(device), lowres.to(device)

            #print("Training D")
            optimizerD.zero_grad()
            D.train()

            #print("Training G")
            optimizerG.zero_grad()
            G.train()

            for i in range(5):
                outputs = G(lowres[:,:,50*i:50*(i+1),:])
                outputs.to(device)
                loss = criterion(outputs,highres[:,:,200*i:200*(i+1),:])
                loss.backward()
                optimizerG.step()
                optimizerD.step()
                
                avg_loss += loss.item()

            del highres
            del lowres
        
            best_weightsG = G.state_dict()
            torch.save(best_weightsG, "./best_weight_Generatorv2.model")

            best_weightsD = D.state_dict()
            torch.save(best_weightsD, "./best_weight_Discriminatorv2.model")
        
            if batch_num%10==0:
                plt.clf()
                G.eval()
                for val_batch_num, (highres,lowres) in enumerate(val_dataloader):
                    if val_batch_num==2:
                        break
                highres, lowres = highres.to(device), lowres.to(device)

                i = 0 
                out = G(lowres[:,:,50*i:50*(i+1),:])
                plt.imshow(out[0,:,:,:].permute(1,2,0).detach().cpu().numpy())
                plt.show()
                print(batch_num,avg_loss/(batch_num+1))

        avg_loss_epoch.append(avg_loss/(batch_num+1))

if __name__ == "__main__":
    main()
