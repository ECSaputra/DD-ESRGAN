


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
        del conv1out
        del conv2out
        del rrdbout

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

