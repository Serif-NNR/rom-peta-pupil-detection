import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from itertools import chain
from math import ceil
from torchvision.models import VGG16_BN_Weights, ResNet50_Weights
from train_and_metric import device





class DynamicConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1):
        max_r = min(x1.shape[-1], x1.shape[-2]) / 4
        r = random.randint(2, int(max_r))
        k = random.randint(0, 0)
        kl = [3]
        if r < int((k + 1) / 2): r = int((k + 1) / 2)
        conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=kl[k], padding=r, dilation=r,
                      padding_mode="replicate"),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        ).to("cuda:0")
        conv[0].weight = self.conv[0].weight
        conv[1].weight = self.conv[1].weight
        output = conv(x1)
        self.conv = conv
        return output


class DynamicNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DynamicNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes


        self.s1_1 = DynamicConv2(1, 2)
        self.s1_2 = DynamicConv2(1, 2)

        self.s2_1 = DynamicConv2(4, 8)
        self.s2_2 = DynamicConv2(4, 8)
        self.s2_3 = DynamicConv2(4, 8)
        self.s2_4 = DynamicConv2(4, 8)

        self.s3_1 = DynamicConv2(16, 32)
        self.s3_2 = DynamicConv2(16, 32)
        self.s3_3 = DynamicConv2(16, 32)
        self.s3_4 = DynamicConv2(16, 32)
        self.s3_5 = DynamicConv2(16, 32)
        self.s3_6 = DynamicConv2(16, 32)

        self.s4 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.s5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.s6 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.s7 = nn.Sequential(
            nn.Conv2d(16, 4, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )

        self.s8 = nn.Sequential(
            nn.Conv2d(4, n_classes, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.Sigmoid()
        )

    def forward(self, x):
        s1_1 = self.s1_1(x)
        s1_2 = self.s1_2(x)

        s2_1 = self.s2_1(torch.cat([s1_1, s1_2], dim=1))
        s2_2 = self.s2_2(torch.cat([s1_1, s1_2], dim=1))
        s2_3 = self.s2_3(torch.cat([s1_1, s1_2], dim=1))
        s2_4 = self.s2_4(torch.cat([s1_1, s1_2], dim=1))

        s3_1 = self.s3_1(torch.cat([s2_1, s2_2], dim=1))
        s3_2 = self.s3_2(torch.cat([s2_3, s2_4], dim=1))
        s3_3 = self.s3_3(torch.cat([s2_1, s2_3], dim=1))
        s3_4 = self.s3_4(torch.cat([s2_2, s2_3], dim=1))
        s3_5 = self.s3_5(torch.cat([s2_1, s2_4], dim=1))
        s3_6 = self.s3_6(torch.cat([s2_2, s2_3], dim=1))

        s4 = self.s4(torch.cat([s3_1, s3_2, s3_3, s3_4, s3_5, s3_6], dim=1))
        s5 = self.s5(s4)
        s6 = self.s6(s5)
        s7 = self.s7(s6)
        return self.s8(s7)




class DynamicConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x1, x2):
        r = random.randint(2, 33)
        k = random.randint(0, 2)
        kl = [3, 5, 7]
        if r < int((k+1)/2): r = int((k+1)/2)
        conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kl[k], padding=r, dilation=r, padding_mode="replicate"),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        ).to("cuda:0")
        conv[0].weight = self.conv[0].weight
        conv[1].weight = self.conv[1].weight
        output = self.conv(x1)
        #self.conv = conv
        return output

class DynamicNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DynamicNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.model = nn.Sequential(

            nn.Conv2d(n_channels, 4, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            DynamicConv(4, 4),
            DynamicConv(4, 16),
            DynamicConv(16, 16),
            DynamicConv(16, 32),
            DynamicConv(32, 32),
            DynamicConv(32, 64),
            DynamicConv(64, 64),

            nn.Conv2d(64, 8, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, n_classes, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.model(x)



class MyNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MyNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        '''
                    nn.Conv2d(1, 2, kernel_size=3, padding=64, dilation=64, padding_mode="replicate"),
                    nn.BatchNorm2d(2),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(2, 4, kernel_size=3, padding=16, dilation=16, padding_mode="replicate"),
                    nn.BatchNorm2d(4),
                    nn.ReLU(inplace=True),
        '''

        self.model = nn.Sequential(

            nn.Conv2d(1, 4, kernel_size=3, padding=32, dilation=32, padding_mode="reflect"),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, kernel_size=3, padding=16, dilation=16, padding_mode="reflect"),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, padding=1, dilation=1, padding_mode="reflect"),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=16, dilation=16, padding_mode="reflect"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=32, dilation=32, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.model(x)


class MyNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MyNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        '''
                    nn.Conv2d(1, 2, kernel_size=3, padding=64, dilation=64, padding_mode="replicate"),
                    nn.BatchNorm2d(2),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(2, 4, kernel_size=3, padding=16, dilation=16, padding_mode="replicate"),
                    nn.BatchNorm2d(4),
                    nn.ReLU(inplace=True),
        '''

        self.model = nn.Sequential(

            nn.Conv2d(1, 4, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(4, 8, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, padding=1, dilation=1, padding_mode="reflect"),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.model(x)








class FullyConnectedAtomicSpace(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self):
        super().__init__()
        weights = torch.Tensor([0.5, 0.5, 0.5])
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor([0.5, 0.5, 0.5])
        self.bias = nn.Parameter(bias)



    def forward(self, x):
        asp = x.clone().cpu().detach().numpy()
        output = asp.copy()

        for i in range(1, x.shape[2]-1):
                    for j in range(1, x.shape[3]-1):
                        #av = np.average(asp[0, 1, i - 1:i + 2, j - 1:j + 2])
                        av = x[0, 1, i, j]
                        p = (x[0, 1] > av).sum()
                        n = (x[0, 1] == av).sum()
                        e = (x[0, 1] < av).sum()
                        output[0, 1, i, j] = (p * self.weights[0] + self.bias[0] +
                                                n * self.weights[1] + self.bias[1] +
                                                e * self.weights[2] + self.bias[2]).cpu().detach().numpy() / 3
        return torch.tensor(output).to("cuda:0")  # w times x + b

class MyUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MyUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.fcas = FullyConnectedAtomicSpace()

        self.inc = DrDoubleConv(n_channels, 8)
        #self.inc = DrDoubleConvWithKS1(n_channels, 8)
        self.down1 = DrDown(8, 16)
        self.down2 = DrDown(16, 32)
        self.down3 = DrDown(32, 64//factor)
        # self.down4 = DrDown(64, 128//factor)
        # self.up1 = DrUp(128, 64//factor, bilinear)
        self.up2 = DrUp(64, 32//factor, bilinear)
        self.up3 = DrUp(32, 16 // factor, bilinear)
        self.up4 = DrUp(16, 8 // factor, bilinear)
        self.outc = OutConv(8// factor, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.fcas(x4)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetFirst(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetFirst, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DrDoubleConv(n_channels, 8)
        #self.inc = DrDoubleConvWithKS1(n_channels, 8)
        self.down1 = DrDown(8, 16)
        self.down2 = DrDown(16, 32)
        self.down3 = DrDown(32, 64//factor)
        # self.down4 = DrDown(64, 128//factor)
        # self.up1 = DrUp(128, 64//factor, bilinear)
        self.up2 = DrUp(64, 32//factor, bilinear)
        self.up3 = DrUp(32, 16 // factor, bilinear)
        self.up4 = DrUp(16, 8 // factor, bilinear)
        self.outc = OutConv(8// factor, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits




class UNet_L2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_L2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.name = "UNetL2"

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64//factor)
        self.up1 = Up(64, 32//factor, bilinear)
        self.up2 = Up(32, 16//factor, bilinear)
        self.outc = OutConv(16, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits





class SRO_Down(nn.Module):
    def __init__(self, n_channels):
        super(SRO_Down, self).__init__()
        self.mid = int(n_channels/2)
        self.mp = nn.MaxPool2d(2)
        self.cs = nn.Conv2d(self.mid, self.mid, kernel_size=3, stride=2)
        self.lr = nn.ReLU()
        self.conv = DoubleConv(n_channels, n_channels)

    def forward(self, x):
        mp = self.mp(x)
        cs = self.cs(x)
        diffY = mp.size()[2] - cs.size()[2]
        diffX = mp.size()[3] - cs.size()[3]

        cs = F.pad(cs, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(self.lr(torch.cat([mp, cs], dim=1)))


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.name = "UNet_AV_Full"
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits




class UNetS(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNetS, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.name = "PERFECT_UNetS_41RATE_"

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256//factor)
        #self.down4 = Down(64, 128 // factor)
        #self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(256, 128//factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32 // factor, bilinear)
        self.outc = OutConv(32 // factor, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class UNetW(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetW, self).__init__()
        self.name = "UNet_PotioNet_Down4"
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1



        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64 // factor)
        self.down4 = Down(64, 128 // factor)
        self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)




        self.inc = DoubleConv(n_channels, 8)

        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64 // factor)
        self.down4 = Down(64, 128 // factor)
        self.down5 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.up5 = Up(16, 8, bilinear)

        self.outc = OutConv(8, n_classes)

        self.inc_c = DynamicConv2forUnet(8, 8)
        self.down1_c = DynamicConv2forUnet(16, 16)
        self.down2_c = DynamicConv2forUnet(32, 32)
        self.down3_c = DynamicConv2forUnet(64, 64)
        self.down4_c = DynamicConv2forUnet(128, 128)


        ###
        '''
        self.inc = DrDoubleConv(n_channels, 8)
        #self.inc = DrDoubleConvWithKS1(n_channels, 8)
        self.down1 = DrDown(8, 16)
        self.down2 = DrDown(16, 32)
        self.down3 = DrDown(32, 64//factor)
        # self.down4 = DrDown(64, 128//factor)
        # self.up1 = DrUp(128, 64//factor, bilinear)
        self.up2 = DrUp(64, 32//factor, bilinear)
        self.up3 = DrUp(32, 16 // factor, bilinear)
        self.up4 = DrUp(16, 8 // factor, bilinear)
        self.outc = OutConv(8// factor, n_classes)
        '''

        #
        '''
        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)
        '''

    def forward(self, x):


        x1 = self.inc(x)

        d1 = self.inc_c(x1)
        x2 = self.down1(x1)
        d2 = self.down1_c(x2)
        x3 = self.down2(x2)
        d3 = self.down2_c(x3)
        x4 = self.down3(x3)
        d4 = self.down3_c(x4)
        x5 = self.down4(x4)
        d5 = self.down4_c(x5)
        x6 = self.down5(x5)
        x = self.up1(x6, (x5 + d5) / 2)
        x = self.up2(x, (x4 + d4) / 2)
        x = self.up3(x, (x3 + d3) / 2)
        x = self.up4(x, (x2 + d2) / 2)
        x = self.up5(x, (x1 + d1) / 2)

        logits = self.outc(x)
        #logits = self.outc(x)
        return logits






class DynamicConv2forUnet(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, trn=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trn = trn
        md_channel = int(self.out_channels / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, md_channel, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(md_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(md_channel, out_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1):
        if True:
            max_r = max(min(x1.shape[-1], x1.shape[-2]) / 16, 1)
            r = random.randint(1, int(max_r))
            k = random.randint(0, 0)
            kl = [3]
            if r < int((k + 1) / 2): r = int((k + 1) / 2)
            md_channel = int(self.out_channels / 2)
            conv = nn.Sequential(
                nn.Conv2d(self.in_channels, md_channel, kernel_size=kl[k], padding=r, dilation=r, padding_mode="replicate"),
                nn.BatchNorm2d(md_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(md_channel, self.out_channels, kernel_size=3, padding=1, padding_mode="replicate"),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            ).to("cuda:0")
            conv[0].weight = self.conv[0].weight
            conv[1].weight = self.conv[1].weight
            output = conv(x1)
            self.conv = conv
        else:
            output = self.conv(x1)
        return output









if __name__ == "__main__":
    model = UNet(1, 1).to(device)
    out = model(torch.randn(1, 1, 256, 256).to(device))
    print(out.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    from prettytable import PrettyTable


    def count_parameters(modelpar):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in modelpar.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params


    count_parameters(model)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        #print("DoubleConv"+str(x.shape))
        return self.double_conv(x)


class DrDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        #print("DrDoubleConv"+str(x.shape))
        return self.double_conv(x)


class DrDoubleConvWithKS1(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        #print("DrDoubleConv"+str(x.shape))
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        #print("Down"+str(x.shape))
        return self.maxpool_conv(x)


class DrDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DrDoubleConv(in_channels, out_channels))

    def forward(self, x):
        #print("DrDown"+str(x.shape))
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DrUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DrDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DrDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)










class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = int(in_channels / 2)
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers).to("cuda:0")

    def forward(self, x):
        return self.decode(x)

class SegNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, pretrained=True):
        super(SegNet, self).__init__()
        vgg = models.vgg19_bn()
        features = list(vgg.features.children())
        if in_channels != 3:
            features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:27])
        self.enc4 = nn.Sequential(*features[27:40])
        self.enc5 = nn.Sequential(*features[40:])
        self.name = "SegNet_41Rate_FuLL_"
        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] +
              [nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True)] * 4)
        )
        self.dec4 = _DecoderBlock(1024, 256, 4)
        self.dec3 = _DecoderBlock(512, 128, 4)
        self.dec2 = _DecoderBlock(256, 64, 2)
        self.dec1 = _DecoderBlock(128, num_classes, 2)
        self.sig = nn.Sigmoid()
        #initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        diffY = enc4.size()[2] - dec5.size()[2]
        diffX = enc4.size()[3] - dec5.size()[3]

        dec5 = F.pad(dec5, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        #enc4 = enc4[:, :, 0: dec5.shape[-2], 0:dec5.shape[-1]]
        dec4 = self.dec4(torch.cat([enc4, dec5], 1))
        #enc3 = enc3[:, :, 0: dec4.shape[-2], 0:dec4.shape[-1]]
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        #enc2 = enc2[:, :, 0: dec3.shape[-2], 0:dec3.shape[-1]]
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        #enc1 = enc1[:, :, 0: dec2.shape[-2], 0:dec2.shape[-1]]
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))
        #dec = dec1[:, :, 0:x.size()[-2], 0:x.size()[-1]]
        out = self.sig(dec1)
        return out




class SegNet_VGG16_L2(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, pretrained=True, freeze_bn=False, **_):
        super(SegNet_VGG16_L2, self).__init__()
        vgg_bn = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        encoder = list(vgg_bn.features.children())
        self.name = "SegNetVGG_L2_S_"
        # Adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # Encoder, VGG without any maxpooling
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        #self.stage1_encoder_after = nn.Sequential(
        #    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #    nn.ReLU(inplace=True)
        #)
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        #self.stage3_encoder = nn.Sequential(*encoder[14:23])
        #self.stage4_encoder = nn.Sequential(*encoder[24:33])
        #self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder, same as the encoder but reversed, maxpool will not be used
        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i + 3][::-1]]
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i + 1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        #self.stage1_decoder = nn.Sequential(*decoder[0:9])
        #self.stage2_decoder = nn.Sequential(*decoder[9:18],
        #                                    nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        #                                    )
        #self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        #self.stage4_decoder_after = nn.Sequential(
        #    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #    nn.ReLU(inplace=True)
        #)
        self.stage5_decoder = nn.Sequential(*decoder[33:],
                                            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1))
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.sig = nn.Sigmoid()
        self._initialize_weights(self.stage4_decoder, self.stage5_decoder)
        #if freeze_bn: self.freeze_bn()
        #if freeze_backbone:
        #    set_trainable([self.stage1_encoder, self.stage2_encoder, self.stage3_encoder, self.stage4_encoder,
        #                   self.stage5_encoder], False)

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        x = self.stage1_encoder(x)
        #x = self.stage1_encoder_after(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        #x = self.stage3_encoder(x)
        #x3_size = x.size()
        #x, indices3 = self.pool(x)

        #x = self.stage4_encoder(x)
        #x4_size = x.size()
        #x, indices4 = self.pool(x)

        #x = self.stage5_encoder(x)
        #x5_size = x.size()
        #x, indices5 = self.pool(x)

        # Decoder
        #x = self.unpool(x, indices=indices5, output_size=x5_size)
        #x = self.stage1_decoder(x)

        #x = self.unpool(x, indices=indices4, output_size=x4_size)
        #x = self.stage2_decoder(x)

        #x = self.unpool(x, indices=indices3, output_size=x3_size)
        #x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)
        #x = self.stage4_decoder_after(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)

        return self.sig(x)

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()




class SegResNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, pretrained=True, freeze_bn=False, **_):
        super(SegResNet, self).__init__()
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        encoder = list(resnet50.children())
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        encoder[3].return_indices = True

        # Encoder
        self.first_conv = nn.Sequential(*encoder[:4])
        resnet50_blocks = list(resnet50.children())[4:-2]
        self.encoder = nn.Sequential(*resnet50_blocks)

        # Decoder
        resnet50_untrained = models.resnet50(weights=None)
        resnet50_blocks = list(resnet50_untrained.children())[4:-2][::-1]
        decoder = []
        channels = (2048, 1024, 512)
        for i, block in enumerate(resnet50_blocks[:-1]):
            new_block = list(block.children())[::-1][:-1]
            decoder.append(nn.Sequential(*new_block, DecoderBottleneck(channels[i])))
        new_block = list(resnet50_blocks[-1].children())[::-1][:-1]
        decoder.append(nn.Sequential(*new_block, LastBottleneck(256)))

        self.decoder = nn.Sequential(*decoder)
        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        #if freeze_bn: self.freeze_bn()
        #if freeze_backbone:
        #    set_trainable([self.first_conv, self.encoder], False)

    def forward(self, x):
        inputsize = x.size()

        # Encoder
        x, indices = self.first_conv(x)
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)
        h_diff = ceil((x.size()[2] - indices.size()[2]) / 2)
        w_diff = ceil((x.size()[3] - indices.size()[3]) / 2)
        if indices.size()[2] % 2 == 1:
            x = x[:, :, h_diff:x.size()[2] - (h_diff - 1), w_diff: x.size()[3] - (w_diff - 1)]
        else:
            x = x[:, :, h_diff:x.size()[2] - h_diff, w_diff: x.size()[3] - w_diff]

        x = F.max_unpool2d(x, indices, kernel_size=2, stride=2)
        x = self.last_conv(x)

        if inputsize != x.size():
            h_diff = (x.size()[2] - inputsize[2]) // 2
            w_diff = (x.size()[3] - inputsize[3]) // 2
            x = x[:, :, h_diff:x.size()[2] - h_diff, w_diff: x.size()[3] - w_diff]
            if h_diff % 2 != 0: x = x[:, :, :-1, :]
            if w_diff % 2 != 0: x = x[:, :, :, :-1]

        return x

    def get_backbone_params(self):
        return chain(self.first_conv.parameters(), self.encoder.parameters())

    def get_decoder_params(self):
        return chain(self.decoder.parameters(), self.last_conv.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()




class SegNet_ResNet_Seq1(nn.Module):
    def __init__(self,  in_channels=1, num_classes=1, pretrained=True, freeze_bn=False, **_):
        super(SegNet_ResNet_Seq1, self).__init__()
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        encoder = list(resnet50.children())
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        encoder[3].return_indices = True

        # Encoder
        self.first_conv = nn.Sequential(*encoder[:4])
        resnet50_blocks = list(resnet50.children())[4:-2]
        self.encoder = nn.Sequential(*resnet50_blocks[0])

        # Decoder
        resnet50_untrained = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet50_blocks = list(resnet50_untrained.children())[4:-2][::-1]
        decoder = []
        channels = (2048, 1024, 512)
        for i, block in enumerate(resnet50_blocks[:-1]):
            new_block = list(block.children())[::-1][:-1]
            decoder.append(nn.Sequential(*new_block, DecoderBottleneck(channels[i])))
        new_block = list(resnet50_blocks[-1].children())[::-1][:-1]
        decoder.append(nn.Sequential(*new_block, LastBottleneck(256)))

        self.decoder = nn.Sequential(*decoder[3])
        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1)
        )
        #if freeze_bn: self.freeze_bn()
        #if freeze_backbone:
        #    set_trainable([self.first_conv, self.encoder], False)
        print("")

    def forward(self, x):
        inputsize = x.size()
        inputshape = x.shape

        # Encoder
        x, indices = self.first_conv(x)
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)
        h_diff = ceil((x.size()[2] - indices.size()[2]) / 2)
        w_diff = ceil((x.size()[3] - indices.size()[3]) / 2)
        if indices.size()[2] % 2 == 1:
            x = x[:, :, h_diff:x.size()[2] - (h_diff - 1), w_diff: x.size()[3] - (w_diff - 1)]
        else:
            x = x[:, :, h_diff:x.size()[2] - h_diff, w_diff: x.size()[3] - w_diff]

        x = F.max_unpool2d(x, indices, kernel_size=2, stride=2)
        x = self.last_conv(x)

        outputsize = x.size()
        if inputsize != outputsize:
            h_diff = (x.size()[2] - inputsize[2]) // 2
            w_diff = (x.size()[3] - inputsize[3]) // 2
            x = x[:, :, h_diff:x.size()[2] - h_diff, w_diff: x.size()[3] - w_diff]
            if h_diff % 2 != 0: x = x[:, :, :-1, :]
            if w_diff % 2 != 0: x = x[:, :, :, :-1]

        x = x[:, :, 0:inputsize[-2], 0:inputsize[-1]]

        return x

    def get_backbone_params(self):
        return chain(self.first_conv.parameters(), self.encoder.parameters())

    def get_decoder_params(self):
        return chain(self.decoder.parameters(), self.last_conv.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()




class DecoderBottleneck(nn.Module):
    def __init__(self, inchannels):
        super(DecoderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels // 4)
        self.conv2 = nn.ConvTranspose2d(inchannels // 4, inchannels // 4, kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels // 4)
        self.conv3 = nn.Conv2d(inchannels // 4, inchannels // 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.ConvTranspose2d(inchannels, inchannels // 2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(inchannels // 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class LastBottleneck(nn.Module):
    def __init__(self, inchannels):
        super(LastBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels // 4)
        self.conv2 = nn.Conv2d(inchannels // 4, inchannels // 4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels // 4)
        self.conv3 = nn.Conv2d(inchannels // 4, inchannels // 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(inchannels, inchannels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(inchannels // 4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
