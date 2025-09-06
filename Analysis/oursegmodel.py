import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import time
import numpy as np
import pandas as pd




class DynamicConv2(nn.Module):
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
        if False:
            max_r = min(x1.shape[-1], x1.shape[-2]) / 4
            r = random.randint(2, int(max_r))
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



class SRO_ConvDilation(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=3):
        super().__init__()
        self.cd = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation,
                      padding_mode="replicate"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.cd(x)


class SRO_ConvDouble(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = int(out_channels / 2)
        self.cd = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.cd(x)


class SRO_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cd = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.cd(x)


class SRO_Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.c1x1(x)


class SRO_Down(nn.Module):
    def __init__(self, n_channels):
        super(SRO_Down, self).__init__()
        self.mp = nn.MaxPool2d(2)
        self.cs = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2)
        self.lr = nn.ReLU()

    def forward(self, x):
        mp = self.mp(x)
        cs = self.cs(x)
        diffY = mp.size()[2] - cs.size()[2]
        diffX = mp.size()[3] - cs.size()[3]

        cs = F.pad(cs, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.lr(torch.cat([mp, cs], dim=1))


class SRO_DownDouleConv(nn.Module):
    def __init__(self, n_channels):
        super(SRO_DownDouleConv, self).__init__()
        self.mp = nn.MaxPool2d(2)
        self.cs = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2)
        self.lr = nn.ReLU()

        self.dc = SRO_ConvDouble(n_channels*2, n_channels*2)

    def forward(self, x):
        mp = self.mp(x)
        cs = self.cs(x)
        diffY = mp.size()[2] - cs.size()[2]
        diffX = mp.size()[3] - cs.size()[3]

        cs = F.pad(cs, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        lr = self.lr(torch.cat([mp, cs], dim=1))
        return self.dc(lr)


class SRO_DownSimple(nn.Module):
    def __init__(self, n_channels):
        super(SRO_Down, self).__init__()
        self.mp = nn.MaxPool2d(2)

    def forward(self, x):
        mp = self.mp(x)
        return mp


class SRO_Up(nn.Module):
    def __init__(self, in_channels):
        super(SRO_Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


class SRO_NetNew(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SRO_NetNew, self).__init__()
        self.name = "NetNew"
        self.s1 = SRO_ConvDouble(n_channels, 8)

        self.l1 = SRO_ConvDouble(8, 8)
        self.l1_d1 = nn.MaxPool2d(2)
        self.l1_c1 = SRO_ConvDouble(8, 8)
        self.l1_u1 = SRO_Up(8)

        self.l2 = SRO_ConvDouble(20, 20)
        self.l2_d1 = nn.MaxPool2d(2)
        self.l2_c1 = SRO_ConvDouble(20, 20)
        self.l2_d2 = nn.MaxPool2d(2)
        self.l2_c2 = SRO_ConvDouble(20, 20)
        self.l2_u2 = SRO_Up(20)
        self.l2_c3 = SRO_ConvDouble(30, 20)
        self.l2_u1 = SRO_Up(20)

        self.l3 = SRO_ConvDouble(50, 32)
        self.l3_d1 = nn.MaxPool2d(2)
        self.l3_c1 = SRO_ConvDouble(32, 32)
        self.l3_d2 = nn.MaxPool2d(2)
        self.l3_c2 = SRO_ConvDouble(32, 32)
        self.l3_d3 = nn.MaxPool2d(2)
        self.l3_c3 = SRO_ConvDouble(32, 32)
        self.l3_u3 = SRO_Up(32)
        self.l3_c4 = SRO_ConvDouble(48, 32)
        self.l3_u2 = SRO_Up(32)
        self.l3_c5 = SRO_ConvDouble(48, 32)
        self.l3_u1 = SRO_Up(32)

        self.l4 = SRO_ConvDouble(80, 64)

        self.l5 = SRO_ConvDouble(214, 64)

        self.git = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        self.sktr = nn.Sigmoid()

    def forward(self, x):
        # x = self.fs(x)

        x = self.s1(x)

        x1 = self.l1(x)
        l1 = self.l1_d1(x1)
        l1 = self.l1_c1(l1)
        l1 = self.l1_u1(l1, x1)
        x1 = torch.cat([x1, l1], dim=1)

        x2 = self.l2(x1)
        l2 = self.l2_d1(x2)
        l2 = self.l2_c1(l2)
        l2_1 = self.l2_d2(l2)
        l2_1 = self.l2_c2(l2_1)
        l2_1 = self.l2_u2(l2_1, l2)
        l2_1 = self.l2_c3(l2_1)
        l2 = self.l2_u1(l2_1, x2)
        x2 = torch.cat([x2, l2], dim=1)

        x3 = self.l3(x2)
        l3 = self.l3_d1(x3)
        l3 = self.l3_c1(l3)
        l3_1 = self.l3_d2(l3)
        l3_1 = self.l3_c2(l3_1)
        l3_2 = self.l3_d3(l3_1)
        l3_2 = self.l3_c3(l3_2)
        l3_3 = self.l3_u3(l3_2, l3_1)
        l3_3 = self.l3_c4(l3_3)
        l3_3 = self.l3_u2(l3_3, l3)
        l3_3 = self.l3_c5(l3_3)
        l3_3 = self.l3_u2(l3_3, x3)
        x3 = torch.cat([x3, l3_3], dim=1)

        x4 = self.l4(x3)

        x5 = self.l5(torch.cat([x1, x2, x3, x4], dim=1))

        git = self.git(x5)
        return self.sktr(git)


class SRO_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SRO_UNet, self).__init__()
        self.name = "SRO_UNet_UandL"
        self.inv = SRO_ConvDouble(n_channels, 16)

        self.down1 = SRO_DownDouleConv(16)
        self.down1_c1 = DynamicConv2(16, 32)
        self.down1_c2 = DynamicConv2(32, 32)
        self.down2 = SRO_DownDouleConv(32)
        self.down2_c1 = DynamicConv2(32, 64)
        self.down3 = SRO_DownDouleConv(64)

        self.up3 = SRO_Up(128)
        self.conv_u3 = DynamicConv2(128, 64)
        self.up2 = SRO_Up(64)
        self.conv_u2 = DynamicConv2(128, 64)
        self.up1 = SRO_Up(64)
        self.conv_u1 = DynamicConv2(80, 32)

        self.lastConv = nn.Conv2d(32, n_classes, kernel_size=3, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        nv = self.inv(x)
        x1_c = self.down1_c1(nv)
        x1_c = self.down1_c2(x1_c)

        x1 = self.down1(nv)

        x2_c = self.down2_c1(x1)
        x2 = self.down2(x1)


        x3 = self.down3(x2)

        u3 = self.conv_u3(self.up3(x3, x2))

        u2 = self.up2(u3, x1)
        u2 = self.conv_u2(torch.cat([x2_c, u2], dim=1))

        u1 = self.up1(u2, nv)
        u1 = self.conv_u1(torch.cat([x1_c, u1], dim=1))

        out = self.lastConv(u1)
        return self.sig(out)

