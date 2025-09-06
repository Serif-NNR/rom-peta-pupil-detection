import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from TransUNet import VisionTransformer, get_r50_b16_config


class DoubleConv(nn.Module):
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
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet1, self).__init__()
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
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

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


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 4)
        self.down1 = Down(4, 8)
        self.down2 = Down(8, 16)
        self.up3 = Up(16, 8 // factor, bilinear)
        self.up4 = Up(8, 4, bilinear)
        self.outc = OutConv(4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 2)
        self.down1 = Down(2, 4)
        self.down2 = Down(4, 8)
        self.up3 = Up(8, 4 // factor, bilinear)
        self.up4 = Up(4, 2, bilinear)
        self.outc = OutConv(2, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet5(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet5, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64 // factor)
        self.down4 = Down(64, 128 // factor)
        self.down5 = Down(128, 256 // factor)
        self.up0 = Up(256, 128 // factor, bilinear)
        self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    #print(table)
    #print(f"Total Trainable Params: {total_params}")
    return total_params


for m in [UNet1, UNet2, UNet3, UNet4, UNet5]:
    fileContent = {"Model Name": list(), "Parameter Count": list(), "Batch Size": list(),
                   "GPU (320x240)": list(), "GPU (105x120)": list(), "GPU Percent": list(),
                   "CPU (320x240)": list(), "CPU (105x120)": list(), "CPU Percent": list()}

    def calculate_prediction_durations(first_tensor, second_tensor, batch_size, loop_count=100, experiment_count=5,
                                       parameter_count="", model_name=""):
        global fileContent
        devs = ["cuda:0", "cpu"]
        tensors = [first_tensor, second_tensor]

        def calculate(model, tensor, device, loop_count):
            durations = []
            outset = None
            with torch.no_grad():
                for i in range(loop_count):
                    outset = time.time()
                    result = model(tensor.to(device))
                    durations.append(time.time() - outset)
            return np.mean(durations)

        timeCpu320 = list()
        timeGpu320 = list()
        timeCpu105 = list()
        timeGpu105 = list()
        for exp in range(1, experiment_count + 1):
            print("\t[EXP {0:4d}]".format(exp), end="")
            control = False
            for tensor in tensors:
                modelGPU = m(1, 1).to("cuda:0")
                modelCPU = m(1, 1).to("cpu")
                durationGPU = calculate(modelGPU, tensor, "cuda:0", loop_count)
                durationCPU = calculate(modelCPU, tensor, "cpu", loop_count)
                if control:
                    print("\t\t\t", end="")
                    timeGpu105.append(durationGPU)
                    timeCpu105.append(durationCPU)
                else:
                    control = True
                    timeGpu320.append(durationGPU)
                    timeCpu320.append(durationCPU)

                print(
                    "{0}{1}\tInput Size: {2:15s}\tGPU Prediction Duration: {3:8f} ({5:8f})\tCPU Prediction Duration: {4:8f} ({6:8f})".
                    format("", "", str(tensor.shape), durationGPU, durationCPU, durationGPU / i, durationCPU / i))

        fileContent["GPU (320x240)"].append(np.mean(timeGpu320))
        fileContent["GPU (105x120)"].append(np.mean(timeGpu105))
        fileContent["GPU Percent"].append(np.mean(timeGpu105)/np.mean(timeGpu320))

        fileContent["CPU (320x240)"].append(np.mean(timeCpu320))
        fileContent["CPU (105x120)"].append(np.mean(timeCpu105))
        fileContent["CPU Percent"].append(np.mean(timeCpu105)/np.mean(timeCpu320))

        fileContent["Model Name"].append(model_name)
        fileContent["Parameter Count"].append(parameter_count)
        fileContent["Batch Size"].append(batch_size)



    for i in range(1, 23, 3):
        tempModel = m(1, 1)
        print(
            "\nMODEL     : {2:5s}\tParameter Count              : {3}\nBATCH SIZE: {0:5d}\tAbout Initialization Duration: {1:8f}".format(
                i, i * 0.002, m.__name__, str(count_parameters(tempModel))))
        calculate_prediction_durations(first_tensor=torch.randn(i, 1, 320, 240),
                                       second_tensor=torch.randn(i, 1, 105, 120), batch_size=i,
                                       parameter_count=str(count_parameters(tempModel)), model_name=m.__name__)

df = pd.DataFrame(fileContent)
df.to_excel("Fixein_BMS_"+str(time.time()).split('.')[0]+".xlsx")