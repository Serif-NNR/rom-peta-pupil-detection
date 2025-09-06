import time
import numpy as np
import torch
import pytorch_memlab
from LEyesEvaluate.models import PupilNet # FOR LEyes
from model_CCSDG import UNetCCSDG, UNetCCSDG_S
from model_EgeUNet import EGEUNet
#from FlexiVit import FlexiVisionTransformer
#from paddleseg.models.backbones.resnet_vd import ResNet_vd
from segmodel import UNet, UNetS, SegNet, SegNet_VGG16_L2
from ENet import ENet
from TransUNet import VisionTransformer, get_r50_b16_config, get_r50_b16_configS
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
#from UniverSeg import UniverSeg
#torch.cuda.synchronize("cuda:0")
#from PaddleSeg import PPMobileSeg
from torch_UnetPP import NestedUNet, NestedUNetS
from torch_SegNext import SegNext
#from paddleseg.models.backbones import *
from torch_DeepLabv3 import *
from paddle_PSPNet_ResNet import PSPNet
from paddle_PPMobileSeg import *

import segmentation_models_pytorch as smp
#flexivitGPU = FlexiVisionTransformer().to("cuda:0")
#flexivitCPU = FlexiVisionTransformer().to("cpu")

#PPbackbone = StrideFormer()
#PPMobileSegGPU = PPMobileSeg(num_classes=1, backbone=MobileSeg_Tiny()).to("gpu")
#PPMobileSegCPU = PPMobileSeg(num_classes=1, backbone=MobileSeg_Tiny()).to("cpu")
import math

def convert_size(size_bytes):
    # https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])


def calculate_prediction_durations(first_tensor, second_tensor, batch_size, loop_count=100, experiment_count=5):
    devs = ["cuda:0", "cpu"]
    tensors = [first_tensor, second_tensor]

    def calculate(model, tensor, device, loop_count, exp_count, batch_size, params, starting):
        durations = []
        outset = None
        tsr = tensor.to(device)

        with torch.no_grad():
            for i in range(10): a = model(tsr)  # warm-up
            #rep = pytorch_memlab.MemReporter(model)
            #rep.report()



            if device != "cpu":
                for i in range(loop_count):
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    starter.record()
                    a = model(tsr)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender) / 1000
                    durations.append(curr_time)

                with profile(profile_memory=True) as profGPU:
                    a = model(tsr)
                rep = pytorch_memlab.MemReporter(model)
                #rep.report()
                print(tsr.shape, " -> ", device)

                #print(profGPU.key_averages().table(sort_by='cuda_memory_usage'))
                profiling_gpu_df = pd.DataFrame({e.key: e.__dict__ for e in profGPU.key_averages()}).T
                list_gpu_mem = np.asarray(profiling_gpu_df["cuda_memory_usage"].tolist(), dtype=float)
                print("gpu mem byte: ", convert_size(list_gpu_mem[list_gpu_mem > 0].sum()))
                list_cpu_mem = np.asarray(profiling_gpu_df["cpu_memory_usage"].tolist(), dtype=float)
                print("cpu mem byte: ", convert_size(list_cpu_mem[list_cpu_mem > 0].sum()))

                #profiling_gpu_df = pd.DataFrame({e.key: e.__dict__ for e in profGPU.key_averages()}).T
                #profiling_gpu_df.to_csv(f"Profiling/{str(starting)}_{model.name}_{device.split(':')[0]}_(CUDA)_{tensor.size()[2]}-{tensor.size()[3]}_exp-{exp_count}_batch-{batch_size}_param-{params}.csv")



            else:
                for i in range(loop_count):
                    outset = time.time()
                    a = model(tsr)
                    durations.append(time.time() - outset)

                with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as profCPU:
                    a = model(tsr)

                print(tsr.shape, " -> ", device)
                profiling_cpu_df = pd.DataFrame({e.key: e.__dict__ for e in profCPU.key_averages()}).T
                list_cpu_mem = np.asarray(profiling_cpu_df["cpu_memory_usage"].tolist(), dtype=float)
                print("cpu mem byte: ", convert_size(list_cpu_mem[list_cpu_mem > 0].sum()))
                #print(profCPU.key_averages().table(sort_by='cpu_memory_usage'))
                # profiling_cpu_df = pd.DataFrame({e.key: e.__dict__ for e in profCPU.key_averages()}).T
                # profiling_cpu_df.to_csv(f"Profiling/{str(starting)}_{model.name}_{device.split(':')[0]}_(CPU)_{tensor.size()[2]}-{tensor.size()[3]}_exp-{exp_count}_batch-{batch_size}_param-{params}.csv")

        return np.mean(durations)
    starting = time.time()
    for exp in range(1, experiment_count + 1):
        print("\t[EXP {0:4d}]".format(exp), end="")
        # model = UNet(1, 1).to(dev)
        control = False
        for tensor in tensors:
            #torch.cuda.synchronize()

            #modelGPU = VisionTransformer(get_r50_b16_configS(), img_size=256).to("cuda:0")# flexivitGPU #
            modelGPU = PupilNet().to("cuda:0") #NestedUNetS().to("cuda:0") #UNetCCSDG_S().to("cuda:0") #EGEUNet(input_channels=1).to("cuda:0")#deeplabv3_resnet50().to("cuda:0")#PSPNet(device="cuda:0").to("cuda:0") #NestedUNetS().to("cuda:0")#PPMobileSeg(1, MobileSeg_TinyS("cuda:0")).to("cuda:0") #SegNext("cuda:0").to("cuda:0") #PPMobileSegGPU#flexivitGPU #UNetS(i, 1).to("cuda:0")
            #modelCPU = VisionTransformer(get_r50_b16_configS(), img_size=256).to("cpu")# flexivitCPU #
            modelCPU = PupilNet().to("cpu")#NestedUNetS().to("cpu")#EGEUNet(input_channels=1).to("cpu")#deeplabv3_resnet50().to("cpu")#PSPNet(device="cpu").to("cpu") #NestedUNetS().to("cpu")#PPMobileSeg(1, MobileSeg_TinyS("cpu")).to("cpu")#SegNext("cpu").to("cpu") #PPMobileSegGPU#flexivitCPU #UNetS(i, 1).to("cpu")

            model_parameters = filter(lambda p: p.requires_grad, modelGPU.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            #print("MODEL NAME :  " + str(model))
            print("MODEL PARM :  " + str(params))
            durationGPU = calculate(modelGPU, tensor, "cuda:0", loop_count, exp, batch_size, params, starting)
            durationCPU = calculate(modelCPU, tensor, "cpu", loop_count, exp, batch_size, params, starting)
            if control:
                print("\t\t\t", end="")
            else:
                control = True
            print(
                "{0}{1}\tInput Size: {2:15s}\tGPU Prediction Duration: {3:8f} ({5:8f})\tCPU Prediction Duration: {4:8f} ({6:8f})\tparams: {7}".
                format("", "", str(tensor.shape), durationGPU, durationCPU, durationGPU / i, durationCPU / i, params))


for i in range(1, 2, 2):
    print("\nBATCH SIZE: {0:5d}\tAbout Initialization Duration: {1:8f}".format(i, i * 0.002))
    calculate_prediction_durations(first_tensor=torch.randn((i, 1, 128, 128)), second_tensor=torch.randn((i, 1, 128, 128)),
                                   batch_size=i)

