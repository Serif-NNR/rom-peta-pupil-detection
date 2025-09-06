import torchsummary, torch, math
from LEyesEvaluate.LEyes_evaluate import PupilNet
from segmodel import UNet
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import pandas as pd
import torchvision.models as models
from torch_UnetPP import NestedUNet, NestedUNetS
import segmentation_models_pytorch as smp

def convert_size(size_bytes):
    # https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])



model = PupilNet().to("cuda:0")
input = torch.randn(1, 1, 128, 128).to("cuda:0")
model =smp.Unet(

            in_channels=1,
            classes=1
        ).to("cuda:0")
#model = NestedUNetS().to("cuda:0")

#torchsummary.summary(model, (3, 224, 224))
#for _ in range(10): a = model(input)

torch.cuda.reset_max_memory_allocated("cuda:0")
start_memory = torch.cuda.memory_allocated()
if True: #with torch.no_grad():
    with profile(profile_memory=True) as profGPU:
        b = model(input)
    end_memory = torch.cuda.memory_allocated("cuda:0")

    memory_allocated = torch.cuda.memory_allocated("cuda:0")
    memory_cached = torch.cuda.memory_cached("cuda:0")

    print(f"Modelin GPU Bellek Kullanımı: {memory_allocated / (1024 ** 3):.2f} GB")
    print(f"GPU Bellek Önbellek Boyutu: {memory_cached / (1024 ** 3):.2f} GB")

    memory_used = end_memory - start_memory

    print(convert_size(memory_used))
    profiling_gpu_df = pd.DataFrame({e.key: e.__dict__ for e in profGPU.key_averages()}).T
    list_gpu_mem = np.asarray(profiling_gpu_df["cuda_memory_usage"].tolist(), dtype=float)
    nonzeros = list_gpu_mem[list_gpu_mem > 0]
    print("gpu mem byte: ", convert_size(nonzeros.sum()))
    list_cpu_mem = np.asarray(profiling_gpu_df["cpu_memory_usage"].tolist(), dtype=float)
    nonzeros = list_cpu_mem[list_cpu_mem > 0]
    print("cpu mem byte: ", convert_size(nonzeros.sum()))
    print("OK")