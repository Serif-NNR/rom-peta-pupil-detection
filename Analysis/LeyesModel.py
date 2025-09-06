
from LEyesEvaluate.models import PupilNet
import torch

model = PupilNet().to("cuda:0")
tsr = torch.zeros(1, 1, 128, 160, device="cuda:0")
_ = model(tsr)
