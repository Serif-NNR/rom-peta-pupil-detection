import torch
from src.PupilDetector.Helpers.constants import Model_Device_Type
from src.PupilDetector.Learning_Model.Models.torch_UnetPP import *
from src.PupilDetector.Helpers.model_state_loader import State
import numpy as np
from threading import Lock, Semaphore
import os
import torch.nn as nn

model_checkpoint_path_full = "/src/PupilDetector/Learning_Model"


class LearningModel(object):

    model = None

    def __init__(self, parameter_path, device_type, num_classes=1, input_channels=1):
        self.parameter_path, self.num_classes, self.input_channel = \
            os.getcwd() + model_checkpoint_path_full +"/Model_Parameters/" + parameter_path, num_classes, input_channels
        self.device = "cpu"
        if device_type == Model_Device_Type.CUDA:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._init_model()

        self.mtx = Semaphore(364)

    def _init_model(self):
        #self.model = NestedUNetS(num_classes=self.num_classes, input_channels=self.input_channel).to(self.device)
        #self.model.load_parameters_from_pt_file(self.parameter_path)
        self.model = torch.load(self.parameter_path)
        #self.model = State.load(self.parameter_path, NestedUNet).model.to(self.device)
        temp_tensor = torch.randn((1, 1, 320, 240)).to(self.device)

        torch.set_grad_enabled(False)
        #torch.backends.cuda.matmul.allow_tf32 = True
        for param in self.model.parameters():
            #pass
            param.grad = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) and False:
                #module.bias = None
                module.training = False

        for _ in range(50): _ = self.model(temp_tensor)

    def segment(self, image):
        if image.shape[0] < 10 or image.shape[1] < 10: return np.zeros(image.shape, dtype=np.uint8)
        #torch.cuda.empty_cache()
        image_expanded = np.expand_dims(image, axis=-1).transpose((2, 0, 1))
        image_tensor = torch.tensor(image_expanded.astype(np.float32) / 255.).unsqueeze(0)
        with self.mtx:
            #with torch.inference_mode():
                try:
                    segmentation_map = self.model(image_tensor.to(self.device))
                    map = (segmentation_map.detach().cpu().numpy()[0, 0, :, :] * 255).astype(np.uint8)
                    return map
                except: return np.zeros(image.shape, dtype=np.uint8)



if __name__ == "__main__":
    #model_checkpoint_path_full = ""
    #model = LearningModel(parameter_path="xstare_nesteds.pth.tar", device_type=Model_Device_Type.CUDA)

    model = torch.load("X:\\Fixein Pupil Detection\\ModelParameters\\1695223462_torch_UNetPPS_AV_2005806\\FPD_6_l0.018_d0.963_vd0.902.pt")

    print(model)
