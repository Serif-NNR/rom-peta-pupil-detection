import torch
from torch import nn

class ILearningModel(nn.Module):
    name = "Unknown Learning Model"

    def load_parameters_from_pt_file(self, parameter_path):
        self.load_state_dict(torch.load(parameter_path))
        self.eval()