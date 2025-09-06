from src.PupilDetector.Helpers.constants import *
from src.PupilDetector.Helpers.filters_and_methods import get_ellipse_parameters
from src.PupilDetector.Learning_Model.learning_model import LearningModel
from src.PupilDetector.Traditional_Method.traditional_method import TraditionalMethod
import time, torch
import numpy as np

class xStare_Detector(object):

    def __init__(self,
                 traditional_patch_type=Traditional_Patch_Type.EXTENDED,
                 input_resolution=(320,240),
                 model_device_type=Model_Device_Type.CUDA,
                 model_parameter_path="FPD_6_l0.018_d0.963_vd0.902.pt"):
        self.traditional_method = TraditionalMethod(patch_type=traditional_patch_type, resolution=input_resolution)
        self.learning_model = LearningModel(parameter_path=model_parameter_path, device_type=model_device_type)

        self.detector_freq_list, self.last_detection_time, self.freq_list_counter = [], 0, 0

        for i in range(5): _ = self.detect_without_rom(np.zeros((240, 320), dtype=np.uint8))


    def _measure_detection_freq(self):
        self.detector_freq_list.append(time.time() - self.last_detection_time)
        self.last_detection_time = time.time()
        if len(self.detector_freq_list) > 1000:
            print("DETECTOR: ", self.freq_list_counter, " -> ", sum(self.detector_freq_list) / len(self.detector_freq_list))
            self.detector_freq_list = []
            self.freq_list_counter += 1


    def set_traditional_configuration(self, pfx, pfy, thm):
        self.traditional_method.pfx = pfx
        self.traditional_method.pfy = pfy
        self.traditional_method.thm = thm
        self.last_detection_time = time.time()

    def get_traditional_configuration(self):
        return [self.traditional_method.pfx, self.traditional_method.pfy, self.traditional_method.thm]


    def detect_with_rom(self, image):
        data = get_detector_data()
        data["timestamp_start"] = time.time()

        start_x, end_x, start_y, end_y, thx, thy, d = self.traditional_method.get_patch(image)
        map = self.learning_model.segment(image[start_y: end_y, start_x: end_x])
        ellipse = get_ellipse_parameters(map)
        data["start_x"], data["end_x"], data["start_y"], data["end_y"], data["thx"], data["thy"] = \
            start_x, end_x, start_y, end_y, thx, thy
        data["timestamp_end"] = time.time()
        self._measure_detection_freq()
        return (start_x, end_x, start_y, end_y, thx, thy, d, map, ellipse)

    def detect_without_rom(self, image):
        data = get_detector_data()
        data["timestamp_start"] = time.time()
        map = self.learning_model.segment(image)
        ellipse = get_ellipse_parameters(map)
        data["timestamp_end"] = time.time()
        self._measure_detection_freq()
        return (None, None, None, None, None, None, None, map, ellipse)



if __name__ == "__main__":
    detector = xStare_Detector()
    print(detector)